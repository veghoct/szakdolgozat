include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

# Sensitivity of the model to utility_beta: sweep utility_beta across
# [0, 0.4433] (benchmark 0.1433 +/- 0.3, clamped at 0 since beta isn't
# meaningful below zero) with 20 sweep points, keep every other parameter
# at its benchmark value, and average the equilibrium-window outcome over
# 30 seeded replicates (seeds 0..29) at each sweep point.
#
# The full [0, 1] range is not used: unlike alpha, beta has no formula
# degeneracy at either endpoint, but equilibrium outcomes get visibly
# noisier for utility_beta >~ 0.9 (likely a real feedback effect, since the
# zone-status term beta weights is itself computed from whichever residents
# currently live there). The sweep is narrowed to a window centered on the
# benchmark to stay clear of that high-variance region.

function run_once(run_id, utility_beta; steps = 150)
    districts = DataFrame(
        id = [1, 2, 3, 4, 5, 6],
        zone_id = [1, 1, 2, 2, 3, 3],
        name = [
            "1. zóna - Hétköznapi",
            "1. Zóna - Luxus",
            "2. zóna - Hétköznapi",
            "2. Zóna - Luxus",
            "3. zóna - Hétköznapi",
            "3. Zóna - Luxus",
        ],
        amenity = [
            1.30627197265625,
            5.25037841796875,
            1.30538818359375,
            3.57867919921875,
            1,
            3.92484619140625,
        ],
        rent = [
            100,
            100,
            100,
            100,
            120,
            100,
        ],
        units = [
            142,
            151,
            363,
            117,
            170,
            20,
        ],
    )

    utility_alpha = 0.4026
    price_change = 0.0980
    affordability_rate = 1
    number_of_households = 799

    model = SegregationModel(
            lognormal_distribution,
            utility_alpha,
            utility_beta,
            price_change,
            affordability_rate,
            number_of_households,
            districts,
            run_id
        )

    run_for!(model, steps)

    district_data = copy(model.district_data)
    model_data = copy(model.model_data)

    district_data.run_id = fill(run_id, nrow(district_data))
    model_data.run_id = fill(run_id, nrow(model_data))

    return (district_data = district_data, model_data = model_data)
end

# Segregation indices are NaN when a zone is missing an income group entirely
# (the index is undefined, not zero); skip those rather than letting mean()
# propagate a single NaN sample into the eq-window average.
nanmean(xs) = (valid = filter(!isnan, xs); isempty(valid) ? NaN : mean(valid))

steps = 1000
n_sweep_points = 20
betas = collect(range(max(0.0, 0.1433 - 0.3), 0.1433 + 0.3, length = n_sweep_points))
replicate_seeds = 0:29
district_ids = 1:6

rent_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
vacancy_rate_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
dissimilarity_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))
exposure_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))

Threads.@threads for sweep_idx in 1:n_sweep_points
    beta = betas[sweep_idx]

    for (rep_idx, seed) in enumerate(replicate_seeds)
        result = run_once(seed, beta; steps = steps)

        district_eq = filter(:step => step -> step > maximum(result.district_data.step) - 50, result.district_data)
        model_eq = filter(:step => step -> step > maximum(result.model_data.step) - 50, result.model_data)

        district_means = sort(
            combine(groupby(district_eq, :id),
                :rent => mean => :rent_mean,
                :vacancy_rate => mean => :vacancy_rate_mean,
            ),
            :id,
        )

        rent_by_replicate[sweep_idx, rep_idx, :] = district_means.rent_mean
        vacancy_rate_by_replicate[sweep_idx, rep_idx, :] = district_means.vacancy_rate_mean
        dissimilarity_by_replicate[sweep_idx, rep_idx] = nanmean(model_eq.dissimilarity_three_groups)
        exposure_by_replicate[sweep_idx, rep_idx] = nanmean(model_eq.exposure_three_groups)
    end
end

rent_by_beta = DataFrame(beta = Float64[], id = Int[], rent = Float64[])
vacancy_rate_by_beta = DataFrame(beta = Float64[], id = Int[], vacancy_rate = Float64[])

for sweep_idx in 1:n_sweep_points
    for (d_idx, id) in enumerate(district_ids)
        push!(rent_by_beta, (beta = betas[sweep_idx], id = id, rent = mean(rent_by_replicate[sweep_idx, :, d_idx])))
        push!(vacancy_rate_by_beta, (beta = betas[sweep_idx], id = id, vacancy_rate = mean(vacancy_rate_by_replicate[sweep_idx, :, d_idx])))
    end
end

dissimilarity_by_beta = DataFrame(
    beta = betas,
    dissimilarity_three_groups = [nanmean(dissimilarity_by_replicate[i, :]) for i in 1:n_sweep_points],
)

exposure_by_beta = DataFrame(
    beta = betas,
    exposure_three_groups = [nanmean(exposure_by_replicate[i, :]) for i in 1:n_sweep_points],
)

println("Equilibrium rent by utility_beta (columns = district id):")
show(unstack(rent_by_beta, :beta, :id, :rent), allrows = true, allcols = true)
println()

println("Equilibrium vacancy rate by utility_beta (columns = district id):")
show(unstack(vacancy_rate_by_beta, :beta, :id, :vacancy_rate), allrows = true, allcols = true)
println()

println("Equilibrium dissimilarity (three groups, citywide) by utility_beta:")
show(dissimilarity_by_beta, allrows = true)
println()

println("Equilibrium exposure (three groups, citywide) by utility_beta:")
show(exposure_by_beta, allrows = true)
println()

mkpath(joinpath(@__DIR__, "beta"))

plot(
    rent_by_beta.beta,
    rent_by_beta.rent,
    group = rent_by_beta.id,
    xlabel = "utility_beta",
    ylabel = "Equilibrium rent",
    title = "Equilibrium rent vs utility_beta",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "beta", "rent.svg"))

plot(
    vacancy_rate_by_beta.beta,
    vacancy_rate_by_beta.vacancy_rate,
    group = vacancy_rate_by_beta.id,
    xlabel = "utility_beta",
    ylabel = "Equilibrium vacancy rate",
    title = "Equilibrium vacancy rate vs utility_beta",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "beta", "vacancy_rate.svg"))

plot(
    dissimilarity_by_beta.beta,
    dissimilarity_by_beta.dissimilarity_three_groups,
    xlabel = "utility_beta",
    ylabel = "Equilibrium dissimilarity (three groups)",
    title = "Equilibrium dissimilarity vs utility_beta",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "beta", "dissimilarity.svg"))

plot(
    exposure_by_beta.beta,
    exposure_by_beta.exposure_three_groups,
    xlabel = "utility_beta",
    ylabel = "Equilibrium exposure (three groups)",
    title = "Equilibrium exposure vs utility_beta",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "beta", "exposure.svg"))
