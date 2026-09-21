include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

# Sensitivity of the model to a citywide scaling of every district's unit
# count: sweep a multiplicative scale factor across [0.75, 1.25] with 20
# sweep points. At each sweep point, every district's benchmark unit count
# is multiplied by that same factor (rounded to the nearest integer per
# district), preserving the districts' relative proportions. Every other
# parameter stays at its benchmark value (including number_of_households =
# 799), and the equilibrium-window outcome is averaged over 30 seeded
# replicates (seeds 0..29) at each sweep point.
#
# The lower end of that range is narrowed from 0.75 to 0.831: below a scale
# factor of ~0.8297 the scaled-down district table (rounded per district)
# has fewer than 799 total units, and SegregationModel requires one home per
# resident, so number_of_households = 799 would not fit. 0.831 keeps a
# small margin above that threshold. The upper bound (1.25) is unaffected.

const BENCHMARK_UNITS = [142, 151, 363, 117, 170, 20]

function run_once(run_id, scale_factor; steps = 150)
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
        units = [round(Int, u * scale_factor) for u in BENCHMARK_UNITS],
    )

    utility_alpha = 0.4026
    utility_beta = 0.1433
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
scale_factors = collect(range(0.831, 1.25, length = n_sweep_points))
replicate_seeds = 0:29
district_ids = 1:6

rent_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
vacancy_rate_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
dissimilarity_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))
exposure_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))

Threads.@threads for sweep_idx in 1:n_sweep_points
    scale_factor = scale_factors[sweep_idx]

    for (rep_idx, seed) in enumerate(replicate_seeds)
        result = run_once(seed, scale_factor; steps = steps)

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

rent_by_scale = DataFrame(scale_factor = Float64[], id = Int[], rent = Float64[])
vacancy_rate_by_scale = DataFrame(scale_factor = Float64[], id = Int[], vacancy_rate = Float64[])

for sweep_idx in 1:n_sweep_points
    for (d_idx, id) in enumerate(district_ids)
        push!(rent_by_scale, (scale_factor = scale_factors[sweep_idx], id = id, rent = mean(rent_by_replicate[sweep_idx, :, d_idx])))
        push!(vacancy_rate_by_scale, (scale_factor = scale_factors[sweep_idx], id = id, vacancy_rate = mean(vacancy_rate_by_replicate[sweep_idx, :, d_idx])))
    end
end

dissimilarity_by_scale = DataFrame(
    scale_factor = scale_factors,
    dissimilarity_three_groups = [nanmean(dissimilarity_by_replicate[i, :]) for i in 1:n_sweep_points],
)

exposure_by_scale = DataFrame(
    scale_factor = scale_factors,
    exposure_three_groups = [nanmean(exposure_by_replicate[i, :]) for i in 1:n_sweep_points],
)

println("Equilibrium rent by district-units scale factor (columns = district id):")
show(unstack(rent_by_scale, :scale_factor, :id, :rent), allrows = true, allcols = true)
println()

println("Equilibrium vacancy rate by district-units scale factor (columns = district id):")
show(unstack(vacancy_rate_by_scale, :scale_factor, :id, :vacancy_rate), allrows = true, allcols = true)
println()

println("Equilibrium dissimilarity (three groups, citywide) by district-units scale factor:")
show(dissimilarity_by_scale, allrows = true)
println()

println("Equilibrium exposure (three groups, citywide) by district-units scale factor:")
show(exposure_by_scale, allrows = true)
println()

mkpath(joinpath(@__DIR__, "district_units"))

plot(
    rent_by_scale.scale_factor,
    rent_by_scale.rent,
    group = rent_by_scale.id,
    xlabel = "District-units scale factor",
    ylabel = "Equilibrium rent",
    title = "Equilibrium rent vs district-units scale factor",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "district_units", "rent.svg"))

plot(
    vacancy_rate_by_scale.scale_factor,
    vacancy_rate_by_scale.vacancy_rate,
    group = vacancy_rate_by_scale.id,
    xlabel = "District-units scale factor",
    ylabel = "Equilibrium vacancy rate",
    title = "Equilibrium vacancy rate vs district-units scale factor",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "district_units", "vacancy_rate.svg"))

plot(
    dissimilarity_by_scale.scale_factor,
    dissimilarity_by_scale.dissimilarity_three_groups,
    xlabel = "District-units scale factor",
    ylabel = "Equilibrium dissimilarity (three groups)",
    title = "Equilibrium dissimilarity vs district-units scale factor",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "district_units", "dissimilarity.svg"))

plot(
    exposure_by_scale.scale_factor,
    exposure_by_scale.exposure_three_groups,
    xlabel = "District-units scale factor",
    ylabel = "Equilibrium exposure (three groups)",
    title = "Equilibrium exposure vs district-units scale factor",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "district_units", "exposure.svg"))
