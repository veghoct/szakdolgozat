include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

# Sensitivity of the model to a single district's unit count (capacity):
# for one district d at a time, sweep d's unit count across
# [0.75, 1.25] × d's benchmark units (20 sweep points, rounded to the
# nearest integer), holding every other district at its benchmark unit
# count and every other model parameter at its benchmark value. Repeated
# once per district. At each sweep point, the equilibrium-window outcome
# is averaged over 30 seeded replicates (seeds 0..29).

const BENCHMARK_UNITS = [142, 151, 363, 117, 170, 20]

function run_once(run_id, target_district_idx, target_units; steps = 150)
    units = copy(BENCHMARK_UNITS)
    units[target_district_idx] = target_units

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
        units = units,
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
replicate_seeds = 0:29
district_ids = 1:6

mkpath(joinpath(@__DIR__, "district_units_per_district"))

for target_district_idx in 1:6
    target_district_id = district_ids[target_district_idx]
    benchmark_units_d = BENCHMARK_UNITS[target_district_idx]
    unit_values = [round(Int, u) for u in range(0.75 * benchmark_units_d, 1.25 * benchmark_units_d, length = n_sweep_points)]

    rent_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
    vacancy_rate_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
    dissimilarity_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))
    exposure_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))

    Threads.@threads for sweep_idx in 1:n_sweep_points
        target_units = unit_values[sweep_idx]

        for (rep_idx, seed) in enumerate(replicate_seeds)
            result = run_once(seed, target_district_idx, target_units; steps = steps)

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

    rent_by_units = DataFrame(units = Int[], id = Int[], rent = Float64[])
    vacancy_rate_by_units = DataFrame(units = Int[], id = Int[], vacancy_rate = Float64[])

    for sweep_idx in 1:n_sweep_points
        for (d_idx, id) in enumerate(district_ids)
            push!(rent_by_units, (units = unit_values[sweep_idx], id = id, rent = mean(rent_by_replicate[sweep_idx, :, d_idx])))
            push!(vacancy_rate_by_units, (units = unit_values[sweep_idx], id = id, vacancy_rate = mean(vacancy_rate_by_replicate[sweep_idx, :, d_idx])))
        end
    end

    # unique(...; keep = :first) drops the exact-duplicate rows produced when
    # [0.75, 1.25] × benchmark units spans fewer distinct integers than
    # n_sweep_points (small-unit districts); every duplicate has identical
    # units and identical replicate seeds, hence identical output.
    dissimilarity_by_units = unique(DataFrame(
        units = unit_values,
        dissimilarity_three_groups = [nanmean(dissimilarity_by_replicate[i, :]) for i in 1:n_sweep_points],
    ), :units)

    exposure_by_units = unique(DataFrame(
        units = unit_values,
        exposure_three_groups = [nanmean(exposure_by_replicate[i, :]) for i in 1:n_sweep_points],
    ), :units)

    println("=== District $target_district_id unit count sweep ===")

    # unit_values can contain duplicates when [0.75, 1.25] × benchmark units
    # spans fewer distinct integers than n_sweep_points (small-unit
    # districts); duplicate rows are exact repeats (same rounded units, same
    # replicate seeds => identical output), so combine = mean is a no-op.
    println("Equilibrium rent vs district $target_district_id unit count (columns = district id):")
    show(unstack(rent_by_units, :units, :id, :rent, combine = mean), allrows = true, allcols = true)
    println()

    println("Equilibrium vacancy rate vs district $target_district_id unit count (columns = district id):")
    show(unstack(vacancy_rate_by_units, :units, :id, :vacancy_rate, combine = mean), allrows = true, allcols = true)
    println()

    println("Equilibrium dissimilarity (three groups, citywide) vs district $target_district_id unit count:")
    show(dissimilarity_by_units, allrows = true)
    println()

    println("Equilibrium exposure (three groups, citywide) vs district $target_district_id unit count:")
    show(exposure_by_units, allrows = true)
    println()

    plot(
        rent_by_units.units,
        rent_by_units.rent,
        group = rent_by_units.id,
        xlabel = "District $target_district_id unit count",
        ylabel = "Equilibrium rent",
        title = "Equilibrium rent vs district $target_district_id unit count",
        size = (900, 500),
    )
    savefig(joinpath(@__DIR__, "district_units_per_district", "d$(target_district_id)_rent.svg"))

    plot(
        vacancy_rate_by_units.units,
        vacancy_rate_by_units.vacancy_rate,
        group = vacancy_rate_by_units.id,
        xlabel = "District $target_district_id unit count",
        ylabel = "Equilibrium vacancy rate",
        title = "Equilibrium vacancy rate vs district $target_district_id unit count",
        size = (900, 500),
    )
    savefig(joinpath(@__DIR__, "district_units_per_district", "d$(target_district_id)_vacancy_rate.svg"))

    plot(
        dissimilarity_by_units.units,
        dissimilarity_by_units.dissimilarity_three_groups,
        xlabel = "District $target_district_id unit count",
        ylabel = "Equilibrium dissimilarity (three groups)",
        title = "Equilibrium dissimilarity vs district $target_district_id unit count",
        size = (900, 500),
        legend = false,
    )
    savefig(joinpath(@__DIR__, "district_units_per_district", "d$(target_district_id)_dissimilarity.svg"))

    plot(
        exposure_by_units.units,
        exposure_by_units.exposure_three_groups,
        xlabel = "District $target_district_id unit count",
        ylabel = "Equilibrium exposure (three groups)",
        title = "Equilibrium exposure vs district $target_district_id unit count",
        size = (900, 500),
        legend = false,
    )
    savefig(joinpath(@__DIR__, "district_units_per_district", "d$(target_district_id)_exposure.svg"))
end
