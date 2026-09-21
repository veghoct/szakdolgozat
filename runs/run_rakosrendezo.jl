include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

function run_once(run_id; steps = 5000)
    districts = DataFrame(
        id = [1, 2, 3, 4, 5, 6, 7],
        zone_id = [1, 1, 2, 2, 3, 3, 2],
        name = [
            "1. zóna - Hétköznapi",
            "1. Zóna - Luxus",
            "2. zóna - Hétköznapi",
            "2. Zóna - Luxus",
            "3. zóna - Hétköznapi",
            "3. Zóna - Luxus",
            "2. Zóna - Rákosrendező",
        ],
        amenity = [
            1.30627197265625,
            5.25037841796875,
            1.30538818359375,
            3.57867919921875,
            1,
            3.92484619140625,
            5.25037841796875,
        ],
        rent = [
            100,
            100,
            100,
            100,
            120,
            100,
            100
        ],
        units = [
            142,
            151,
            363,
            117,
            170,
            20,
            50
        ],
    )

    utiltiy_alpha = 0.4026
    utility_beta = 0.1433
    price_change = 0.0980

    affordability_rate = 1
    number_of_households = 799

    model = SegregationModel(
            lognormal_distribution,
            utiltiy_alpha,
            utility_beta,
            price_change,
            affordability_rate,
            number_of_households,
            districts,
            run_id
        )

    run_for!(model, steps)

    district_data = copy(model.district_data)
    zone_data = copy(model.zone_data)
    model_data = copy(model.model_data)

    district_data.run_id = fill(run_id, nrow(district_data))
    zone_data.run_id = fill(run_id, nrow(zone_data))
    model_data.run_id = fill(run_id, nrow(model_data))

    return (district_data = district_data, zone_data = zone_data, model_data = model_data)
end

runs = 100
steps = 150

district_data_by_run = Vector{DataFrame}(undef, runs)
zone_data_by_run = Vector{DataFrame}(undef, runs)
model_data_by_run = Vector{DataFrame}(undef, runs)
Threads.@threads for i in 0:(runs - 1)
    result = run_once(i; steps = steps)
    district_data_by_run[i + 1] = result.district_data
    zone_data_by_run[i + 1] = result.zone_data
    model_data_by_run[i + 1] = result.model_data
end
district_data = vcat(district_data_by_run...)
zone_data = vcat(zone_data_by_run...)
model_data = vcat(model_data_by_run...)

ci_lower(x) = quantile(x, 0.1)
ci_upper(x) = quantile(x, 0.9)

# Segregation indices are NaN when a zone is missing an income group entirely
# (the index is undefined, not zero); skip those rather than letting mean()
# propagate a single NaN sample into the whole average.
nanmean(xs) = (valid = filter(!isnan, xs); isempty(valid) ? NaN : mean(valid))

rent_by_market = combine(groupby(district_data, [:step, :id]), 
    :rent => mean => :rent_mean,
    :rent => ci_lower => :rent_lower,
    :rent => ci_upper => :rent_upper
)

vacancy_rate_by_market = combine(groupby(district_data, [:step, :id]), 
    :vacancy_rate => mean => :vacancy_rate_mean,
    :vacancy_rate => ci_lower => :vacancy_rate_lower,
    :vacancy_rate => ci_upper => :vacancy_rate_upper
)

district_data_eq_state_steps = filter(:step => step -> step > maximum(district_data.step) - 50, district_data)
zone_data_eq_state_steps = filter(:step => step -> step > maximum(zone_data.step) - 50, zone_data)
model_data_eq_state_steps = filter(:step => step -> step > maximum(model_data.step) - 50, model_data)

#println(combine(groupby(district_data_eq_state_steps, [:id]), :rent => mean, :rent => ci_lower, :rent => ci_upper))
#println(combine(groupby(district_data_eq_state_steps, [:id]), :vacancy_rate => mean, :vacancy_rate => ci_lower, :vacancy_rate => ci_upper))

println("Mean income stats by zone:")
println(combine(
    groupby(zone_data_eq_state_steps, :id),
    :average_income => nanmean,
    :median_income => nanmean,
    :gini => nanmean,
    :share_bottom_10 => nanmean,
    :share_bottom_20 => nanmean,
    :share_bottom_50 => nanmean,
    :share_top_20 => nanmean,
    :share_top_10 => nanmean,
))

index_cols = names(model_data_eq_state_steps, Not([:step, :run_id]))

println("Mean segregation indices (citywide):")
println(combine(model_data_eq_state_steps, [col => nanmean => col for col in index_cols]...))

exit()

plot(
    vacancy_rate_by_market.step,
    vacancy_rate_by_market.vacancy_rate_mean,
    group = vacancy_rate_by_market.id,
    xlabel = "Step",
    ylabel = "Vacancy rate",
    title = "Vacancy rate over time",
    size = (900, 500),
)

savefig("vacancy_rate.svg")

plot(
    rent_by_market.step,
    rent_by_market.rent_mean,
    group = rent_by_market.id,
    xlabel = "Step",
    ylabel = "Rent",
    title = "Rent over time",
    size = (900, 500),
)

savefig("rent.svg")