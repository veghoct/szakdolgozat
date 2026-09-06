include("../util/imports.jl")
include("interfaces.jl")
include("model.jl")
include("distributions.jl")

function run_once(run_id; steps = 5000)
    districts = DataFrame(
        id = [1, 2, 3, 4, 5, 6],
        zone_id = [1, 1, 2, 2, 3, 3],
        name = [
            "1. zóna - Hétköznapi",
            "1. Zóna - Luxus",
            "2. zóna - Hétköznapi",
            "2. Zóna - Luxus",
            "3. zóna - Hétköznapi",
            "3. Zóna - Luxus"
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
            150,
            363,
            117,
            170,
            20,
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

    agent_data = copy(model.district_data)
    agent_data.run_id = fill(run_id, nrow(agent_data))

    return agent_data
end

runs = 100
steps = 150

results = Vector{Any}(undef, runs)
Threads.@threads for i in 0:(runs - 1)
    results[i + 1] = run_once(i; steps = steps)
end
all_runs = vcat(results...)

ci_lower(x) = quantile(x, 0.1)
ci_upper(x) = quantile(x, 0.9)

rent_by_market = combine(groupby(all_runs, [:step, :id]), 
    :rent => mean => :rent_mean,
    :rent => ci_lower => :rent_lower,
    :rent => ci_upper => :rent_upper
)

vacancy_rate_by_market = combine(groupby(all_runs, [:step, :id]), 
    :vacancy_rate => mean => :vacancy_rate_mean,
    :vacancy_rate => ci_lower => :vacancy_rate_lower,
    :vacancy_rate => ci_upper => :vacancy_rate_upper
)

equilibrium_state_steps = filter(:step => step -> step > maximum(all_runs.step) - 50, all_runs)

println(combine(groupby(equilibrium_state_steps, [:id]), :rent => mean, :rent => ci_lower, :rent => ci_upper))
println(combine(groupby(equilibrium_state_steps, [:id]), :vacancy_rate => mean, :vacancy_rate => ci_lower, :vacancy_rate => ci_upper))

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