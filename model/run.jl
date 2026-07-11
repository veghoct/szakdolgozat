include("../util/imports.jl")
include("interfaces.jl")
include("model.jl")

function lognormal_distribution(rng)
    u = rand(rng)
    mu = 12.584991083577634
    sigma = 0.5457737566188

    return round(exp(mu + sigma * quantile(Normal(), u)) / 1000)
end

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
            170,
            600,
            130,
            400,
            150,
            560
        ],
        rent = [
            100,
            100,
            100,
            100,
            100,
            100
        ],
        units = [
            215,
            74,
            182,
            8,
            464,
            16
        ],
    )

    utiltiy_alpha = 0.40
    utility_beta = 0.14
    price_change = 0.098

    minimum_disposable_income = 0
    number_of_households = 799

    model = SegregationModel(
            lognormal_distribution,
            utiltiy_alpha,
            utility_beta,
            price_change,
            minimum_disposable_income,
            number_of_households,
            districts
        )

    run_for!(model, steps)

    agent_data = copy(model.district_data)
    agent_data.run_id = fill(run_id, nrow(agent_data))

    return agent_data
end

runs = 100
steps = 1000

all_runs = vcat([run_once(i; steps = steps) for i in 0:(runs - 1)]...)
last_100_steps = vcat([run_once(i; steps = steps) for i in 0:(runs - 1)]...) |> filter(:step => step -> step > 900)

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

println(combine(groupby(last_100_steps, [:id]), :rent => mean, :rent => ci_lower, :rent => ci_upper))
println(combine(groupby(last_100_steps, [:id]), :vacancy_rate => mean, :vacancy_rate => ci_lower, :vacancy_rate => ci_upper))

#plot(
    #vacancy_rate_by_market.step,
    #vacancy_rate_by_market.vacancy_rate_mean,
    #group = vacancy_rate_by_market.id,
    #xlabel = "Step",
    #ylabel = "Vacancy rate",
    #title = "Vacancy rate over time",
    #size = (900, 500),
#)
#
#savefig("vacancy_rate.svg")
#
#plot(
    #rent_by_market.step,
    #rent_by_market.rent_mean,
    #group = rent_by_market.id,
    #xlabel = "Step",
    #ylabel = "Rent",
    #title = "Rent over time",
    #size = (900, 500),
#)
#
#savefig("rent.svg")