include("../util/imports.jl")
include("interfaces.jl")
include("model.jl")

# Avoid evaluating inverse CDF formulas exactly at 0 or 1
function open_unit_random(rng)
    return clamp(rand(rng), nextfloat(0.0), prevfloat(1.0))
end


function lognormal_distribution(rng)
    u = open_unit_random(rng)

    mu = 12.584991083577634
    sigma = 0.5457737566188

    income = exp(mu + sigma * quantile(Normal(), u))

    return round(income / 1000)
end


function dagum_distribution(rng)
    u = open_unit_random(rng)

    a = 3.6096943927208094
    b = 402994.56509825075
    p = 0.7713443751258857

    income = b * (u^(-1 / p) - 1)^(-1 / a)

    return round(income / 1000)
end


function singh_maddala_distribution(rng)
    u = open_unit_random(rng)

    a = 2.9793276391217565
    b = 410377.5624011284
    q = 1.3168477922529758

    income = b * ((1 - u)^(-1 / q) - 1)^(1 / a)

    return round(income / 1000)
end


function gb2_distribution(rng)
    u = open_unit_random(rng)

    a = 3.1062231709020507
    b = 408872.1438747776
    p = 0.9429998373808317
    q = 1.238925941658504

    z = quantile(Beta(p, q), u)
    income = b * (z / (1 - z))^(1 / a)

    return round(income / 1000)
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
            3.4,
            4,
            3,
            3.6,
            2.9,
            3.2,
        ],
        rent = [
            100,
            100,
            100,
            100,
            100,
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

    utiltiy_alpha = 0.8
    utility_beta = 0.14
    price_change = 0.098 #yearly

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

runs = 10
steps = 150

all_runs = vcat([run_once(i; steps = steps) for i in 0:(runs - 1)]...)
last_100_steps = vcat([run_once(i; steps = steps) for i in 0:(runs - 1)]...) |> filter(:step => step -> step > maximum(all_runs.step) - 100)

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