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