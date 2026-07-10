include("../util/imports.jl")

# Source: https://www.ksh.hu/stadat_files/jov/hu/jov0005.html
annual_income_per_person = [564_041, 881_304, 1_057_318, 1_305_526, 1_459_115, 1_666_901, 1_899_253, 2_150_731, 2_562_414, 4_120_089]

# Source: https://www.ksh.hu/stadat_files/jov/hu/jov0022.html
household_size = [3.0, 3.0, 2.9, 2.6, 2.4, 2.4, 2.1, 2.1, 2.0, 1.7]

national_budapest_average_income_ratio = [ 	]

# Since the ABM is presuming that a residential agent is a household, we want to work with per household data
# not per person data
monthly_income_per_household = annual_income_per_person .* 2.3 ./ 12
monthly_income_per_household = round.(Int, monthly_income_per_household)

# -----------------------------
# Distribution fitting
# -----------------------------

function interval_mean_from_quantile(qfun, θ, lo, hi; ngrid = 500)
    s = 0.0

    for k in 1:ngrid
        # midpoint rule, so it never evaluates exactly at u = 0 or u = 1
        u = lo + (k - 0.5) * (hi - lo) / ngrid
        val = qfun(u, θ)

        if !isfinite(val) || val <= 0
            return Inf
        end

        s += val
    end

    return s / ngrid
end

function model_decile_means(qfun, θ; ngrid = 500)
    return [
        interval_mean_from_quantile(qfun, θ, (j - 1) / 10, j / 10; ngrid = ngrid)
        for j in 1:10
    ]
end

function log_objective(y_obs, y_hat)
    if any(!isfinite, y_hat) || any(y_hat .<= 0)
        return 1e12
    end

    return sum((log.(y_obs) .- log.(y_hat)).^2)
end

function multistart_optimize(objective, starts)
    best_result = nothing
    best_value = Inf

    for start in starts
        result = optimize(
            objective,
            start,
            NelderMead(),
            Optim.Options(
                iterations = 50_000,
                f_abstol = 1e-10,
                x_abstol = 1e-10
            )
        )

        value = Optim.minimum(result)

        if value < best_value
            best_value = value
            best_result = result
        end
    end

    return best_result
end

# -----------------------------
# Dagum fitting
# -----------------------------

function dagum_quantile(u, θ)
    a, b, p = θ

    return b * (u^(-1 / p) - 1)^(-1 / a)
end

function fit_dagum_decile_means(y_obs)
    function unpack(rawθ)
        a = 1.0 + exp(rawθ[1])   # ensures a > 1, finite mean
        b = exp(rawθ[2])
        p = exp(rawθ[3])

        return [a, b, p]
    end

    function objective(rawθ)
        θ = unpack(rawθ)
        y_hat = model_decile_means(dagum_quantile, θ)

        return log_objective(y_obs, y_hat)
    end

    starts = [
        [log(1.5 - 1), log(median(y_obs)), log(0.5)],
        [log(2.0 - 1), log(median(y_obs)), log(1.0)],
        [log(3.0 - 1), log(median(y_obs)), log(1.0)],
        [log(2.0 - 1), log(mean(y_obs)), log(2.0)],
        [log(5.0 - 1), log(mean(y_obs)), log(1.0)]
    ]

    result = multistart_optimize(objective, starts)

    θ_hat = unpack(Optim.minimizer(result))
    y_hat = model_decile_means(dagum_quantile, θ_hat)

    return (
        distribution = "Dagum",
        a = θ_hat[1],
        b = θ_hat[2],
        p = θ_hat[3],
        fitted = y_hat,
        objective = Optim.minimum(result),
        rmse_log = sqrt(mean((log.(y_obs) .- log.(y_hat)).^2))
    )
end

# -----------------------------
# Burr fitting
# -----------------------------

function singh_maddala_quantile(u, θ)
    a, b, q = θ

    return b * ((1 - u)^(-1 / q) - 1)^(1 / a)
end

function fit_singh_maddala_decile_means(y_obs)
    function unpack(rawθ)
        a = exp(rawθ[1])
        b = exp(rawθ[2])

        # ensures a * q > 1, finite mean
        q = (1.0 + exp(rawθ[3])) / a

        return [a, b, q]
    end

    function objective(rawθ)
        θ = unpack(rawθ)
        y_hat = model_decile_means(singh_maddala_quantile, θ)

        return log_objective(y_obs, y_hat)
    end

    starts = [
        [log(1.5), log(median(y_obs)), log(1.5 * 1.5 - 1)],
        [log(2.0), log(median(y_obs)), log(2.0 * 1.0 - 1)],
        [log(2.0), log(mean(y_obs)), log(2.0 * 2.0 - 1)],
        [log(3.0), log(mean(y_obs)), log(3.0 * 1.0 - 1)],
        [log(5.0), log(mean(y_obs)), log(5.0 * 1.0 - 1)]
    ]

    result = multistart_optimize(objective, starts)

    θ_hat = unpack(Optim.minimizer(result))
    y_hat = model_decile_means(singh_maddala_quantile, θ_hat)

    return (
        distribution = "Singh-Maddala",
        a = θ_hat[1],
        b = θ_hat[2],
        q = θ_hat[3],
        fitted = y_hat,
        objective = Optim.minimum(result),
        rmse_log = sqrt(mean((log.(y_obs) .- log.(y_hat)).^2))
    )
end

# -----------------------------
# Lognormal fitting
# -----------------------------

function lognormal_interval_mean(μ, σ, lo, hi)
    N = Normal()

    zlo = lo == 0.0 ? -Inf : quantile(N, lo)
    zhi = hi == 1.0 ? Inf : quantile(N, hi)

    numerator =
        exp(μ + σ^2 / 2) *
        (cdf(N, zhi - σ) - cdf(N, zlo - σ))

    return numerator / (hi - lo)
end

function lognormal_decile_means(μ, σ)
    return [
        lognormal_interval_mean(μ, σ, (j - 1) / 10, j / 10)
        for j in 1:10
    ]
end

function fit_lognormal_decile_means(y_obs)
    function objective(rawθ)
        μ = rawθ[1]
        σ = exp(rawθ[2])

        y_hat = lognormal_decile_means(μ, σ)

        return log_objective(y_obs, y_hat)
    end

    starts = [
        [log(mean(y_obs)), log(0.2)],
        [log(mean(y_obs)), log(0.4)],
        [log(mean(y_obs)), log(0.6)],
        [log(median(y_obs)), log(0.4)],
        [log(median(y_obs)), log(0.8)]
    ]

    result = multistart_optimize(objective, starts)

    rawθ_hat = Optim.minimizer(result)

    μ_hat = rawθ_hat[1]
    σ_hat = exp(rawθ_hat[2])

    y_hat = lognormal_decile_means(μ_hat, σ_hat)

    return (
        distribution = "Lognormal",
        μ = μ_hat,
        σ = σ_hat,
        fitted = y_hat,
        objective = Optim.minimum(result),
        rmse_log = sqrt(mean((log.(y_obs) .- log.(y_hat)).^2))
    )
end

# -----------------------------
# Fit the dist
# -----------------------------

function gb2_quantile(u, θ)
    a, b, p, q = θ

    z = quantile(Beta(p, q), u)

    return b * (z / (1 - z))^(1 / a)
end

function fit_gb2_decile_means(y_obs)
    function unpack(rawθ)
        a = exp(rawθ[1])
        b = exp(rawθ[2])
        p = exp(rawθ[3])

        # ensures a * q > 1, finite mean
        q = (1.0 + exp(rawθ[4])) / a

        return [a, b, p, q]
    end

    function objective(rawθ)
        θ = unpack(rawθ)
        y_hat = model_decile_means(gb2_quantile, θ; ngrid = 300)

        return log_objective(y_obs, y_hat)
    end

    starts = [
        [log(1.5), log(median(y_obs)), log(1.0), log(1.5 * 1.5 - 1)],
        [log(2.0), log(median(y_obs)), log(1.0), log(2.0 * 1.0 - 1)],
        [log(2.0), log(mean(y_obs)), log(2.0), log(2.0 * 2.0 - 1)],
        [log(3.0), log(mean(y_obs)), log(1.0), log(3.0 * 1.0 - 1)],
        [log(5.0), log(mean(y_obs)), log(1.0), log(5.0 * 1.0 - 1)]
    ]

    result = multistart_optimize(objective, starts)

    θ_hat = unpack(Optim.minimizer(result))
    y_hat = model_decile_means(gb2_quantile, θ_hat; ngrid = 1000)

    return (
        distribution = "GB2",
        a = θ_hat[1],
        b = θ_hat[2],
        p = θ_hat[3],
        q = θ_hat[4],
        fitted = y_hat,
        objective = Optim.minimum(result),
        rmse_log = sqrt(mean((log.(y_obs) .- log.(y_hat)).^2))
    )
end

# -----------------------------
# Fit the dist
# -----------------------------

fit_ln = fit_lognormal_decile_means(monthly_income_per_household)
fit_dagum = fit_dagum_decile_means(monthly_income_per_household)
fit_sm = fit_singh_maddala_decile_means(monthly_income_per_household)
fit_gb2 = fit_gb2_decile_means(monthly_income_per_household)

fits = [fit_ln, fit_dagum, fit_sm, fit_gb2]

for fit in fits
    println()
    println(fit.distribution)
    println("RMSE log: ", fit.rmse_log)
    println("Objective: ", fit.objective)
    println("Fitted decile means:")
    println(round.(fit.fitted; digits = 0))
end

println("Actual data:")
println(monthly_income_per_household)

# -----------------------------
# Plots
# -----------------------------

function plot_fit(fit, y_obs; savepath=nothing)
    d = 1:10

    p = bar(d, y_obs;
        color=:red, alpha=0.45, label="Observed",
        xlabel="Decile", ylabel="Income",
        title=fit.distribution,
        xticks=d
    )

    plot!(p, d, fit.fitted;
        lw=3, marker=:circle, label="Fitted"
    )

    savepath !== nothing && savefig(p, savepath)
    return p
end

function q_lognormal(u, fit)
    exp(fit.μ + fit.σ * quantile(Normal(), u))
end

function q_dagum(u, fit)
    fit.b * (u^(-1 / fit.p) - 1)^(-1 / fit.a)
end

function q_singh_maddala(u, fit)
    fit.b * ((1 - u)^(-1 / fit.q) - 1)^(1 / fit.a)
end

function q_gb2(u, fit)
    z = quantile(Beta(fit.p, fit.q), u)
    fit.b * (z / (1 - z))^(1 / fit.a)
end

function fitted_quantile(u, fit)
    if fit.distribution == "Lognormal"
        return q_lognormal(u, fit)
    elseif fit.distribution == "Dagum"
        return q_dagum(u, fit)
    elseif fit.distribution == "Singh-Maddala"
        return q_singh_maddala(u, fit)
    elseif fit.distribution == "GB2"
        return q_gb2(u, fit)
    else
        error("Unknown distribution: $(fit.distribution)")
    end
end

function plot_distribution(fit, y_obs; savepath=nothing)
    u = range(0.001, 0.999; length=2000)
    x = [fitted_quantile(ui, fit) for ui in u]

    dxdu = diff(x) ./ diff(collect(u))
    x_mid = (x[1:end-1] .+ x[2:end]) ./ 2
    pdf = 1.0 ./ dxdu

    p = plot(
        x_mid, pdf;
        lw=3,
        label="Fitted density",
        xlabel="Income",
        ylabel="Density",
        title="$(fit.distribution) distribution"
    )

    vline!(p, y_obs; color=:red, alpha=0.45, lw=1.5, label=false)

    savepath !== nothing && savefig(p, savepath)
    return p
end

plot_fit(fit_ln, monthly_income_per_household, savepath=".figures/income/ln.svg")
plot_fit(fit_dagum, monthly_income_per_household, savepath=".figures/income/dagum.svg")
plot_fit(fit_sm, monthly_income_per_household, savepath=".figures/income/sm.svg")
plot_fit(fit_gb2, monthly_income_per_household, savepath=".figures/income/gb2.svg")

plot_distribution(fit_ln, monthly_income_per_household, savepath=".figures/income/ln_dist.svg")
plot_distribution(fit_dagum, monthly_income_per_household, savepath=".figures/income/dagum_dis.svg")
plot_distribution(fit_sm, monthly_income_per_household, savepath=".figures/income/sm_dist.svg")
plot_distribution(fit_gb2, monthly_income_per_household, savepath=".figures/income/gb2_dist.svg")
