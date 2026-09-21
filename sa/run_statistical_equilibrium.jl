include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

# Statistical equilibrium scenario: measure how many simulation steps it
# takes for the model to settle into a statistical equilibrium, i.e. the
# point at which run-to-run stochastic noise (from the matching process and
# income draws) stops shrinking. A single random parameter set is drawn
# once (with its own fixed seed, distinct from the 30 replicate seeds), then
# simulated 30 times with the standard locked replicate seeds (0..29).
# Unlike every other SA scenario, the quantity of interest here is variance,
# not an equilibrium-window average, and two complementary variance measures
# are computed: variance ACROSS the 30 replicates at each step (do the
# replicates resemble each other?), and, on a single representative
# replicate, a rolling variance WITHIN that one run's own trajectory (has
# that run itself settled into a fixed point?) -- see below.

const BENCHMARK_AMENITY = [
    1.30627197265625,
    5.25037841796875,
    1.30538818359375,
    3.57867919921875,
    1,
    3.92484619140625,
]
const BENCHMARK_UNITS = [142, 151, 363, 117, 170, 20]
const BENCHMARK_NUMBER_OF_HOUSEHOLDS = 799

# Fixed seed for the one-off random parameter draw below. Distinct from the
# 30 replicate seeds (0..29) used to run that parameter set, so the draw
# itself is reproducible independently of the replicate RNG streams.
#
# number_of_households and units are drawn independently below, so nothing
# guarantees number_of_households <= sum(units) for an arbitrary seed --
# SegregationModel requires one home per resident and will throw an
# out-of-bounds error otherwise. This fixed seed happens to draw a feasible
# set; per an explicit decision, no automatic guard is implemented, so
# changing this seed requires manually re-checking that inequality.
const PARAMETER_DRAW_SEED = 20260919

draw_rng = MersenneTwister(PARAMETER_DRAW_SEED)

amenity = [rand(draw_rng, Uniform(1, 4 * a)) for a in BENCHMARK_AMENITY]
utility_alpha = rand(draw_rng, Uniform(0, 1))
utility_beta = rand(draw_rng, Uniform(0, 1))
number_of_households = round(Int, rand(draw_rng, Uniform(0.75 * BENCHMARK_NUMBER_OF_HOUSEHOLDS, 1.25 * BENCHMARK_NUMBER_OF_HOUSEHOLDS)))
units = [round(Int, rand(draw_rng, Uniform(0.75 * u, 1.25 * u))) for u in BENCHMARK_UNITS]

price_change = 0.0980
affordability_rate = 1

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
    amenity = amenity,
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

println("Statistical equilibrium scenario — random parameter set (draw seed = $PARAMETER_DRAW_SEED):")
println("utility_alpha = $utility_alpha")
println("utility_beta = $utility_beta")
println("number_of_households = $number_of_households")
println(districts)
println()

function run_once(run_id; steps)
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
# (the index is undefined, not zero); skip those rather than letting var()
# propagate a single NaN sample into the across-replicate variance.
nanvar(xs) = (valid = filter(!isnan, xs); length(valid) < 2 ? NaN : var(valid))

steps = 1000
replicate_seeds = 0:29

district_data_by_run = Vector{DataFrame}(undef, length(replicate_seeds))
model_data_by_run = Vector{DataFrame}(undef, length(replicate_seeds))

Threads.@threads for i in eachindex(replicate_seeds)
    result = run_once(replicate_seeds[i]; steps = steps)
    district_data_by_run[i] = result.district_data
    model_data_by_run[i] = result.model_data
end

district_data = vcat(district_data_by_run...)
model_data = vcat(model_data_by_run...)

rent_variance_by_step = combine(groupby(district_data, [:step, :id]), :rent => var => :rent_variance)
vacancy_rate_variance_by_step = combine(groupby(district_data, [:step, :id]), :vacancy_rate => var => :vacancy_rate_variance)
dissimilarity_variance_by_step = combine(groupby(model_data, :step), :dissimilarity_three_groups => nanvar => :dissimilarity_three_groups_variance)
exposure_variance_by_step = combine(groupby(model_data, :step), :exposure_three_groups => nanvar => :exposure_three_groups_variance)

println("Rent variance across the 30 replicates, by step (columns = district id):")
show(unstack(rent_variance_by_step, :step, :id, :rent_variance), allrows = true, allcols = true)
println()

println("Vacancy rate variance across the 30 replicates, by step (columns = district id):")
show(unstack(vacancy_rate_variance_by_step, :step, :id, :vacancy_rate_variance), allrows = true, allcols = true)
println()

println("Dissimilarity (three groups, citywide) variance across the 30 replicates, by step:")
show(dissimilarity_variance_by_step, allrows = true)
println()

println("Exposure (three groups, citywide) variance across the 30 replicates, by step:")
show(exposure_variance_by_step, allrows = true)
println()

mkpath(joinpath(@__DIR__, "statistical_equilibrium"))

plot(
    rent_variance_by_step.step,
    rent_variance_by_step.rent_variance,
    group = rent_variance_by_step.id,
    xlabel = "Simulation step",
    ylabel = "Variance across replicates",
    title = "Rent variance across replicates vs simulation step",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "rent.svg"))

plot(
    vacancy_rate_variance_by_step.step,
    vacancy_rate_variance_by_step.vacancy_rate_variance,
    group = vacancy_rate_variance_by_step.id,
    xlabel = "Simulation step",
    ylabel = "Variance across replicates",
    title = "Vacancy rate variance across replicates vs simulation step",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "vacancy_rate.svg"))

plot(
    dissimilarity_variance_by_step.step,
    dissimilarity_variance_by_step.dissimilarity_three_groups_variance,
    xlabel = "Simulation step",
    ylabel = "Variance across replicates",
    title = "Dissimilarity (three groups) variance across replicates vs simulation step",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "dissimilarity.svg"))

plot(
    exposure_variance_by_step.step,
    exposure_variance_by_step.exposure_three_groups_variance,
    xlabel = "Simulation step",
    ylabel = "Variance across replicates",
    title = "Exposure (three groups) variance across replicates vs simulation step",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "exposure.svg"))

# --- Within-run variance ---------------------------------------------------
# The across-replicate variance above shows whether the 30 replicates come
# to resemble EACH OTHER — it does not show whether any single run has
# itself stopped changing (replicates could drift together in lockstep and
# still look identical to each other at every step). This section instead
# tracks, on one representative replicate, the variance of its own trailing
# WITHIN_RUN_WINDOW-step window at each point in its trajectory — i.e.
# whether that one run's own state has settled into a fixed point.
const WITHIN_RUN_REPLICATE_SEED = 0
const WITHIN_RUN_WINDOW = 20

function rolling_var(xs::AbstractVector{<:Real}, window::Int)
    result = fill(NaN, length(xs))
    for i in window:length(xs)
        result[i] = var(view(xs, (i - window + 1):i))
    end
    return result
end

function rolling_nanvar(xs::AbstractVector{<:Real}, window::Int)
    result = fill(NaN, length(xs))
    for i in window:length(xs)
        result[i] = nanvar(view(xs, (i - window + 1):i))
    end
    return result
end

within_run_idx = findfirst(==(WITHIN_RUN_REPLICATE_SEED), replicate_seeds)
within_run_district_data = sort(district_data_by_run[within_run_idx], [:id, :step])
within_run_model_data = sort(model_data_by_run[within_run_idx], :step)

within_run_rent_variance = combine(groupby(within_run_district_data, :id)) do sub
    DataFrame(step = sub.step, rent_variance = rolling_var(sub.rent, WITHIN_RUN_WINDOW))
end

within_run_vacancy_rate_variance = combine(groupby(within_run_district_data, :id)) do sub
    DataFrame(step = sub.step, vacancy_rate_variance = rolling_var(sub.vacancy_rate, WITHIN_RUN_WINDOW))
end

within_run_dissimilarity_variance = DataFrame(
    step = within_run_model_data.step,
    dissimilarity_three_groups_variance = rolling_nanvar(within_run_model_data.dissimilarity_three_groups, WITHIN_RUN_WINDOW),
)

within_run_exposure_variance = DataFrame(
    step = within_run_model_data.step,
    exposure_three_groups_variance = rolling_nanvar(within_run_model_data.exposure_three_groups, WITHIN_RUN_WINDOW),
)

println("Within-run rent variance (replicate seed=$WITHIN_RUN_REPLICATE_SEED, trailing $WITHIN_RUN_WINDOW-step window), by step (columns = district id):")
show(unstack(within_run_rent_variance, :step, :id, :rent_variance), allrows = true, allcols = true)
println()

println("Within-run vacancy rate variance (replicate seed=$WITHIN_RUN_REPLICATE_SEED, trailing $WITHIN_RUN_WINDOW-step window), by step (columns = district id):")
show(unstack(within_run_vacancy_rate_variance, :step, :id, :vacancy_rate_variance), allrows = true, allcols = true)
println()

println("Within-run dissimilarity (three groups, citywide) variance (replicate seed=$WITHIN_RUN_REPLICATE_SEED, trailing $WITHIN_RUN_WINDOW-step window), by step:")
show(within_run_dissimilarity_variance, allrows = true)
println()

println("Within-run exposure (three groups, citywide) variance (replicate seed=$WITHIN_RUN_REPLICATE_SEED, trailing $WITHIN_RUN_WINDOW-step window), by step:")
show(within_run_exposure_variance, allrows = true)
println()

plot(
    within_run_rent_variance.step,
    within_run_rent_variance.rent_variance,
    group = within_run_rent_variance.id,
    xlabel = "Simulation step",
    ylabel = "Within-run variance (trailing $WITHIN_RUN_WINDOW steps)",
    title = "Rent within-run variance vs simulation step (replicate seed=$WITHIN_RUN_REPLICATE_SEED)",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "rent_within_run.svg"))

plot(
    within_run_vacancy_rate_variance.step,
    within_run_vacancy_rate_variance.vacancy_rate_variance,
    group = within_run_vacancy_rate_variance.id,
    xlabel = "Simulation step",
    ylabel = "Within-run variance (trailing $WITHIN_RUN_WINDOW steps)",
    title = "Vacancy rate within-run variance vs simulation step (replicate seed=$WITHIN_RUN_REPLICATE_SEED)",
    size = (900, 500),
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "vacancy_rate_within_run.svg"))

plot(
    within_run_dissimilarity_variance.step,
    within_run_dissimilarity_variance.dissimilarity_three_groups_variance,
    xlabel = "Simulation step",
    ylabel = "Within-run variance (trailing $WITHIN_RUN_WINDOW steps)",
    title = "Dissimilarity (three groups) within-run variance vs simulation step (replicate seed=$WITHIN_RUN_REPLICATE_SEED)",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "dissimilarity_within_run.svg"))

plot(
    within_run_exposure_variance.step,
    within_run_exposure_variance.exposure_three_groups_variance,
    xlabel = "Simulation step",
    ylabel = "Within-run variance (trailing $WITHIN_RUN_WINDOW steps)",
    title = "Exposure (three groups) within-run variance vs simulation step (replicate seed=$WITHIN_RUN_REPLICATE_SEED)",
    size = (900, 500),
    legend = false,
)
savefig(joinpath(@__DIR__, "statistical_equilibrium", "exposure_within_run.svg"))
