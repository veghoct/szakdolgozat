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
# simulated 30 times with the standard locked replicate seeds (260927..260956).
# Unlike every other SA scenario, the quantity of interest here is how the
# trajectories settle, not an equilibrium-window average. Two complementary
# views are computed: variance ACROSS the 30 replicates at each step (do the
# replicates resemble each other?), and WITHIN-run stationarity of every
# replicate's own trajectory (has each run's level stopped drifting?) via
# the MSER-10 warm-up rule and a standardized window-drift statistic -- see
# below.

const BENCHMARK_AMENITY = [
    1.0950729919433595,
    2.8319473571777345,
    1.1353562103271484,
    2.319447576904297,
    1,
    2.5699244720458987,
]
const BENCHMARK_UNITS = [142, 151, 363, 117, 170, 20]
const BENCHMARK_NUMBER_OF_HOUSEHOLDS = 799

# Fixed seed for the one-off random parameter draw below. Distinct from the
# 30 replicate seeds (260927..260956) used to run that parameter set, so the draw
# itself is reproducible independently of the replicate RNG streams.
# number_of_households and units stay at their benchmark values (799 <= 963),
# so every seed yields a feasible parameter set.
const PARAMETER_DRAW_SEED = 260926

draw_rng = MersenneTwister(PARAMETER_DRAW_SEED)

amenity = [rand(draw_rng, Uniform(1, 1.25 * a)) for a in BENCHMARK_AMENITY]
utility_alpha = rand(draw_rng, Uniform(0.05, 0.95))
utility_beta = rand(draw_rng, Uniform(0.05, 0.95))
number_of_households = BENCHMARK_NUMBER_OF_HOUSEHOLDS
units = BENCHMARK_UNITS

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
        150,
        150,
        150,
        150,
        150,
        150,
    ],
    units = units,
)

println("Statistical equilibrium scenario — random parameter set (draw seed = $PARAMETER_DRAW_SEED):")
println("utility_alpha = $utility_alpha")
println("utility_beta = $utility_beta")
println("price_change = $price_change")
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
replicate_seeds = 260927:260956

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

# Thesis figures (Hungarian, sized for \includesvg at \textwidth = 16 cm).
# GKSwstype = 100 renders GR headless (no display server needed).
ENV["GKSwstype"] = "100"

const SUBMARKET_NAMES = [
    "1. övezet – megfizethető",
    "1. övezet – prémium",
    "2. övezet – megfizethető",
    "2. övezet – prémium",
    "3. övezet – megfizethető",
    "3. övezet – prémium",
]
const ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
const CITYWIDE_COLOR = "#333333"
# serif-roman (Computer Modern) is embedded as outlines, so the SVG needs no
# installed font and covers the Hungarian and Greek characters used below.
const FIGURE_FONT = "serif-roman"
const STEP_LABEL = "szimulációs lépés"

submarket_color(id) = ZONE_COLORS[cld(id, 2)]
submarket_linestyle(id) = isodd(id) ? :solid : :dash
hu_number(x) = (r = round(x, digits = 4); isinteger(r) ? string(Int(r)) : replace(string(r), "." => ","))

# Hungarian label of a stationarity metric name (rent_d3, vacancy_rate_d6, ...).
function metric_label(metric)
    metric == "dissimilarity_three_groups" && return "Disszimilaritási index"
    metric == "exposure_three_groups" && return "Kitettségi index"
    id = parse(Int, last(split(metric, "_d")))
    quantity = startswith(metric, "rent_d") ? "Albérleti díj" : "Üresedési ráta"
    return "$quantity – $(SUBMARKET_NAMES[id])"
end

const PANEL_STYLE = (
    fontfamily = FIGURE_FONT, xlabel = STEP_LABEL, xformatter = hu_number, yformatter = hu_number,
    titlefontsize = 8, guidefontsize = 8, tickfontsize = 7, legend = false,
    gridalpha = 0.15, framestyle = :axes,
)

# Shared legend below a 1x3 figure: one row per zone (affordable | premium),
# then any extra (label, color, linestyle) entries.
function submarket_legend(extra_entries)
    legend_panel = plot(; framestyle = :none, legend = :top, legend_column = 2, legendfontsize = 7, fontfamily = FIGURE_FONT,
        foreground_color_legend = nothing, background_color_legend = nothing)
    for id in 1:6
        plot!(legend_panel, [NaN], [NaN]; label = SUBMARKET_NAMES[id], color = submarket_color(id), linestyle = submarket_linestyle(id), linewidth = 1.2)
    end
    for (label, color, linestyle) in extra_entries
        plot!(legend_panel, [NaN], [NaN]; label = label, color = color, linestyle = linestyle, linewidth = 1.2)
    end
    return legend_panel
end

function save_three_panel_figure(panels, legend_panel, title, filename)
    figure = plot(panels..., legend_panel;
        layout = @layout([a b c; d{0.24h}]), size = (605, 320),
        plot_title = title, plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT,
        left_margin = 1Plots.mm, right_margin = 1Plots.mm, top_margin = 0Plots.mm, bottom_margin = 1Plots.mm)
    savefig(figure, joinpath(@__DIR__, "statistical_equilibrium", filename))
end

# Across-replicate variance: rent per submarket, dissimilarity, exposure.
# Vacancy is printed above but not plotted, matching the sweep figures.
rent_variance_panel = plot(; title = "Albérleti díj", PANEL_STYLE...)
for id in 1:6
    sub = sort(filter(:id => ==(id), rent_variance_by_step), :step)
    plot!(rent_variance_panel, sub.step, sub.rent_variance; color = submarket_color(id), linestyle = submarket_linestyle(id), linewidth = 0.8)
end
dissimilarity_variance_panel = plot(dissimilarity_variance_by_step.step, dissimilarity_variance_by_step.dissimilarity_three_groups_variance;
    title = "Disszimilaritási index", color = CITYWIDE_COLOR, linewidth = 0.8, PANEL_STYLE...)
exposure_variance_panel = plot(exposure_variance_by_step.step, exposure_variance_by_step.exposure_three_groups_variance;
    title = "Kitettségi index", color = CITYWIDE_COLOR, linewidth = 0.8, PANEL_STYLE...)
save_three_panel_figure(
    (rent_variance_panel, dissimilarity_variance_panel, exposure_variance_panel),
    submarket_legend([("városi index", CITYWIDE_COLOR, :solid)]),
    "Szórásnégyzet a $(length(replicate_seeds)) ismétlés között, szimulációs lépésenként",
    "variance.svg",
)

# --- Within-run stationarity -----------------------------------------------
# The across-replicate variance above shows whether the 30 replicates come
# to resemble EACH OTHER — it does not show whether any single run has
# itself stopped changing (replicates could drift together in lockstep and
# still look identical to each other at every step).
#
# step! draws no random numbers (randomness enters only through the initial
# home shuffle and income draws), yet no run settles into a fixed point or an
# exact cycle: every trajectory keeps fluctuating around its level through
# the endogenous rent/occupancy feedback. A within-run variance therefore
# plateaus at a nonzero fluctuation level early on, and a rolling variance
# over a short window is blind to a slow drift of the level itself (a drift
# of slope s adds only s^2 * w * (w + 1) / 12 to a w-step window's variance).
# Steady state here means the LEVEL has stopped drifting, so this section
# measures that, on every replicate rather than a single one:
#
# 1. MSER-10 warm-up length (MSER: White 1997; batched variants evaluated in
#    White, Cobb & Spratt 2000; automated and tested by Hoad, Robinson &
#    Davies 2010, who found it the best-performing warm-up rule). Batches of
#    10 steps are used instead of the usual 5: small districts' vacancy
#    oscillates with a period of ~5-7 steps, which 5-step batches alias into
#    a slow beat that MSER mistakes for a trend. 20-step batches removed that
#    but left only 50 batches in a 1000-step run, too few for a stable MSER
#    objective near the end of the search range.
#    The series is averaged into batches of 10 steps, and the truncation point
#    d minimizing the squared standard error of the mean of the remaining
#    batches, sum((y[d+1:k] - mean(y[d+1:k])).^2) / (k - d)^2, is the
#    warm-up length. Following Hoad et al., the last 5 batches (50 steps) are excluded
#    from the search (MSER spuriously favours truncation points right at the
#    end of a series) and a truncation point beyond half the run is rejected:
#    it means the run is too short to establish a steady state for that
#    metric, and more steps are needed.
# 2. Standardized window drift: at each step t, the mean of the last
#    DRIFT_WINDOW steps minus the mean of the DRIFT_WINDOW steps before
#    that, divided by the standard deviation over both windows together
#    (the idea of comparing an earlier and a later segment's means is that of
#    Geweke's 1992 convergence diagnostic). It is ~0 once the level stops
#    moving, however large the fluctuations around it. Averaged across the 30
#    independent replicates, with a 95% band of +-1.96 standard errors, it
#    shows whether the runs are still drifting systematically at step t.
#
# References:
# - White, K. P. Jr. (1997). An effective truncation heuristic for bias
#   reduction in simulation output. Simulation, 69(6), 323-334.
#   https://doi.org/10.1177/003754979706900601
# - White, K. P. Jr., Cobb, M. J. & Spratt, S. C. (2000). A comparison of
#   five steady-state truncation heuristics for simulation. Proceedings of
#   the 2000 Winter Simulation Conference, 755-760.
#   https://doi.org/10.1109/WSC.2000.899843
# - Hoad, K., Robinson, S. & Davies, R. (2010). Automating warm-up length
#   estimation. Journal of the Operational Research Society, 61(9),
#   1389-1403. https://doi.org/10.1057/jors.2009.87
# - Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
#   to the calculation of posterior moments. In J. M. Bernardo, J. O. Berger,
#   A. P. Dawid & A. F. M. Smith (Eds.), Bayesian Statistics 4 (pp. 169-193).
#   Oxford University Press.
const MSER_BATCH_SIZE = 10
const MSER_EXCLUDED_END_BATCHES = 5
const DRIFT_WINDOW = 50

# Warm-up length in simulation steps, or `missing` if the series contains a
# NaN (an undefined segregation index), which would break the time index.
function mser_truncation(xs::AbstractVector{<:Real})
    any(isnan, xs) && return missing

    k = length(xs) ÷ MSER_BATCH_SIZE
    batch_means = [mean(view(xs, ((j - 1) * MSER_BATCH_SIZE + 1):(j * MSER_BATCH_SIZE))) for j in 1:k]

    best_d, best_mser = 0, Inf
    for d in 0:(k - MSER_EXCLUDED_END_BATCHES)
        remaining = view(batch_means, (d + 1):k)
        mser = sum(abs2, remaining .- mean(remaining)) / length(remaining)^2
        if mser < best_mser
            best_d, best_mser = d, mser
        end
    end

    return best_d * MSER_BATCH_SIZE
end

function standardized_drift(xs::AbstractVector{<:Real}, window::Int)
    result = fill(NaN, length(xs))
    for t in (2 * window):length(xs)
        earlier = view(xs, (t - 2 * window + 1):(t - window))
        later = view(xs, (t - window + 1):t)
        spread = std(view(xs, (t - 2 * window + 1):t))
        result[t] = spread == 0 ? 0.0 : (mean(later) - mean(earlier)) / spread
    end
    return result
end

nanmean(xs) = (valid = filter(!isnan, xs); isempty(valid) ? NaN : mean(valid))
nan_ci95_halfwidth(xs) = (valid = filter(!isnan, xs); length(valid) < 2 ? NaN : 1.96 * std(valid) / sqrt(length(valid)))

# Every examined quantity of one replicate as a (metric name => series) list,
# in a fixed order: rent per district, vacancy rate per district,
# dissimilarity, exposure.
function stationarity_series(run_district_data::DataFrame, run_model_data::DataFrame)
    by_district = groupby(sort(run_district_data, [:id, :step]), :id, sort = true)
    run_model_data = sort(run_model_data, :step)

    return vcat(
        ["rent_d$(first(sub.id))" => sub.rent for sub in by_district],
        ["vacancy_rate_d$(first(sub.id))" => sub.vacancy_rate for sub in by_district],
        ["dissimilarity_three_groups" => run_model_data.dissimilarity_three_groups],
        ["exposure_three_groups" => run_model_data.exposure_three_groups],
    )
end

mser_rows = DataFrame(run_id = Int[], metric = String[], truncation_step = Union{Int, Missing}[])
drift_rows = DataFrame(run_id = Int[], metric = String[], step = Int[], drift = Float64[])

for i in eachindex(replicate_seeds)
    run_steps = sort(model_data_by_run[i], :step).step
    for (metric, xs) in stationarity_series(district_data_by_run[i], model_data_by_run[i])
        push!(mser_rows, (replicate_seeds[i], metric, mser_truncation(xs)))
        append!(drift_rows, DataFrame(run_id = replicate_seeds[i], metric = metric, step = run_steps, drift = standardized_drift(xs, DRIFT_WINDOW)))
    end
end

metric_order = unique(mser_rows.metric)
mser_rows.rejected = coalesce.(mser_rows.truncation_step .> steps ÷ 2, false)

println("MSER-$MSER_BATCH_SIZE warm-up length (steps) per replicate (rows = replicate seed, columns = metric); values > $(steps ÷ 2) (= half the run) are rejected as 'run too short for this metric':")
show(unstack(mser_rows, :run_id, :metric, :truncation_step), allrows = true, allcols = true)
println()

mser_summary = combine(groupby(mser_rows, :metric, sort = false),
    :truncation_step => (xs -> median(skipmissing(xs))) => :median_steps,
    [:truncation_step, :rejected] => ((xs, r) -> maximum(skipmissing(xs[.!r]); init = 0)) => :max_accepted_steps,
    :rejected => sum => :rejected_runs,
    :truncation_step => (xs -> count(ismissing, xs)) => :undefined_runs,
)
println("MSER-$MSER_BATCH_SIZE warm-up length summary across the $(length(replicate_seeds)) replicates (rejected_runs = replicates whose warm-up exceeded half the $steps-step run; undefined_runs = series containing NaN):")
show(mser_summary, allrows = true, allcols = true)
println()

# MSER warm-up length per replicate, one row per metric (first metric on top).
const ACCEPTED_COLOR = "#2a78d6"
const REJECTED_COLOR = "#eb6834"

metric_row = Dict(metric => length(metric_order) - i + 1 for (i, metric) in enumerate(metric_order))
defined_mser = filter(:truncation_step => !ismissing, mser_rows)
# Vertical jitter of the dots only (cosmetic), seeded for a reproducible figure.
row_jitter = 0.25 .* (rand(MersenneTwister(PARAMETER_DRAW_SEED), nrow(defined_mser)) .- 0.5)
warmup_plot = plot(;
    fontfamily = FIGURE_FONT, xlabel = "MSER-$MSER_BATCH_SIZE bemelegedési szakasz hossza (lépés)",
    xformatter = hu_number, yticks = ([metric_row[m] for m in metric_order], metric_label.(metric_order)),
    ylims = (0.4, length(metric_order) + 0.6), xlims = (-0.02 * steps, 1.02 * steps),
    guidefontsize = 8, tickfontsize = 7, legendfontsize = 7, gridalpha = 0.15, framestyle = :axes,
    legend = :outerbottom, legend_column = 2, foreground_color_legend = nothing,
    size = (605, 400), left_margin = 1Plots.mm, right_margin = 2Plots.mm, bottom_margin = 1Plots.mm,
    title = "MSER-$MSER_BATCH_SIZE bemelegedési szakasz ismétlésenként és mutatónként",
    titlefontsize = 9,
)
vline!(warmup_plot, [steps ÷ 2]; color = :black, linestyle = :dash, linewidth = 1, label = "elutasítási küszöb (a futás fele)")
for (rejected, color, label) in ((false, ACCEPTED_COLOR, "elfogadott"), (true, REJECTED_COLOR, "elutasított"))
    selected = defined_mser.rejected .== rejected
    scatter!(warmup_plot, defined_mser.truncation_step[selected],
        [metric_row[m] for m in defined_mser.metric[selected]] .+ row_jitter[selected];
        color = color, markersize = 3, markerstrokewidth = 0, markeralpha = 0.7, label = "$label ($(count(selected)))")
end
savefig(warmup_plot, joinpath(@__DIR__, "statistical_equilibrium", "mser_warmup.svg"))

# Is a rejected (late) MSER truncation point a real late warm-up, or noise?
# If a series were still settling, its level would differ between an earlier
# and a later part of the run. Test: mean of steps (0.6n, n] minus mean of
# steps (0.2n, 0.6n], as a t statistic. Its standard error comes from the
# means of 8 blocks of n/10 steps each (blocks long enough to be roughly
# independent despite the step-to-step autocorrelation), so |t| is compared
# with the 97.5% quantile of a t distribution with 8 - 2 = 6 degrees of
# freedom. Steps before 0.2n are left out so the initial transient doesn't
# dominate; any warm-up longer than that still sits in the earlier window
# and would push |t| up, so the test errs towards finding drift.
# The last truncation point MSER may return is the end of its search range:
# (number of batches - MSER_EXCLUDED_END_BATCHES) * MSER_BATCH_SIZE.
const LEVEL_SHIFT_BLOCKS = 10
level_shift_critical = quantile(TDist(LEVEL_SHIFT_BLOCKS - 2 - 2), 0.975)
mser_search_limit = (steps ÷ MSER_BATCH_SIZE - MSER_EXCLUDED_END_BATCHES) * MSER_BATCH_SIZE

function level_shift_t(xs::AbstractVector{<:Real})
    any(isnan, xs) && return NaN
    block = length(xs) ÷ LEVEL_SHIFT_BLOCKS
    block_means = [mean(view(xs, ((j - 1) * block + 1):(j * block))) for j in 3:LEVEL_SHIFT_BLOCKS]
    half = length(block_means) ÷ 2
    early, late = block_means[1:half], block_means[(half + 1):end]
    pooled_sd = sqrt((var(early) + var(late)) / 2)
    pooled_sd == 0 && return 0.0
    return (mean(late) - mean(early)) / (pooled_sd * sqrt(2 / half))
end

series_by_key = Dict{Tuple{Int, String}, Vector{Float64}}()
for i in eachindex(replicate_seeds)
    for (metric, xs) in stationarity_series(district_data_by_run[i], model_data_by_run[i])
        series_by_key[(replicate_seeds[i], metric)] = xs
    end
end
mser_rows.level_shift_t = [series_by_key[(r.run_id, r.metric)] |> level_shift_t for r in eachrow(mser_rows)]

defined_rows = filter(r -> !ismissing(r.truncation_step) && !isnan(r.level_shift_t), mser_rows)
accepted_rows = filter(:rejected => !, defined_rows)
rejected_rows = filter(:rejected => identity, defined_rows)
significant_share(rows) = nrow(rows) == 0 ? NaN : mean(abs.(rows.level_shift_t) .> level_shift_critical)

println("Level-shift check of the MSER warm-up lengths (t = late-minus-early mean difference; |t| > $(round(level_shift_critical, digits = 2)) means the level changed significantly):")
show(DataFrame(
    group = ["accepted", "rejected"],
    series = [nrow(accepted_rows), nrow(rejected_rows)],
    at_search_limit = [count(==(mser_search_limit), accepted_rows.truncation_step), count(==(mser_search_limit), rejected_rows.truncation_step)],
    median_abs_t = [isempty(accepted_rows.level_shift_t) ? NaN : median(abs.(accepted_rows.level_shift_t)), isempty(rejected_rows.level_shift_t) ? NaN : median(abs.(rejected_rows.level_shift_t))],
    share_significant = [significant_share(accepted_rows), significant_share(rejected_rows)],
), allcols = true)
println()
println("Rejected series (MSER truncation step, level-shift t):")
show(sort(rejected_rows[:, [:run_id, :metric, :truncation_step, :level_shift_t]], :truncation_step), allrows = true, allcols = true)
println()

boundary_plot = plot(;
    fontfamily = FIGURE_FONT,
    xlabel = "MSER-$MSER_BATCH_SIZE bemelegedési szakasz hossza (lépés)",
    ylabel = "szintváltozás (t-érték)",
    title = "Késői vs. korábbi futásszakasz átlagának eltérése (ismétlésenként és mutatónként)",
    titlefontsize = 9, guidefontsize = 8, tickfontsize = 7, legendfontsize = 7,
    xformatter = hu_number, yformatter = hu_number, gridalpha = 0.15, framestyle = :axes,
    legend = :outerbottom, legend_column = 1, foreground_color_legend = nothing,
    size = (605, 420), left_margin = 1Plots.mm, right_margin = 2Plots.mm, bottom_margin = 1Plots.mm,
    xlims = (-0.02 * steps, 1.02 * steps),
)
hspan!(boundary_plot, [-level_shift_critical, level_shift_critical]; color = :gray, alpha = 0.12, linealpha = 0,
    label = "nincs szignifikáns szintváltozás (95%, |t| < $(hu_number(round(level_shift_critical, digits = 2))))")
vline!(boundary_plot, [steps ÷ 2]; color = :black, linestyle = :dash, linewidth = 1, label = "elutasítási küszöb (a futás fele)")
vline!(boundary_plot, [mser_search_limit]; color = :black, linestyle = :dot, linewidth = 1, label = "az MSER-keresési tartomány vége")
scatter!(boundary_plot, accepted_rows.truncation_step, accepted_rows.level_shift_t;
    color = ACCEPTED_COLOR, markersize = 3, markerstrokewidth = 0, markeralpha = 0.6, label = "elfogadott ($(nrow(accepted_rows)))")
scatter!(boundary_plot, rejected_rows.truncation_step, rejected_rows.level_shift_t;
    color = REJECTED_COLOR, markersize = 4.5, markerstrokecolor = :white, markerstrokewidth = 0.8, label = "elutasított ($(nrow(rejected_rows)))")
savefig(boundary_plot, joinpath(@__DIR__, "statistical_equilibrium", "mser_boundary_check.svg"))

# The rejected series themselves: raw series, MSER batch means, the MSER
# truncation point, and the early/late means compared above.
if nrow(rejected_rows) > 0
    # At most 8 panels (4 rows of 2) so the figure fits on one page.
    shown = first(sort(rejected_rows, :truncation_step, rev = true), 8)
    panels = map(eachrow(shown)) do r
        xs = series_by_key[(r.run_id, r.metric)]
        n = length(xs)
        k = n ÷ MSER_BATCH_SIZE
        batch_mid = [(j - 0.5) * MSER_BATCH_SIZE for j in 1:k]
        batch_means = [mean(view(xs, ((j - 1) * MSER_BATCH_SIZE + 1):(j * MSER_BATCH_SIZE))) for j in 1:k]
        early_range = (n ÷ 5 + 1):(3 * n ÷ 5)
        late_range = (3 * n ÷ 5 + 1):n
        p = plot(1:n, xs; color = :gray, alpha = 0.35, linewidth = 0.6,
            title = "$(metric_label(r.metric))\n$(findfirst(==(r.run_id), replicate_seeds)). ismétlés, MSER-vágás: $(r.truncation_step), t = $(hu_number(round(r.level_shift_t, digits = 2)))",
            PANEL_STYLE..., titlefontsize = 7)
        plot!(p, batch_mid, batch_means; color = ACCEPTED_COLOR, linewidth = 1.2)
        plot!(p, [first(early_range), last(early_range)], fill(mean(xs[early_range]), 2); color = :black, linewidth = 1.5)
        plot!(p, [first(late_range), last(late_range)], fill(mean(xs[late_range]), 2); color = :black, linewidth = 1.5)
        vline!(p, [r.truncation_step]; color = REJECTED_COLOR, linewidth = 1.5)
        p
    end
    columns = min(2, length(panels))
    rows = cld(length(panels), columns)
    # "(8/74)" when only the latest-cut 8 of more rejected series are shown.
    shown_note = nrow(shown) < nrow(rejected_rows) ? " ($(nrow(shown))/$(nrow(rejected_rows)))" : ""
    plot(panels...; layout = (rows, columns), size = (605, 50 + 190 * rows),
        plot_title = "Elutasított idősorok$shown_note: nyers (szürke), blokkátlag (kék), MSER-vágás (narancs), korábbi/késői átlag (fekete)",
        plot_titlefontsize = 8, plot_titlefontfamily = FIGURE_FONT,
        left_margin = 1Plots.mm, right_margin = 2Plots.mm, bottom_margin = 1Plots.mm)
    savefig(joinpath(@__DIR__, "statistical_equilibrium", "mser_rejected_series.svg"))
end

drift_summary = combine(groupby(drift_rows, [:metric, :step], sort = false),
    :drift => nanmean => :mean_drift,
    :drift => nan_ci95_halfwidth => :ci95_halfwidth,
)

rent_metrics = filter(startswith("rent_d"), metric_order)
vacancy_rate_metrics = filter(startswith("vacancy_rate_d"), metric_order)

function print_drift_table(metrics, description)
    sub = filter(:metric => in(metrics), drift_summary)
    println("Standardized $DRIFT_WINDOW-step window drift of $description, mean across the $(length(replicate_seeds)) replicates, by step (columns = metric):")
    show(unstack(sub, :step, :metric, :mean_drift), allrows = true, allcols = true)
    println()
    println("95% confidence half-width of the above mean drift, by step (columns = metric):")
    show(unstack(sub, :step, :metric, :ci95_halfwidth), allrows = true, allcols = true)
    println()
end

print_drift_table(rent_metrics, "rent")
print_drift_table(vacancy_rate_metrics, "vacancy rate")
print_drift_table(["dissimilarity_three_groups", "exposure_three_groups"], "dissimilarity and exposure (three groups, citywide)")

# Standardized window drift: rent per submarket, dissimilarity, exposure,
# each with its 95% band across the replicates.
function drift_series(metric)
    sub = sort(filter(:metric => ==(metric), drift_summary), :step)
    return sub.step, sub.mean_drift, sub.ci95_halfwidth
end

rent_drift_panel = plot(; title = "Albérleti díj", PANEL_STYLE...)
for (id, metric) in enumerate(rent_metrics)
    step, drift, halfwidth = drift_series(metric)
    plot!(rent_drift_panel, step, drift; ribbon = halfwidth, fillalpha = 0.12, color = submarket_color(id), linestyle = submarket_linestyle(id), linewidth = 0.8)
end
citywide_drift_panels = map((("dissimilarity_three_groups", "Disszimilaritási index"), ("exposure_three_groups", "Kitettségi index"))) do (metric, title)
    step, drift, halfwidth = drift_series(metric)
    plot(step, drift; ribbon = halfwidth, fillalpha = 0.2, color = CITYWIDE_COLOR, linewidth = 0.8, title = title, PANEL_STYLE...)
end
drift_panels = (rent_drift_panel, citywide_drift_panels...)
for panel in drift_panels
    hline!(panel, [0]; color = :black, linestyle = :dot, linewidth = 0.8)
end
save_three_panel_figure(
    drift_panels,
    submarket_legend([("városi index", CITYWIDE_COLOR, :solid), ("nulla (nincs szintváltozás)", :black, :dot)]),
    "Standardizált szintváltozás (utolsó $DRIFT_WINDOW vs. előző $DRIFT_WINDOW lépés; $(length(replicate_seeds)) ismétlés átlaga, 95%-os sáv)",
    "drift.svg",
)
