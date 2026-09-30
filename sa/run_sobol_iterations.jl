include("../util/imports.jl")

# Thesis figures of the Sobol-sequence SMM calibration of the submarket
# amenities (model/sobol_smm_calibration_iter1.jl ... iter4.jl). Each
# iteration evaluated a Sobol sample of the five free amenities inside a search
# box, and the next iteration searched a narrower box (iteration 4's box is
# iteration 3's best point +/- 10%). This script only reads the saved result
# CSVs and runs no simulation. It prints the numbers behind every figure and
# saves five figures into sa/sobol_iterations/:
#   search_box.svg - search box, range of the best 1% and best point per
#                    iteration, one panel per free amenity
#   objective.svg  - distribution of the objective per iteration
#   rent_fit.svg   - each iteration's best point: simulated vs empirical rent
#   profile.svg    - lowest objective along each amenity (iterations 1 and 4)
#   pairs.svg      - the amenities of iteration 4's best 1%, pair by pair
#
# The objective throughout is objective_per_diff, the sum over the six
# submarkets of ((simulated - empirical) / empirical)^2, the one the
# calibration minimized (each result file is sorted by it).

# Settings of each calibration run, copied from its script. sobol_points is
# the size of the Sobol sample the script generated (LAST_SAMPLE). The search
# box covers amenities 1, 2, 3, 4 and 6; amenity 5 is fixed at 1 in every run.
const ITERATIONS = [
    (file = "sobol_smm_results_iter1.csv", sobol_points = 100_000, replications = 10, start_rent = 120,
        lower = [1.0, 1.0, 1.0, 1.0, 1.0], upper = [10.0, 10.0, 10.0, 10.0, 10.0]),
    (file = "sobol_smm_results_iter2.csv", sobol_points = 100_000, replications = 10, start_rent = 120,
        lower = [1.0, 2.8, 1.0, 1.9, 1.0], upper = [4.6, 10.0, 4.6, 8.2, 9.1]),
    (file = "sobol_smm_results_iter3.csv", sobol_points = 100_000, replications = 10, start_rent = 120,
        lower = [1.0, 2.8, 1.0, 1.9, 1.81], upper = [2.8, 7.12, 2.8, 5.68, 5.86]),
    (file = "sobol_smm_results_iter4.csv", sobol_points = 25_000, replications = 25, start_rent = 130,
        lower = [1.0396, 2.6274, 1.0697, 2.2887, 2.5413], upper = [1.2706, 3.2112, 1.3074, 2.7973, 3.1060]),
]
const ITERATION_4_RENT_120_FILE = "sobol_smm_results_iter4 copy.csv"
const FREE_DISTRICTS = [1, 2, 3, 4, 6]
const AMENITY_COLUMNS = [Symbol("amenity_$d") for d in FREE_DISTRICTS]
const EMPIRICAL_RENTS = [138.0, 300.0, 130.0, 240.0, 120.0, 235.0]
const TOP_SHARE = 0.01

# Iteration 4's result file holds two interleaved passes over the same 25 000
# Sobol points: one run with starting rent 120 (identical to the copy file)
# and one with starting rent 130, the iteration-4 script's setting, which the
# current model code reproduces. The 130 pass is every row whose objective
# differs from the copy's for the same sample_id. One line fuses two rows
# (concurrent writes) and is skipped, which loses the 130-pass row of one
# point, so 24 999 of the 25 000 points remain.
function load_rent130_pass(combined_file, rent120_file)
    rent120 = CSV.read(joinpath(@__DIR__, "..", rent120_file), DataFrame)
    rent120_objective = Dict(zip(rent120.sample_id, rent120.objective_per_diff))
    lines = readlines(joinpath(@__DIR__, "..", combined_file))
    intact = filter(line -> count(==(','), line) == count(==(','), lines[1]), lines)
    combined = CSV.read(IOBuffer(join(intact, "\n")), DataFrame)
    return filter(row -> row.objective_per_diff != rent120_objective[row.sample_id], combined)
end

results = [CSV.read(joinpath(@__DIR__, "..", iteration.file), DataFrame) for iteration in ITERATIONS[1:3]]
push!(results, load_rent130_pass(ITERATIONS[4].file, ITERATION_4_RENT_120_FILE))
best_rows = [r[argmin(r.objective_per_diff), :] for r in results]
top_sets = [first(sort(r, :objective_per_diff), ceil(Int, TOP_SHARE * nrow(r))) for r in results]
model_rents(row) = [row[Symbol("model_rent_$d")] for d in 1:6]
rent_deviation_pct(row) = 100 .* (model_rents(row) .- EMPIRICAL_RENTS) ./ EMPIRICAL_RENTS

objective_quantiles = [quantile(r.objective_per_diff, [0.01, 0.25, 0.5, 0.75, 0.99]) for r in results]

println("Sobol calibration iterations (objective = objective_per_diff; p01..p99 = its percentiles):")
show(DataFrame(
    iteration = 1:4,
    sobol_points = [it.sobol_points for it in ITERATIONS],
    rows_in_file = nrow.(results),
    replications = [it.replications for it in ITERATIONS],
    start_rent = [it.start_rent for it in ITERATIONS],
    best_sample_id = [b.sample_id for b in best_rows],
    best_objective = [b.objective_per_diff for b in best_rows],
    p01 = [q[1] for q in objective_quantiles],
    p25 = [q[2] for q in objective_quantiles],
    median = [q[3] for q in objective_quantiles],
    p75 = [q[4] for q in objective_quantiles],
    p99 = [q[5] for q in objective_quantiles],
), allcols = true)
println()

box_rows = DataFrame(iteration = Int[], amenity = Int[], box_lower = Float64[], box_upper = Float64[],
    top_min = Float64[], top_max = Float64[], best = Float64[])
for k in 1:4, (a, column) in enumerate(AMENITY_COLUMNS)
    push!(box_rows, (k, FREE_DISTRICTS[a], ITERATIONS[k].lower[a], ITERATIONS[k].upper[a],
        minimum(top_sets[k][!, column]), maximum(top_sets[k][!, column]), best_rows[k][column]))
end
println("Search box, range of the best $(round(Int, 100 * TOP_SHARE))% (top_min..top_max) and best point per iteration and amenity:")
show(box_rows, allrows = true, allcols = true)
println()

rent_rows = DataFrame(iteration = Int[], id = Int[], empirical_rent = Float64[], model_rent = Float64[], deviation_pct = Float64[])
for k in 1:4, d in 1:6
    push!(rent_rows, (k, d, EMPIRICAL_RENTS[d], model_rents(best_rows[k])[d], rent_deviation_pct(best_rows[k])[d]))
end
println("Simulated rent of each iteration's best point vs the empirical median rent (ezer Ft):")
show(rent_rows, allrows = true, allcols = true)
println()

final_top = top_sets[4]
println("Correlation of the amenities within iteration 4's best $(nrow(final_top)) points (rows/columns = amenity 1, 2, 3, 4, 6; the size depends on the $(round(Int, 100 * TOP_SHARE))% cut, so pairs.svg does not show it):")
show(DataFrame(cor(Matrix(final_top[:, AMENITY_COLUMNS])), string.("amenity_", FREE_DISTRICTS)), allcols = true)
println()

# Thesis figures (Hungarian, sized for \includesvg at \textwidth = 16 cm), in
# the format of the sensitivity-analysis figures: colour encodes the zone,
# line style the submarket (solid = megfizethető, dashed = prémium).
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
const SUBMARKET_SHORT_NAMES = ["1. öv. – megf.", "1. öv. – prém.", "2. öv. – megf.", "2. öv. – prém.", "3. öv. – megf.", "3. öv. – prém."]
const ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
const CITYWIDE_COLOR = "#333333"
const BENCHMARK_COLOR = "#8a8a8a"
const BOX_FILL_COLOR = "#e4e4e4"
const HIGHLIGHT_COLOR = "#2a78d6"
# Iterations 1 -> 4 as a one-hue ordinal ramp (light -> dark), with marker
# shape as a second channel since the lightest step is below 3:1 on white.
const ITERATION_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
const ITERATION_MARKERS = [:circle, :rect, :diamond, :utriangle]
# serif-roman (Computer Modern) is embedded as outlines, so the SVG needs no
# installed font and covers the Hungarian characters used below.
const FIGURE_FONT = "serif-roman"

submarket_color(id) = ZONE_COLORS[cld(id, 2)]
submarket_linestyle(id) = isodd(id) ? :solid : :dash
hu_number(x) = (r = round(x, digits = 4); isinteger(r) ? string(Int(r)) : replace(string(r), "." => ","))
hu_significant(x) = replace(string(round(x, sigdigits = 3)), "." => ",")
hu_thousands(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => " ")
hu_text(s, args...) = text(s, FIGURE_FONT, args...)

# Ticks strictly inside [lo, hi]: the axes below span exactly a search box,
# and a tick outside it would be drawn past the end of the axis.
function hu_ticks(lo, hi)
    ticks = filter(t -> lo <= t <= hi, Plots.PlotUtils.optimize_ticks(lo, hi; k_min = 2, k_max = 4)[1])
    return (ticks, hu_number.(ticks))
end

const PANEL_STYLE = (
    fontfamily = FIGURE_FONT, titlefontsize = 8, guidefontsize = 8, tickfontsize = 7, legend = false,
    gridalpha = 0.15, framestyle = :axes,
)
const LEGEND_STYLE = (
    framestyle = :none, legendfontsize = 7, fontfamily = FIGURE_FONT,
    foreground_color_legend = nothing, background_color_legend = nothing,
)
const FIGURE_MARGINS = (left_margin = 1Plots.mm, right_margin = 2Plots.mm, top_margin = 0Plots.mm, bottom_margin = 1Plots.mm)

mkpath(joinpath(@__DIR__, "sobol_iterations"))
figure_path(filename) = joinpath(@__DIR__, "sobol_iterations", filename)
top_percent_label = "$(round(Int, 100 * TOP_SHARE))%"

# A: how the search box narrowed. Per free amenity and iteration: the searched
# range (light bar), the range of the best 1% of points (dark bar) and the
# best point (dot, joined across iterations). Log scale, since the boxes
# shrink multiplicatively (iteration 4: +/- 10% around a point).
const AMENITY_TICKS = [1, 1.5, 2, 3, 5, 10]

search_box_panels = map(enumerate(FREE_DISTRICTS)) do (a, district)
    color = submarket_color(district)
    panel = plot(; title = SUBMARKET_NAMES[district], xlabel = "iteráció",
        xticks = (1:4, ["1.", "2.", "3.", "4."]), xlims = (0.4, 4.6),
        yscale = :log10, yticks = (AMENITY_TICKS, hu_number.(AMENITY_TICKS)), ylims = (0.92, 11),
        ylabel = a in (1, 4) ? "amenitásszint" : "", PANEL_STYLE...)
    for k in 1:4
        plot!(panel, [k, k], [ITERATIONS[k].lower[a], ITERATIONS[k].upper[a]]; color = color, alpha = 0.25, linewidth = 9)
        plot!(panel, [k, k], collect(extrema(top_sets[k][!, AMENITY_COLUMNS[a]])); color = color, linewidth = 3)
    end
    best = [b[AMENITY_COLUMNS[a]] for b in best_rows]
    plot!(panel, 1:4, best; color = :black, linewidth = 0.6, markershape = :circle, markersize = 2.5, markercolor = :black, markerstrokewidth = 0)
    panel
end
search_box_legend = plot(; LEGEND_STYLE..., legend = :top, xlims = (0, 1), ylims = (0, 1))
plot!(search_box_legend, [NaN], [NaN]; label = "keresési tartomány", color = BENCHMARK_COLOR, alpha = 0.35, linewidth = 9)
plot!(search_box_legend, [NaN], [NaN]; label = "a legjobb $top_percent_label tartománya", color = BENCHMARK_COLOR, linewidth = 3)
plot!(search_box_legend, [NaN], [NaN]; label = "legjobb pont", color = :black, linewidth = 0.6,
    markershape = :circle, markersize = 2.5, markercolor = :black, markerstrokewidth = 0)
annotate!(search_box_legend, 0.5, 0.2, hu_text("A 3. övezet – megfizethető\namenitása mindvégig 1 (rögzített).", 7, :center))
search_box_figure = plot(search_box_panels..., search_box_legend;
    layout = @layout([a b c; d e f]), size = (605, 420),
    plot_title = "A keresési tartomány szűkülése iterációnként (logaritmikus skála)",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(search_box_figure, figure_path("search_box.svg"))

# B: distribution of the objective per iteration (log scale): 1st-99th
# percentile whisker, interquartile box, median, and the best value.
iteration_labels = ["$k. iteráció\nN = $(hu_thousands(it.sobol_points))\n$(it.replications) ismétlés" for (k, it) in enumerate(ITERATIONS)]
best_objectives = [b.objective_per_diff for b in best_rows]
box_half_width = 0.16

objective_figure = plot(; xticks = (1:4, iteration_labels), xlims = (0.4, 4.6),
    yscale = :log10, yticks = ([0.001, 0.01, 0.1, 1, 10], hu_number.([0.001, 0.01, 0.1, 1, 10])), ylims = (6e-4, 12),
    ylabel = "célfüggvény-érték (logaritmikus skála)",
    title = "A célfüggvény (relatív eltérések négyzetösszege) eloszlása iterációnként",
    PANEL_STYLE..., titlefontsize = 9, legend = :outerbottom, legend_column = 2, legendfontsize = 7,
    foreground_color_legend = nothing, size = (605, 370), FIGURE_MARGINS...)
for (k, q) in enumerate(objective_quantiles)
    plot!(objective_figure, [k, k], [q[1], q[5]]; color = BENCHMARK_COLOR, linewidth = 1, label = k == 1 ? "1–99. percentilis" : false)
    plot!(objective_figure, Shape(k .+ box_half_width .* [-1, 1, 1, -1], [q[2], q[2], q[4], q[4]]);
        fillcolor = BOX_FILL_COLOR, linecolor = BENCHMARK_COLOR, linewidth = 1, label = k == 1 ? "25–75. percentilis" : false)
    plot!(objective_figure, k .+ box_half_width .* [-1, 1], [q[3], q[3]]; color = CITYWIDE_COLOR, linewidth = 2, label = k == 1 ? "medián" : false)
end
plot!(objective_figure, 1:4, best_objectives; color = HIGHLIGHT_COLOR, linewidth = 0.8,
    markershape = :circle, markersize = 3.5, markercolor = HIGHLIGHT_COLOR, markerstrokecolor = :white, markerstrokewidth = 0.8,
    label = "legjobb érték (minimum)")
for (k, value) in enumerate(best_objectives)
    annotate!(objective_figure, k + 0.07, value, hu_text(hu_significant(value), 7, :left, :top))
end
savefig(objective_figure, figure_path("objective.svg"))

# C: each iteration's best point, simulated rent per submarket as a
# percentage deviation from the empirical median rent (the terms the
# objective squares). Stems from zero, one marker per iteration; only the
# final iteration's values are written out.
submarket_labels = ["$(replace(SUBMARKET_NAMES[d], " – " => "\n"))\n($(Int(EMPIRICAL_RENTS[d])) ezer Ft)" for d in 1:6]
iteration_offsets = [-0.27, -0.09, 0.09, 0.27]
rent_fit_figure = plot(; xticks = (1:6, submarket_labels), xlims = (0.5, 6.5),
    yformatter = hu_number, ylabel = "eltérés az empirikus mediántól (%)",
    title = "Az iterációk legjobb pontja: szimulált vs. empirikus albérleti díj",
    PANEL_STYLE..., titlefontsize = 9, legend = :outerbottom, legend_column = 2, legendfontsize = 7,
    foreground_color_legend = nothing, size = (605, 350), FIGURE_MARGINS...)
hline!(rent_fit_figure, [0]; color = CITYWIDE_COLOR, linewidth = 0.8, label = false)
for k in 1:4
    x = (1:6) .+ iteration_offsets[k]
    y = rent_deviation_pct(best_rows[k])
    stems_x = vcat([[xi, xi, NaN] for xi in x]...)
    stems_y = vcat([[0, yi, NaN] for yi in y]...)
    plot!(rent_fit_figure, stems_x, stems_y; color = ITERATION_COLORS[k], linewidth = 1.2, label = false)
    scatter!(rent_fit_figure, x, y; color = ITERATION_COLORS[k], markershape = ITERATION_MARKERS[k], markersize = 4,
        markerstrokewidth = 0, label = k == 4 ? "4. iteráció (végső becslés)" : "$k. iteráció")
end
for (xi, yi) in zip((1:6) .+ iteration_offsets[4], rent_deviation_pct(best_rows[4]))
    # One decimal throughout ("1,0", not "1"); + 0.0 turns a rounded -0.0 into 0.0.
    annotate!(rent_fit_figure, xi + 0.07, yi, hu_text(replace(string(round(yi, digits = 1) + 0.0), "." => ","), 6, :left, yi < 0 ? :top : :bottom))
end
savefig(rent_fit_figure, figure_path("rent_fit.svg"))

# D: lowest objective within equal-width bins of each amenity (the other four
# amenities vary freely within the box). Top row: iteration 1 over the full
# [1, 10] range; bottom row: iteration 4 inside its narrow box. The minimum
# over a bin approximates the objective's profile along that amenity, so a
# sharp valley means the rents pin the amenity down. The dotted line is the
# final estimate. Each row shares its y-axis so the amenities are comparable.
# Lines are solid in every panel: each panel holds one series, named by its
# title, and the dashed premium style made the jagged bin minima hard to read.
function binned_minimum(values, objective, lo, hi, n_bins)
    edges = range(lo, hi, length = n_bins + 1)
    minima = fill(Inf, n_bins)
    for (v, o) in zip(values, objective)
        bin = clamp(searchsortedlast(edges, v), 1, n_bins)
        minima[bin] = min(minima[bin], o)
    end
    minima[isinf.(minima)] .= NaN
    return (edges[1:(end - 1)] .+ edges[2:end]) ./ 2, minima
end

const PROFILE_ROWS = [(iteration = 1, n_bins = 40), (iteration = 4, n_bins = 25)]
profiles = [[binned_minimum(results[row.iteration][!, column], results[row.iteration].objective_per_diff,
    ITERATIONS[row.iteration].lower[a], ITERATIONS[row.iteration].upper[a], row.n_bins)
    for (a, column) in enumerate(AMENITY_COLUMNS)] for row in PROFILE_ROWS]

println("Binned minimum of the objective along each amenity (iteration, amenity, bin centre, minimum):")
profile_table = DataFrame(iteration = Int[], amenity = Int[], bin_centre = Float64[], min_objective = Float64[])
for (r, row) in enumerate(PROFILE_ROWS), (a, (centres, minima)) in enumerate(profiles[r]), (c, m) in zip(centres, minima)
    push!(profile_table, (row.iteration, FREE_DISTRICTS[a], c, m))
end
show(profile_table, allrows = true, allcols = true)
println()

profile_panels = Plots.Plot[]
for (r, row) in enumerate(PROFILE_ROWS)
    row_range = extrema(filter(!isnan, vcat([m for (_, m) in profiles[r]]...)))
    row_ylims = (row_range[1] / 1.25, row_range[2] * 1.25)
    row_yticks = filter(t -> row_ylims[1] <= t <= row_ylims[2], [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    for (a, district) in enumerate(FREE_DISTRICTS)
        centres, minima = profiles[r][a]
        lo, hi = ITERATIONS[row.iteration].lower[a], ITERATIONS[row.iteration].upper[a]
        panel = plot(centres, minima; color = submarket_color(district), linewidth = 1.2,
            title = r == 1 ? replace(SUBMARKET_NAMES[district], " – " => "\n") : "",
            xlims = (lo, hi), xticks = hu_ticks(lo, hi),
            yscale = :log10, ylims = row_ylims, yticks = (row_yticks, a == 1 ? hu_number.(row_yticks) : fill("", length(row_yticks))),
            ylabel = a == 1 ? "$(row.iteration). iteráció" : "", xlabel = r == 2 && a == 3 ? "amenitásszint" : "",
            PANEL_STYLE..., titlefontsize = 7)
        vline!(panel, [best_rows[4][AMENITY_COLUMNS[a]]]; color = BENCHMARK_COLOR, linestyle = :dot, linewidth = 1)
        push!(profile_panels, panel)
    end
end
profile_legend = plot(; LEGEND_STYLE..., legend = :top, legend_column = 2)
plot!(profile_legend, [NaN], [NaN]; label = "legkisebb célfüggvény-érték sávonként", color = CITYWIDE_COLOR, linewidth = 1.2)
plot!(profile_legend, [NaN], [NaN]; label = "végső becslés (a 4. iteráció legjobb pontja)", color = BENCHMARK_COLOR, linestyle = :dot, linewidth = 1)
profile_figure = plot(profile_panels..., profile_legend;
    layout = @layout([grid(2, 5); a{0.07h}]), size = (605, 390),
    plot_title = "A célfüggvény legkisebb értéke az egyes amenitások mentén (logaritmikus skála)",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(profile_figure, figure_path("profile.svg"))

# E: iteration 4's best 1% of points, every pair of amenities (lower
# triangle), each axis spanning iteration 4's search box. A tilted cloud means
# two amenities can make up for each other; a cloud against an edge means the
# best points reach the box's boundary. No correlation coefficients on the
# panels: their size depends on where the "best" cut is drawn (printed above).
pair_panels = Plots.Plot[]
for r in 1:4, c in 1:4
    y_index, x_index = r + 1, c
    if x_index >= y_index
        push!(pair_panels, plot(; framestyle = :none, grid = false, legend = false))
        continue
    end
    x_lo, x_hi = ITERATIONS[4].lower[x_index], ITERATIONS[4].upper[x_index]
    y_lo, y_hi = ITERATIONS[4].lower[y_index], ITERATIONS[4].upper[y_index]
    x_values, y_values = final_top[!, AMENITY_COLUMNS[x_index]], final_top[!, AMENITY_COLUMNS[y_index]]
    panel = scatter(x_values, y_values; color = CITYWIDE_COLOR, markersize = 1.8, markeralpha = 0.45, markerstrokewidth = 0,
        xlims = (x_lo, x_hi), ylims = (y_lo, y_hi),
        xticks = r == 4 ? hu_ticks(x_lo, x_hi) : (hu_ticks(x_lo, x_hi)[1], fill("", length(hu_ticks(x_lo, x_hi)[1]))),
        yticks = c == 1 ? hu_ticks(y_lo, y_hi) : (hu_ticks(y_lo, y_hi)[1], fill("", length(hu_ticks(y_lo, y_hi)[1]))),
        xlabel = r == 4 ? SUBMARKET_SHORT_NAMES[FREE_DISTRICTS[x_index]] : "",
        ylabel = c == 1 ? SUBMARKET_SHORT_NAMES[FREE_DISTRICTS[y_index]] : "",
        PANEL_STYLE..., guidefontsize = 7)
    scatter!(panel, [best_rows[4][AMENITY_COLUMNS[x_index]]], [best_rows[4][AMENITY_COLUMNS[y_index]]];
        color = HIGHLIGHT_COLOR, markershape = :diamond, markersize = 4, markerstrokecolor = :white, markerstrokewidth = 0.8)
    push!(pair_panels, panel)
end
pair_legend = pair_panels[4]
scatter!(pair_legend, [NaN], [NaN]; label = "a 4. iteráció legjobb $top_percent_label-a ($(nrow(final_top)) pont)",
    color = CITYWIDE_COLOR, markersize = 2.5, markerstrokewidth = 0, markeralpha = 0.6)
scatter!(pair_legend, [NaN], [NaN]; label = "végső becslés", color = HIGHLIGHT_COLOR, markershape = :diamond, markersize = 4,
    markerstrokecolor = :white, markerstrokewidth = 0.8)
plot!(pair_legend; legend = :topright, legendfontsize = 7, fontfamily = FIGURE_FONT,
    foreground_color_legend = nothing, background_color_legend = nothing, xlims = (0, 1), ylims = (0, 1))
annotate!(pair_legend, 1, 0.35, hu_text("A tengelyek a 4. iteráció\nkeresési tartományát fedik le.", 7, :right))
pair_figure = plot(pair_panels...; layout = (4, 4), size = (605, 560),
    plot_title = "Amenitásszintek páronként a 4. iteráció legjobb $top_percent_label-ában",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(pair_figure, figure_path("pairs.svg"))
