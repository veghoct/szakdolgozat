include("../util/imports.jl")
using LinearAlgebra
using Printf

# Local identification diagnostics of the amenity calibration at its estimate:
# iteration 4's best point (sample 11049) of model/sobol_smm_calibration_iter4.jl.
# Reads the saved result CSVs only and runs no simulation. Prints the numbers
# behind every figure and saves three figures into sa/sobol_identification/:
#   sensitivity.svg - elasticity of each simulated rent with respect to each
#                     amenity (the rank condition of local identification)
#   eigen.svg       - curvature of the objective along its principal
#                     directions and what those directions are made of
#                     (eigenvalue spread, condition number)
#   collinearity.svg - Brun, Reichert & Künsch (2001) identifiability
#                      measures: sensitivity per amenity, collinearity index
#                      per amenity subset
#
# How the sensitivities are estimated: the model has no analytic derivative,
# so each of the six simulated rents is regressed on a full quadratic in the
# five free amenities over iteration 4's 24 999 points (all simulated with the
# same seeds), centred at the estimate; the linear coefficients are the
# gradient there. The objective is objective_per_diff, the sum of squared
# relative rent deviations, so rents are measured relative to their empirical
# targets and amenities in percent (log units): element (i, j) is the
# percentage change of rent i, relative to its target, per 1% change of
# amenity j. The curvature of the objective is then 2 S'S (Gauss-Newton).

const FREE_DISTRICTS = [1, 2, 3, 4, 6]
const AMENITY_COLUMNS = [Symbol("amenity_$d") for d in FREE_DISTRICTS]
const EMPIRICAL_RENTS = [138.0, 300.0, 130.0, 240.0, 120.0, 235.0]
const COLLINEARITY_RULE_OF_THUMB = 20

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

points = load_rent130_pass("sobol_smm_results_iter4.csv", "sobol_smm_results_iter4 copy.csv")
best_row = points[argmin(points.objective_per_diff), :]
estimate = [best_row[c] for c in AMENITY_COLUMNS]
println("Iteration 4, start-rent-130 pass: $(nrow(points)) points; estimate = sample $(best_row.sample_id), objective $(best_row.objective_per_diff), amenities (1, 2, 3, 4, 6) = $(round.(estimate, digits = 4))")

amenities = Matrix{Float64}(points[:, AMENITY_COLUMNS]) .- estimate'
linear_design = hcat(ones(nrow(points)), amenities)
quadratic_design = hcat(linear_design, reduce(hcat, [amenities[:, i] .* amenities[:, j] for i in 1:5 for j in i:5]))
r_squared(y, design) = (residual = y - design * (design \ y); 1 - sum(residual .^ 2) / sum((y .- mean(y)) .^ 2))

gradient = zeros(6, 5)
fit_rows = DataFrame(id = Int[], r2_linear = Float64[], r2_quadratic = Float64[])
for d in 1:6
    y = points[!, Symbol("model_rent_$d")]
    gradient[d, :] = (quadratic_design \ y)[2:6]
    push!(fit_rows, (d, r_squared(y, linear_design), r_squared(y, quadratic_design)))
end
println("Fit of each simulated rent on the amenities (R^2 of a linear and a full quadratic regression):")
show(fit_rows, allcols = true)
println()

# S: % change of rent i (relative to its target) per 1% change of amenity j.
sensitivity = gradient .* estimate' ./ EMPIRICAL_RENTS
println("Sensitivity matrix S at the estimate (rows = rent 1..6, columns = amenity 1, 2, 3, 4, 6):")
show(DataFrame(sensitivity, string.("amenity_", FREE_DISTRICTS)), allcols = true)
println()

singular_values = svdvals(sensitivity)
curvature = eigen(Symmetric(sensitivity' * sensitivity))
eigenvalues = curvature.values                      # ascending: weakest direction first
# An eigenvector's sign is arbitrary; make its largest loading positive.
directions = mapslices(v -> v .* sign(v[argmax(abs.(v))]), curvature.vectors; dims = 1)
condition_number = sqrt(eigenvalues[end] / eigenvalues[1])
println("Singular values of S: $(round.(singular_values, sigdigits = 3)) (rank $(rank(sensitivity)) of 5)")
println("Eigenvalues of S'S, weakest first: $(round.(eigenvalues, sigdigits = 3)); condition number sqrt(max/min) = $(round(condition_number, digits = 2))")
println("Eigenvectors (columns, weakest first; rows = amenity 1, 2, 3, 4, 6):")
show(DataFrame(directions, string.("direction_", 1:5)), allcols = true)
println()

# Brun et al. (2001): delta_msqr_j = root mean square of column j of S; the
# collinearity index of an amenity subset K is 1 / sqrt(smallest eigenvalue of
# S_K'S_K) with every column of S scaled to unit length.
delta_msqr = vec(sqrt.(mean(sensitivity .^ 2, dims = 1)))
unit_columns = sensitivity ./ sqrt.(sum(sensitivity .^ 2, dims = 1))
collinearity_index(subset) = 1 / sqrt(eigmin(Symmetric(unit_columns[:, subset]' * unit_columns[:, subset])))
subsets = [findall(j -> (mask >> (j - 1)) & 1 == 1, 1:5) for mask in 1:31 if count_ones(mask) >= 2]
collinearity_rows = DataFrame(size = length.(subsets), amenities = [FREE_DISTRICTS[s] for s in subsets], index = collinearity_index.(subsets))
println("Parameter sensitivity delta_msqr per amenity (1, 2, 3, 4, 6): $(round.(delta_msqr, digits = 3))")
println("Collinearity index per amenity subset (rule of thumb: identifiable below about $COLLINEARITY_RULE_OF_THUMB):")
show(sort(collinearity_rows, [:size, order(:index, rev = true)]), allrows = true, allcols = true)
println()

# Andrews, Gentzkow & Shapiro (2017) sensitivity of the estimate to the
# moments: % change of each amenity per +1 percentage point deviation of each
# simulated rent from its target, Lambda = -(S'S)^-1 S'.
moment_sensitivity = -inv(sensitivity' * sensitivity) * sensitivity'
println("Lambda (rows = amenity 1, 2, 3, 4, 6; columns = rent 1..6):")
show(DataFrame(moment_sensitivity, string.("rent_", 1:6)), allcols = true)
println()

# Thesis figures (Hungarian, sized for \includesvg at \textwidth = 16 cm), in
# the format of the sensitivity-analysis figures.
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
# Compact subset labels: zone number + M (megfizethető) / P (prémium).
const SUBMARKET_CODES = ["1M", "1P", "2M", "2P", "3M", "3P"]
const CITYWIDE_COLOR = "#333333"
const BENCHMARK_COLOR = "#8a8a8a"
# Diverging scale for signed values: red (negative) - neutral gray - blue
# (positive), the two arms matched step by step in OKLCH lightness
# (0.385, 0.622, 0.764; midpoint 0.952).
const DIVERGING = cgrad(["#762221", "#d75853", "#ea9a93", "#f0efec", "#86b6ef", "#3987e5", "#104281"])
# serif-roman (Computer Modern) is embedded as outlines, so the SVG needs no
# installed font and covers the Hungarian and Greek characters used below.
const FIGURE_FONT = "serif-roman"

hu_number(x) = (r = round(x, digits = 4); isinteger(r) ? string(Int(r)) : replace(string(r), "." => ","))
# Fixed decimals with a decimal comma; + 0.0 turns a rounded -0.0 into 0.0.
hu_fixed(x, digits) = replace(@sprintf("%.*f", digits, round(x, digits = digits) + 0.0), "." => ",")
hu_text(s, args...) = text(s, FIGURE_FONT, args...)

const PANEL_STYLE = (
    fontfamily = FIGURE_FONT, titlefontsize = 8, guidefontsize = 8, tickfontsize = 7, legend = false,
    gridalpha = 0.15, framestyle = :axes,
)
const FIGURE_MARGINS = (left_margin = 1Plots.mm, right_margin = 2Plots.mm, top_margin = 0Plots.mm, bottom_margin = 1Plots.mm)

mkpath(joinpath(@__DIR__, "sobol_identification"))
figure_path(filename) = joinpath(@__DIR__, "sobol_identification", filename)

# A signed matrix as a heatmap with its value in every cell (the cell labels
# double as the table view); ink flips to white on the dark ends of the scale.
function diverging_heatmap(values, limit; xlabels, ylabels, kwargs...)
    n_rows, n_columns = size(values)
    panel = heatmap(1:n_columns, 1:n_rows, values; c = DIVERGING, clims = (-limit, limit), colorbar = false, yflip = true,
        xticks = (1:n_columns, xlabels), yticks = (1:n_rows, ylabels), PANEL_STYLE..., grid = false, framestyle = :box, kwargs...)
    for i in 1:n_rows, j in 1:n_columns
        annotate!(panel, j, i, hu_text(hu_fixed(values[i, j], 2), 7, abs(values[i, j]) > 0.6 * limit ? :white : :black))
    end
    return panel
end

# 1: sensitivity matrix. Local identification needs it to have full column
# rank; a row of near-zeros is a moment that carries no information.
sensitivity_limit = ceil(maximum(abs.(sensitivity)), digits = 1)
sensitivity_panel = diverging_heatmap(sensitivity, sensitivity_limit;
    xlabels = SUBMARKET_SHORT_NAMES[FREE_DISTRICTS], ylabels = SUBMARKET_NAMES,
    xlabel = "amenitás (1%-os növelés)", ylabel = "szimulált albérleti díj")
sensitivity_figure = plot(sensitivity_panel; size = (605, 320),
    plot_title = "Az albérleti díjak rugalmassága az amenitásokra (rang: $(rank(sensitivity)) / 5)",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(sensitivity_figure, figure_path("sensitivity.svg"))

# 2: eigenvalues of S'S (curvature of the objective along each principal
# direction, log scale) and the directions' composition. A small eigenvalue is
# a parameter combination the rents barely respond to.
eigenvalue_ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
eigenvalue_panel = plot(1:5, eigenvalues; seriestype = :scatter, color = CITYWIDE_COLOR, markersize = 4, markerstrokewidth = 0,
    yscale = :log10, yticks = (eigenvalue_ticks, hu_number.(eigenvalue_ticks)), ylims = (0.02, 2.5),
    xticks = (1:5, string.(1:5, ".")), xlims = (0.5, 5.5), xlabel = "irány (a leggyengébbtől)", ylabel = "sajátérték",
    title = "Görbület irányonként", PANEL_STYLE...)
for (k, value) in enumerate(eigenvalues)
    annotate!(eigenvalue_panel, k + 0.15, value, hu_text(hu_number(round(value, sigdigits = 2)), 7, :left))
end
annotate!(eigenvalue_panel, 0.65, 2.1, hu_text("kondíciószám: $(hu_fixed(condition_number, 1))", 7, :left, :top))
direction_panel = diverging_heatmap(directions, 1.0;
    xlabels = string.(1:5, "."), ylabels = SUBMARKET_SHORT_NAMES[FREE_DISTRICTS],
    xlabel = "irány (a leggyengébbtől)", title = "Az irányok összetétele (sajátvektorok)")
eigen_figure = plot(eigenvalue_panel, direction_panel; layout = @layout([a{0.4w} b]), size = (605, 290),
    plot_title = "A célfüggvény görbülete a fő irányok mentén a végső becslésnél",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(eigen_figure, figure_path("eigen.svg"))

# 3: Brun et al. (2001) measures. Left: sensitivity of the rents to each
# amenity alone. Right: collinearity index of every subset of two or more
# amenities, against the rule-of-thumb limit; the largest index per subset
# size is labelled (to the right of its point; the single five-amenity subset
# to the left, so the label stays inside the panel).
delta_panel = bar(1:5, delta_msqr; color = CITYWIDE_COLOR, linecolor = :match, bar_width = 0.55,
    xticks = (1:5, SUBMARKET_CODES[FREE_DISTRICTS]), ylims = (0, 1.15 * maximum(delta_msqr)), yformatter = hu_number,
    ylabel = "érzékenység (δ msqr)", title = "Érzékenység amenitásonként", PANEL_STYLE...)
subset_code(amenity_ids) = join(SUBMARKET_CODES[amenity_ids], "+")
jitter = 0.18 .* (rand(MersenneTwister(2026), nrow(collinearity_rows)) .- 0.5)
collinearity_ticks = [1, 2, 5, 10, 20]
collinearity_panel = plot(; yscale = :log10, yticks = (collinearity_ticks, hu_number.(collinearity_ticks)), ylims = (0.9, 28),
    xticks = (2:5, string.(2:5)), xlims = (1.5, 5.5), xlabel = "részhalmaz mérete (amenitások száma)", ylabel = "kollinearitási index",
    title = "Kollinearitási index részhalmazonként", PANEL_STYLE...)
hline!(collinearity_panel, [COLLINEARITY_RULE_OF_THUMB]; color = BENCHMARK_COLOR, linestyle = :dash, linewidth = 1)
annotate!(collinearity_panel, 5.45, COLLINEARITY_RULE_OF_THUMB * 0.93, hu_text("hüvelykujjszabály: $COLLINEARITY_RULE_OF_THUMB", 7, :right, :top))
scatter!(collinearity_panel, collinearity_rows.size .+ jitter, collinearity_rows.index;
    color = CITYWIDE_COLOR, markersize = 3, markerstrokewidth = 0, markeralpha = 0.8)
for group in groupby(collinearity_rows, :size)
    top = group[argmax(group.index), :]
    label = top.size == 5 ? "mind az öt: $(hu_fixed(top.index, 1))" : "$(subset_code(top.amenities)): $(hu_fixed(top.index, 1))"
    annotate!(collinearity_panel, top.size + (top.size == 5 ? -0.15 : 0.15), top.index,
        hu_text(label, 7, top.size == 5 ? :right : :left, :vcenter))
end
collinearity_figure = plot(delta_panel, collinearity_panel; layout = @layout([a{0.38w} b]), size = (605, 300),
    plot_title = "Azonosíthatósági mérőszámok a végső becslésnél (M: megfizethető, P: prémium)",
    plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT, FIGURE_MARGINS...)
savefig(collinearity_figure, figure_path("collinearity.svg"))
