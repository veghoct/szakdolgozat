include("../util/imports.jl")
include("../model/interfaces.jl")
include("../model/model.jl")
include("../model/segregation.jl")
include("../model/distributions.jl")

# Sensitivity of the model to a single district's unit count (capacity):
# for one district d at a time, sweep d's unit count across
# [0.75, 1.25] × d's benchmark units (20 sweep points, rounded to the
# nearest integer), holding every other district at its benchmark unit
# count and every other model parameter at its benchmark value. Repeated
# once per district. At each sweep point, the equilibrium-window outcome
# is averaged over 30 seeded replicates (seeds 260927..260956).

const BENCHMARK_UNITS = [142, 151, 363, 117, 170, 20]

function run_once(run_id, target_district_idx, target_units; steps = 150)
    units = copy(BENCHMARK_UNITS)
    units[target_district_idx] = target_units

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
        amenity = [
            1.0950729919433595,
            2.8319473571777345,
            1.1353562103271484,
            2.319447576904297,
            1,
            2.5699244720458987,
        ],
        rent = [
            130,
            130,
            130,
            130,
            130,
            130,
        ],
        units = units,
    )

    utility_alpha = 0.4026
    utility_beta = 0.1433
    price_change = 0.0980
    affordability_rate = 1
    number_of_households = 799

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
# (the index is undefined, not zero); skip those rather than letting mean()
# propagate a single NaN sample into the eq-window average.
nanmean(xs) = (valid = filter(!isnan, xs); isempty(valid) ? NaN : mean(valid))

steps = 1000
n_sweep_points = 20
replicate_seeds = 260927:260956
district_ids = 1:6

# Thesis figure (Hungarian, sized for \includesvg at \textwidth = 16 cm):
# equilibrium rent per submarket, citywide dissimilarity and citywide exposure
# against the swept parameter, as one 1x3 SVG. Vacancy is printed above but
# not plotted: in every submarket it settles at 1 - households/units by
# construction of the rent rule, so it carries no information here.
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
const BENCHMARK_COLOR = "#8a8a8a"
# serif-roman (Computer Modern) is embedded as outlines, so the SVG needs no
# installed font and covers the Hungarian and Greek characters used below.
const FIGURE_FONT = "serif-roman"

submarket_color(id) = ZONE_COLORS[cld(id, 2)]
submarket_linestyle(id) = isodd(id) ? :solid : :dash
hu_number(x) = (r = round(x, digits = 4); isinteger(r) ? string(Int(r)) : replace(string(r), "." => ","))

function hu_ticks(lo, hi)
    ticks = Plots.PlotUtils.optimize_ticks(lo, hi; k_min = 3, k_max = 5)[1]
    return (ticks, hu_number.(ticks))
end

function thesis_figure(rent_df, x_column, dissimilarity_df, exposure_df; xlabel, title, benchmark, path)
    x = dissimilarity_df[!, x_column]
    common = (
        fontfamily = FIGURE_FONT, xlabel = xlabel, xticks = hu_ticks(extrema(x)...), yformatter = hu_number,
        titlefontsize = 8, guidefontsize = 8, tickfontsize = 7, legend = false,
        linewidth = 1.2, markershape = :circle, markersize = 2, markerstrokewidth = 0,
        gridalpha = 0.15, framestyle = :axes,
    )

    rent_panel = plot(; title = "Egyensúlyi albérleti díj (ezer Ft)", common...)
    for id in 1:6
        sub = sort(unique(rent_df[rent_df.id .== id, :], x_column), x_column)
        plot!(rent_panel, sub[!, x_column], sub.rent; color = submarket_color(id), markercolor = submarket_color(id), linestyle = submarket_linestyle(id))
    end
    dissimilarity_panel = plot(x, dissimilarity_df.dissimilarity_three_groups; title = "Disszimilaritási index", color = CITYWIDE_COLOR, markercolor = CITYWIDE_COLOR, common...)
    exposure_panel = plot(exposure_df[!, x_column], exposure_df.exposure_three_groups; title = "Kitettségi index", color = CITYWIDE_COLOR, markercolor = CITYWIDE_COLOR, common...)
    for panel in (rent_panel, dissimilarity_panel, exposure_panel)
        vline!(panel, [benchmark]; color = BENCHMARK_COLOR, linestyle = :dot, linewidth = 1, label = false)
    end

    # Shared legend: one row per zone (affordable | premium), then the
    # citywide index line and the benchmark marker.
    legend_panel = plot(; framestyle = :none, legend = :top, legend_column = 2, legendfontsize = 7, fontfamily = FIGURE_FONT,
        foreground_color_legend = nothing, background_color_legend = nothing)
    for id in 1:6
        plot!(legend_panel, [NaN], [NaN]; label = SUBMARKET_NAMES[id], color = submarket_color(id), linestyle = submarket_linestyle(id), linewidth = 1.2)
    end
    plot!(legend_panel, [NaN], [NaN]; label = "városi index", color = CITYWIDE_COLOR, linewidth = 1.2)
    plot!(legend_panel, [NaN], [NaN]; label = "referenciaérték", color = BENCHMARK_COLOR, linestyle = :dot, linewidth = 1)

    figure = plot(rent_panel, dissimilarity_panel, exposure_panel, legend_panel;
        layout = @layout([a b c; d{0.24h}]), size = (605, 320),
        plot_title = title, plot_titlefontsize = 9, plot_titlefontfamily = FIGURE_FONT,
        left_margin = 1Plots.mm, right_margin = 1Plots.mm, top_margin = 0Plots.mm, bottom_margin = 1Plots.mm)
    savefig(figure, path)
end

# Hungarian title of a per-submarket sweep: "Az 1. övezet prémium részpiaci
# ... hatása" ("Az" before 1, "A" before 2 and 3).
submarket_title(id, noun) = string(cld(id, 2) == 1 ? "Az" : "A", " ", cld(id, 2), ". övezet ", isodd(id) ? "megfizethető" : "prémium", " részpiaci ", noun, " hatása")

mkpath(joinpath(@__DIR__, "district_units_per_district"))

for target_district_idx in 1:6
    target_district_id = district_ids[target_district_idx]
    benchmark_units_d = BENCHMARK_UNITS[target_district_idx]
    unit_values = [round(Int, u) for u in range(0.75 * benchmark_units_d, 1.25 * benchmark_units_d, length = n_sweep_points)]

    rent_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
    vacancy_rate_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds), length(district_ids))
    dissimilarity_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))
    exposure_by_replicate = Array{Float64}(undef, n_sweep_points, length(replicate_seeds))

    Threads.@threads for sweep_idx in 1:n_sweep_points
        target_units = unit_values[sweep_idx]

        for (rep_idx, seed) in enumerate(replicate_seeds)
            result = run_once(seed, target_district_idx, target_units; steps = steps)

            district_eq = filter(:step => step -> step > maximum(result.district_data.step) - 50, result.district_data)
            model_eq = filter(:step => step -> step > maximum(result.model_data.step) - 50, result.model_data)

            district_means = sort(
                combine(groupby(district_eq, :id),
                    :rent => mean => :rent_mean,
                    :vacancy_rate => mean => :vacancy_rate_mean,
                ),
                :id,
            )

            rent_by_replicate[sweep_idx, rep_idx, :] = district_means.rent_mean
            vacancy_rate_by_replicate[sweep_idx, rep_idx, :] = district_means.vacancy_rate_mean
            dissimilarity_by_replicate[sweep_idx, rep_idx] = nanmean(model_eq.dissimilarity_three_groups)
            exposure_by_replicate[sweep_idx, rep_idx] = nanmean(model_eq.exposure_three_groups)
        end
    end

    rent_by_units = DataFrame(units = Int[], id = Int[], rent = Float64[])
    vacancy_rate_by_units = DataFrame(units = Int[], id = Int[], vacancy_rate = Float64[])

    for sweep_idx in 1:n_sweep_points
        for (d_idx, id) in enumerate(district_ids)
            push!(rent_by_units, (units = unit_values[sweep_idx], id = id, rent = mean(rent_by_replicate[sweep_idx, :, d_idx])))
            push!(vacancy_rate_by_units, (units = unit_values[sweep_idx], id = id, vacancy_rate = mean(vacancy_rate_by_replicate[sweep_idx, :, d_idx])))
        end
    end

    # unique(...; keep = :first) drops the exact-duplicate rows produced when
    # [0.75, 1.25] × benchmark units spans fewer distinct integers than
    # n_sweep_points (small-unit districts); every duplicate has identical
    # units and identical replicate seeds, hence identical output.
    dissimilarity_by_units = unique(DataFrame(
        units = unit_values,
        dissimilarity_three_groups = [nanmean(dissimilarity_by_replicate[i, :]) for i in 1:n_sweep_points],
    ), :units)

    exposure_by_units = unique(DataFrame(
        units = unit_values,
        exposure_three_groups = [nanmean(exposure_by_replicate[i, :]) for i in 1:n_sweep_points],
    ), :units)

    println("=== District $target_district_id unit count sweep ===")

    # unit_values can contain duplicates when [0.75, 1.25] × benchmark units
    # spans fewer distinct integers than n_sweep_points (small-unit
    # districts); duplicate rows are exact repeats (same rounded units, same
    # replicate seeds => identical output), so combine = mean is a no-op.
    println("Equilibrium rent vs district $target_district_id unit count (columns = district id):")
    show(unstack(rent_by_units, :units, :id, :rent, combine = mean), allrows = true, allcols = true)
    println()

    println("Equilibrium vacancy rate vs district $target_district_id unit count (columns = district id):")
    show(unstack(vacancy_rate_by_units, :units, :id, :vacancy_rate, combine = mean), allrows = true, allcols = true)
    println()

    println("Equilibrium dissimilarity (three groups, citywide) vs district $target_district_id unit count:")
    show(dissimilarity_by_units, allrows = true)
    println()

    println("Equilibrium exposure (three groups, citywide) vs district $target_district_id unit count:")
    show(exposure_by_units, allrows = true)
    println()

    thesis_figure(rent_by_units, :units, dissimilarity_by_units, exposure_by_units;
        xlabel = "lakásállomány (db)",
        title = submarket_title(target_district_id, "lakásállományának"),
        benchmark = benchmark_units_d,
        path = joinpath(@__DIR__, "district_units_per_district", "d$(target_district_id).svg"),
    )
end
