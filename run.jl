include("imports.jl")
include("interfaces.jl")
include("model.jl")

function run_once(run_id; steps = 5000, lux_units = 100, com_units = 400)
    districts = DataFrame(
        name = ["Downtown", "Downtown", "Rest of City"],
        level = ["Luxury", "Common", "Common"],
        amenity = [2000, 1750, 1250],
        rent = [1000, 1000, 1000],
        units = [10, 40, 50],
    )

    model = SegregationModel(2400, 0.75, 0.75, 0, 0, 90, districts)

    run_for!(model, steps)

    agent_data = copy(model.district_data)
    agent_data.run_id = fill(run_id, nrow(agent_data))
    agent_data.lux_units = fill(lux_units, nrow(agent_data))
    agent_data.com_units = fill(com_units, nrow(agent_data))

    return agent_data
end

function lineplot_grouped!(plt, df::DataFrame, xcol::Symbol, ycol::Symbol; groupcol::Symbol, subplot::Int = 1, xlabel = "", ylabel = "", title = "", marker = :none)
    agg = combine(groupby(df, [xcol, groupcol]), ycol => mean => :y)
    sort!(agg, [groupcol, xcol])

    for group_df in groupby(agg, groupcol)
        label = string(first(group_df[!, groupcol]))
        plot!(
            plt,
            group_df[!, xcol],
            group_df[!, :y],
            subplot = subplot,
            label = label,
            xlabel = xlabel,
            ylabel = ylabel,
            title = title,
            marker = marker,
        )
    end

    return plt
end

runs = 100
steps = 2000

all_runs = vcat([run_once(i; steps = steps) for i in 0:(runs - 1)]...)

@assert nrow(all_runs) == runs * steps * 3
@assert all([:Step, :rent, :agent_uid, :run_id, :lux_units, :com_units] .∈ Ref(propertynames(all_runs)))

using DataFrames

avg_last_rent = all_runs |>
    x -> filter(:Step => s -> s >= maximum(x.Step) - 499, x) |>
    x -> combine(groupby(x, :agent_uid), :rent => mean => :avg_rent)

print(avg_last_rent)

rent_plot = plot(size = (900, 500))

savefig(lineplot_grouped!(
    rent_plot,
    all_runs,
    :Step,
    :rent,
    groupcol = :agent_uid,
    xlabel = "Step",
    ylabel = "rent",
    title = "Rent over time",
), "rent.png")