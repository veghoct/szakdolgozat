mean_or_nan(xs) = isempty(xs) ? NaN : mean(xs)
median_or_nan(xs) = isempty(xs) ? NaN : median(xs)

function empty_model_data()
    DataFrame(
        Step = Int[],
    )
end

function empty_district_data()
    DataFrame(
        Step = Int[],
        AgentID = Int[],
        district = String[],
        level = String[],
        rent = Float64[],
        agent_uid = String[],
    )
end

function SegregationModel(mean_param, sigma, alpha, beta, P, residents, districts_df::DataFrame, seed = nothing)
    rng = seed === nothing ? MersenneTwister() : MersenneTwister(seed)

    districts = DistrictAgent[]
    for (idx, row) in enumerate(eachrow(districts_df))
        push!(districts, DistrictAgent(
            idx,
            String(row[:name]),
            String(row[:level]),
            Float64(row[:amenity]),
            Float64(row[:rent]),
            Int(row[:units]),
            NaN,
            NaN,
        ))
    end

    district_agents = sort(copy(districts), by = d -> d.amenity)

    homes = DistrictAgent[]
    for d in district_agents
        for _ in 1:d.units
            push!(homes, d)
        end
    end
    shuffle!(rng, homes)

    mu = Float64(log(mean_param) - 0.5 * (sigma^2))
    incomes = exp.(mu .+ Float64(sigma) .* randn(rng, Int(residents)))

    resident_agents = ResidentAgent[]
    for i in 1:Int(residents)
        push!(resident_agents, ResidentAgent(i, 0, nothing, Float64(incomes[i]), nothing, homes[i]))
    end

    model = SegregationModel(
        Float64(mean_param),
        Float64(sigma),
        Float64(alpha),
        Float64(beta),
        Float64(P),
        Int(residents),
        rng,
        districts,
        resident_agents,
        Dict{String, Float64}(),
        empty_model_data(),
        empty_district_data(),
        0,
    )

    return model
end

function collect_data!(model::SegregationModel)
    push!(model.model_data, (
        Step = model.step_count,
    ))

    for district in model.districts
        push!(model.district_data, (
            Step = model.step_count,
            AgentID = district.id,
            district = district.name,
            level = district.level,
            rent = district.rent,
            agent_uid = "$(district.name):$(district.level)",
        ))
    end

    return nothing
end

function move_attempt_by_utility!(resident::ResidentAgent, model::SegregationModel)
    alpha = model.alpha
    beta = model.beta

    affordable_blocks = [d for d in model.districts if d.rent / resident.income < 0.5]

    function utility(block::DistrictAgent)
        status = model.mean_income_by_district_group[block.name]
        return alpha * log(resident.income - block.rent) +
               (1 - alpha) * (1 - beta) * log(block.amenity) +
               (1 - alpha) * beta * log(status)
    end

    sort!(affordable_blocks, by = utility)

    if isempty(affordable_blocks)
        return nothing
    end

    highest_utility_block = pop!(affordable_blocks)

    if highest_utility_block === resident.home
        return nothing
    end

    resident.target = highest_utility_block
    return nothing
end

function auction!(district::DistrictAgent, model::SegregationModel)
    current_residents = [a for a in model.residents if a.home === district]
    prospective_residents = sort(
        [a for a in model.residents if a.target === district],
        by = r -> r.income,
    )

    available = district.units - length(current_residents)

    while available > 0 && !isempty(prospective_residents)
        moving_resident = pop!(prospective_residents)
        moving_resident.home = moving_resident.target::DistrictAgent
        moving_resident.target = nothing
        available -= 1
    end

    for resident in prospective_residents
        resident.target = nothing
    end

    return nothing
end

function rent_hike!(district::DistrictAgent, model::SegregationModel)
    current_residents = [a for a in model.residents if a.home === district]
    vacancy_rate = 1 - length(current_residents) / district.units

    if vacancy_rate > 0.15
      district.rent *= 0.95
    end

    if vacancy_rate < 0.15
      district.rent *= 1.05
    end
end

function remove_and_replace_late_agents!(model::SegregationModel)
    filter!(model.residents) do resident
        any(d.rent / resident.income < 0.5 for d in model.districts)
    end

    missing_residents = model.agent_number - length(model.residents)

    occupancy = Dict(d => 0 for d in model.districts)
    for a in model.residents
        occupancy[a.home] += 1
    end

    for _ in 1:missing_residents
        mu = Float64(log(model.mean_param) - 0.5 * (model.sigma^2))
        income = exp(mu + Float64(model.sigma) * randn(model.rng))

        affordable_blocks = [d for d in model.districts if d.rent / income < 0.5 && occupancy[d] < d.units]

        if isempty(affordable_blocks)
            continue
        end

        chosen = rand(model.rng, affordable_blocks)
        occupancy[chosen] += 1

        new_resident = ResidentAgent(1, 0, nothing, Float64(income), nothing, chosen)

        push!(model.residents, new_resident)
    end

    return nothing
end

function calculate_district_group_mean_income!(model::SegregationModel)
    empty!(model.mean_income_by_district_group)

    for district in model.districts
        incomes = [a.income for a in model.residents if a.home.name == district.name]
        model.mean_income_by_district_group[district.name] = mean_or_nan(incomes)
    end

    return nothing
end

function step!(model::SegregationModel)
    model.step_count += 1

    calculate_district_group_mean_income!(model)

    for resident in model.residents
        move_attempt_by_utility!(resident, model)
    end
    
    for district in model.districts
        auction!(district, model)
    end

    for district in model.districts
        rent_hike!(district, model)
    end

    collect_data!(model)

    remove_and_replace_late_agents!(model)

    return nothing
end

function run_for!(model::SegregationModel, steps::Integer)
    for _ in 1:steps
        step!(model)
    end
    return model
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