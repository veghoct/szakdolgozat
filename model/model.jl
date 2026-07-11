mean_or_nan(xs) = isempty(xs) ? NaN : mean(xs)
median_or_nan(xs) = isempty(xs) ? NaN : median(xs)

function empty_model_data()
    DataFrame(
        step = Int[],
    )
end

function empty_district_data()
    DataFrame(
        step = Int[],
        id = Int[],
        zone_id = Int[],
        district = String[],
        rent = Float64[],
        vacancy_rate = Float64[],
    )
end

function collect_data!(model::SegregationModel)
    push!(model.model_data, (
        step = model.step_count,
    ))

    for district in model.districts
        push!(model.district_data, (
            step = model.step_count,
            id = district.id,
            zone_id = district.zone_id,
            district = district.name,
            rent = district.rent,
            vacancy_rate = 1 - district.occupied_units / district.units,
        ))
    end

    return nothing
end

function residential_choice_by_utility!(resident::ResidentAgent, model::SegregationModel)
    alpha = model.utility_alpha
    beta = model.utility_beta

    affordable_markets = [
            d for d in model.districts
            if (resident.income - d.rent) > model.minimum_disposable_income &&
            ((d.units - d.occupied_units) > 0 || resident.home === d)
        ]

    function utility(market::DistrictAgent)
        status = model.mean_income_by_zone[market.zone_id]
        return (1 - alpha) * log(resident.income - market.rent) +
               alpha * (1 - beta) * log(market.amenity) +
               alpha * beta * log(status)
    end

    sort!([affordable_markets], by = utility)

    if isempty(affordable_markets)
        return nothing
    end

    highest_utility_market = pop!(affordable_markets)

    if highest_utility_market === resident.home
        return nothing
    end

    resident.target = highest_utility_market

    return nothing
end

function auction_and_move_attempt!(district::DistrictAgent, model::SegregationModel)
    prospective_residents = sort(
        [a for a in model.residents if a.target === district],
        by = r -> r.income,
    )

    leaving_residents = [a for a in model.residents if a.home === district && a.target !== nothing]

    while district.units - district.occupied_units > 0 && !isempty(prospective_residents)
        moving_resident = pop!(prospective_residents)

        moving_resident.home.occupied_units -= 1
        moving_resident.target.occupied_units += 1

        moving_resident.home = moving_resident.target
        moving_resident.target = nothing
    end

    for resident in prospective_residents
        resident.target = nothing
    end

    return nothing
end

function rent_hike!(district::DistrictAgent, model::SegregationModel)
    vacancy_rate = 1 - (district.occupied_units / district.units)

    if vacancy_rate > model.natural_vacancy_rate
      district.rent *= (1 - model.price_change)
    end

    if vacancy_rate < model.natural_vacancy_rate
      district.rent *= (1 + model.price_change)
    end
end

function remove_and_replace_late_agents!(model::SegregationModel)
    filter!(model.residents) do resident

        isAnyAffordable = false
        for district in model.districts
            if district.rent <= (resident.income - model.minimum_disposable_income)
                isAnyAffordable = true
                break
            end
        end

        if (isAnyAffordable === false)
            resident.home.occupied_units -= 1
        end

        return isAnyAffordable
    end

    for _ in 1:(model.number_of_residents - length(model.residents))
        income = model.income_distribution(model.rng)

        available_markets = [d for d in model.districts if (income - d.rent) > model.minimum_disposable_income && (d.units - d.occupied_units) > 0]
        
        if (length(available_markets) === 0)
            continue
        end

        district = rand(available_markets)

        resident = ResidentAgent(0, 0, nothing, income, nothing, district)

        push!(model.residents, resident)
    end

    return nothing
end

function calculate_district_group_mean_income!(model::SegregationModel)
    empty!(model.mean_income_by_zone)

    for district in model.districts
        incomes = [a.income for a in model.residents if a.home.zone_id == district.zone_id]
        model.mean_income_by_zone[district.zone_id] = mean_or_nan(incomes)
    end

    return nothing
end

function step!(model::SegregationModel)
    model.step_count += 1

    calculate_district_group_mean_income!(model)

    for resident in model.residents
        residential_choice_by_utility!(resident, model)
    end
    
    for district in model.districts
        auction_and_move_attempt!(district, model)
    end

    for district in model.districts
        rent_hike!(district, model)
    end

    collect_data!(model)

    remove_and_replace_late_agents!(model)

    return nothing
end

function SegregationModel(
        income_distribution,
        utility_alpha,
        utility_beta,
        price_change,
        minimum_disposable_income,
        number_of_residents,
        districts_df::DataFrame,
        seed = nothing
    )

    rng = seed === nothing ? MersenneTwister() : MersenneTwister(seed)

    districts = DistrictAgent[]
    for (idx, row) in enumerate(eachrow(districts_df))
        push!(districts, DistrictAgent(
            row[:id],
            row[:zone_id],
            String(row[:name]),
            Float64(row[:amenity]),
            Float64(row[:rent]),
            Int(row[:units]),
            0,
            NaN,
            NaN,
        ))
    end

    homes = DistrictAgent[]
    for d in copy(districts)
        for _ in 1:d.units
            push!(homes, d)
        end
    end
    shuffle!(rng, homes)

    residents = ResidentAgent[]
    for i in 1:Int(number_of_residents)
        homes[i].occupied_units += 1
        push!(residents, ResidentAgent(i, 0, nothing, income_distribution(rng), nothing, homes[i]))
    end

    natural_vacancy_rate = 1 - number_of_residents / length(homes)

    model = SegregationModel(
        Float64(utility_alpha),
        Float64(utility_beta),
        Float64(price_change),
        Float64(minimum_disposable_income),
        Float64(natural_vacancy_rate),
        Int(number_of_residents),
        income_distribution,
        rng,
        districts,
        residents,
        Dict{String, Float64}(),
        empty_model_data(),
        empty_district_data(),
        0,
    )

    return model
end

function run_for!(model::SegregationModel, steps::Integer)
    for _ in 1:steps
        step!(model)
    end

    return model
end