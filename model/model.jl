mean_or_nan(xs) = isempty(xs) ? NaN : mean(xs)
median_or_nan(xs) = isempty(xs) ? NaN : median(xs)

function empty_model_data()
    DataFrame(
        step = Int[],
        dissimilarity_10 = Float64[],
        dissimilarity_20 = Float64[],
        dissimilarity_50 = Float64[],
        dissimilarity_80 = Float64[],
        dissimilarity_90 = Float64[],
        dissimilarity_three_groups = Float64[],
        exposure_10 = Float64[],
        exposure_20 = Float64[],
        exposure_50 = Float64[],
        exposure_80 = Float64[],
        exposure_90 = Float64[],
        exposure_three_groups = Float64[],
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

function empty_zone_data()
    DataFrame(
        step = Int[],
        id = Int[],
        average_income = Float64[],
        median_income = Float64[],
        gini = Float64[],
        share_bottom_10 = Float64[],
        share_bottom_20 = Float64[],
        share_bottom_50 = Float64[],
        share_top_20 = Float64[],
        share_top_10 = Float64[],
    )
end

function collect_data!(model::SegregationModel)
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

    all_incomes = [r.income for r in model.residents]
    lower_tail_cutoffs = Dict(p => percentile_cutoffs(all_incomes, [p / 100]) for p in (10, 20, 50, 80, 90))
    cutoffs_three_groups = percentile_cutoffs(all_incomes, [0.2, 0.8])

    push!(model.model_data, model_segregation_row(model, lower_tail_cutoffs, cutoffs_three_groups))

    for zone_id in unique(d.zone_id for d in model.districts)
        residents = filter(r -> r.home !== nothing && r.home.zone_id == zone_id, model.residents)

        push!(model.zone_data, zone_income_row(zone_id, residents, model.step_count, lower_tail_cutoffs))
    end

    return nothing
end

function preference_list(resident::ResidentAgent, model::SegregationModel)
    alpha = model.utility_alpha
    beta = model.utility_beta

    function utility(district::DistrictAgent)
        status = model.mean_income_by_zone[district.zone_id]

        return (1 - alpha) * log(resident.income - district.rent) +
               alpha * (1 - beta) * log(district.amenity) +
               alpha * beta * log(status)
    end

    affordable = [district for district in model.districts if district.rent / resident.income <= model.affordability_rate]
    unaffordable = [district for district in model.districts if district.rent / resident.income > model.affordability_rate]

    sort!(affordable, by = utility, rev = true)
    sort!(unaffordable, by = district -> district.rent)

    return vcat(affordable, unaffordable)
end

function multi_round_matching(model::SegregationModel)
    for resident in sort(model.residents, by = r -> r.income, rev = true)
        targets = preference_list(resident, model)

        while !isempty(targets)
            target = popfirst!(targets)

            if (target === resident.home)
                break
            end

            if (target.units === target.occupied_units)
                continue
            end

            target.occupied_units += 1
            resident.home.occupied_units -= 1
            resident.home = target
            break
        end
    end
end

function rent_hike!(district::DistrictAgent, model::SegregationModel)
    target_occupied = (1 - model.natural_vacancy_rate) * district.units

    occupancy_gap = (district.occupied_units - target_occupied) / district.units
    
    district.rent = max(district.rent * (1 + model.price_change)^occupancy_gap, 120)
end

function calculate_district_group_mean_income!(model::SegregationModel)
    empty!(model.mean_income_by_zone)

    for district in model.districts
        incomes = [a.income for a in model.residents if a.home !== nothing && a.home.zone_id == district.zone_id]
        model.mean_income_by_zone[district.zone_id] = mean_or_nan(incomes)
    end

    return nothing
end

function step!(model::SegregationModel)
    model.step_count += 1

    calculate_district_group_mean_income!(model)
    
    multi_round_matching(model)

    for district in model.districts
        rent_hike!(district, model)
    end

    collect_data!(model)

    return nothing
end

function SegregationModel(
        income_distribution,
        utility_alpha,
        utility_beta,
        price_change,
        affordability_rate,
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
        push!(residents, ResidentAgent(i, 0, nothing, income_distribution(rng), nothing, [], homes[i]))
    end

    natural_vacancy_rate = 1 - number_of_residents / length(homes)

    model = SegregationModel(
        Float64(utility_alpha),
        Float64(utility_beta),
        Float64(price_change),
        Float64(affordability_rate),
        Float64(natural_vacancy_rate),
        Int(number_of_residents),
        income_distribution,
        rng,
        districts,
        residents,
        Dict{String, Float64}(),
        empty_model_data(),
        empty_district_data(),
        empty_zone_data(),
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