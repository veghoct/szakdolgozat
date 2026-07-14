mutable struct DistrictAgent
    id::Int
    zone_id::Int
    name::String
    amenity::Float64
    rent::Float64
    units::Int
    occupied_units::Int
    mean_income::Float64
    median_income::Float64
end

mutable struct ResidentAgent
    id::Int
    type::Int
    utility::Union{Float64, Nothing}
    income::Float64
    target::Union{DistrictAgent, Nothing}
    home::DistrictAgent
end

mutable struct SegregationModel
    utility_alpha::Float64
    utility_beta::Float64
    price_change::Float64

    affordability_rate::Float64
    natural_vacancy_rate::Float64

    number_of_residents::Int

    income_distribution::Function

    rng::MersenneTwister

    districts::Vector{DistrictAgent}
    residents::Vector{ResidentAgent}

    mean_income_by_zone::Dict{Int, Float64}

    model_data::DataFrame
    district_data::DataFrame

    step_count::Int
end