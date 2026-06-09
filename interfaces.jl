mutable struct DistrictAgent
    id::Int
    name::String
    level::String
    amenity::Float64
    rent::Float64
    units::Int
    mean_income::Float64
    median_income::Float64
end

mutable struct ResidentAgent
    id::Int
    quartile::Int
    utility::Union{Float64, Nothing}
    income::Float64
    target::Union{DistrictAgent, Nothing}
    home::DistrictAgent
end

mutable struct SegregationModel
    mean_param::Float64
    sigma::Float64
    alpha::Float64
    beta::Float64
    P::Float64
    agent_number::Int
    rng::MersenneTwister
    districts::Vector{DistrictAgent}
    residents::Vector{ResidentAgent}
    mean_income_by_district_group::Dict{String, Float64}
    model_data::DataFrame
    district_data::DataFrame
    step_count::Int
end