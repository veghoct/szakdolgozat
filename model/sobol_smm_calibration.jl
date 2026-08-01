using Random
using Statistics
using DataFrames
using CSV
using Distributions
using QuasiMonteCarlo

include(joinpath(@__DIR__, "interfaces.jl"))
include(joinpath(@__DIR__, "model.jl"))


# ============================================================
# SETTINGS — edit these
# ============================================================

const FIRST_SAMPLE = 1
const LAST_SAMPLE = 10000

const REPLICATIONS = 10
const STEPS = 150
const AVERAGING_WINDOW = 50
const BASE_SEED = 12_345

const OUTPUT_PATH = joinpath(@__DIR__, "../sobol_smm_results.csv")

const FIXED_AMENITY = 1

const LOWER_BOUNDS = [
    1.0,   # amenity_1
    4.0,   # amenity_2
    1.0,   # amenity_3
    3.0,   # amenity_4
    2.5,   # amenity_6
]

const UPPER_BOUNDS = [
    3.5,   # amenity_1
    10.0,  # amenity_2
    4.0,   # amenity_3
    10.0,  # amenity_4
    9.5,   # amenity_6
]

const EMPIRICAL_RENTS = [
    138.0,
    300.0,
    130.0,
    240.0,
    120.0,
    235.0,
]

const LOG_INCOME_MU = 12.584991083577634
const LOG_INCOME_SIGMA = 0.5457737566188


# ============================================================
# MODEL RUN
# ============================================================

function run_model(amenities, seed)
    income_distribution(rng) = round(rand(rng, LogNormal(LOG_INCOME_MU, LOG_INCOME_SIGMA)) / 1_000)

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
        amenity = amenities,
        rent = [120, 120, 120, 120, 120, 120],
        units = [142, 150, 363, 117, 170, 20],
    )

    model = SegregationModel(
        income_distribution,
        0.4026,    # utility_alpha
        0.1433,   # utility_beta
        0.0980,  # price_change
        1,    # affordability_rate
        799,    # number_of_households
        districts,
        seed,
    )

    run_for!(model, STEPS)

    last_steps = filter(
        :step => step -> step > STEPS - AVERAGING_WINDOW,
        model.district_data,
    )

    rent_means = combine( 
        groupby(last_steps, :id),
        :rent => mean => :rent_mean,
    )

    sort!(rent_means, :id)

    return rent_means.rent_mean
end


function simulated_rents(amenities)
    rents = zeros(6)

    for replication in 1:REPLICATIONS
        rents .+= run_model(
            amenities,
            BASE_SEED + replication - 1,
        )
    end

    return rents ./ REPLICATIONS
end


# ============================================================
# SOBOL BLOCK
# ============================================================

function run_sobol_block()
    # LAST_SAMPLE is the number of Sobol points generated.
    # The loop below uses only FIRST_SAMPLE:LAST_SAMPLE.
    samples = QuasiMonteCarlo.sample(
        LAST_SAMPLE,
        LOWER_BOUNDS,
        UPPER_BOUNDS,
        SobolSample(),
    )

    for sample_id in FIRST_SAMPLE:LAST_SAMPLE
        θ = samples[:, sample_id]

        amenities = [
            θ[1],
            θ[2],
            θ[3],
            θ[4],
            FIXED_AMENITY,
            θ[5],
        ]

        model_rents = simulated_rents(amenities)

        objective_per_diff = sum(((model_rents .- EMPIRICAL_RENTS) ./ EMPIRICAL_RENTS) .^ 2)
        objective_log_diff = sum((log.(model_rents) .- log.(EMPIRICAL_RENTS)) .^ 2)
        objective_abs_diff = sum((model_rents .- EMPIRICAL_RENTS) .^ 2)

        result = (
            sample_id = sample_id,
            amenity_1 = amenities[1],
            amenity_2 = amenities[2],
            amenity_3 = amenities[3],
            amenity_4 = amenities[4],
            amenity_5 = amenities[5],
            amenity_6 = amenities[6],
            model_rent_1 = model_rents[1],
            model_rent_2 = model_rents[2],
            model_rent_3 = model_rents[3],
            model_rent_4 = model_rents[4],
            model_rent_5 = model_rents[5],
            model_rent_6 = model_rents[6],
            objective_per_diff = objective_per_diff,
            objective_log_diff = objective_log_diff,
            objective_abs_diff = objective_abs_diff,
        )

        CSV.write(
            OUTPUT_PATH,
            DataFrame([result]);
            append = isfile(OUTPUT_PATH),
        )

    end
end


run_sobol_block()
