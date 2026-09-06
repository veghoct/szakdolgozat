# Income segregation indices for model_data.
#
# Implements the measures from Szakszeminarium_1.pdf section 2.4:
# the Massey & Denton (1988) / Reardon & Firebaugh (2002) generalized
# dissimilarity index, and the corresponding normalized exposure/isolation
# index, computed citywide across the model's zones. Both are written once
# for an arbitrary number of income groups (the two-group formulas are a
# special case of the multi-group ones), and reused for the two-group
# (10, 25, 50, 75, 90) and three-group (tercile) columns in model_data.

function gini(incomes::AbstractVector{<:Real})
    n = length(incomes)
    n == 0 && return NaN
    n == 1 && return 0.0

    sorted_incomes = sort(Float64.(incomes))
    total = sum(sorted_incomes)
    total == 0 && return NaN

    weighted_sum = sum(i * income for (i, income) in enumerate(sorted_incomes))

    return 2 * weighted_sum / (n * total) - (n + 1) / n
end

# Assigns each resident to a group based on where their income falls
# relative to a set of (citywide) percentile cutoffs.
percentile_cutoffs(incomes, ps) = [quantile(incomes, p) for p in ps]

function group_index(income::Real, cutoffs::Vector{Float64})
    for (g, cutoff) in enumerate(cutoffs)
        income <= cutoff && return g
    end
    return length(cutoffs) + 1
end

# counts[z][g] = number of group-g residents living in zone z, across the
# whole city.
function zone_group_counts(model::SegregationModel, cutoffs::Vector{Float64})
    ngroups = length(cutoffs) + 1
    zone_ids = unique(d.zone_id for d in model.districts)
    index_of = Dict(z => i for (i, z) in enumerate(zone_ids))
    counts = [zeros(Int, ngroups) for _ in zone_ids]

    for r in model.residents
        counts[index_of[r.home.zone_id]][group_index(r.income, cutoffs)] += 1
    end

    return counts
end

function dissimilarity_index(counts::Vector{Vector{Int}}, ngroups::Int)
    total = sum(sum, counts)
    total == 0 && return NaN

    shares = [sum(c[g] for c in counts) for g in 1:ngroups] ./ total
    evenness = sum(p * (1 - p) for p in shares)
    evenness == 0 && return NaN

    numerator = sum(
        sum(c) * sum(abs(c[g] / sum(c) - shares[g]) for g in 1:ngroups)
        for c in counts if sum(c) > 0
    )

    return numerator / (2 * total * evenness)
end

function exposure_index(counts::Vector{Vector{Int}}, ngroups::Int)
    total = sum(sum, counts)
    total == 0 && return NaN

    shares = [sum(c[g] for c in counts) for g in 1:ngroups] ./ total

    return sum(
        sum(
            (sum(c) / total) * (c[g] / sum(c) - shares[g])^2
            for c in counts if sum(c) > 0
        ) / (1 - shares[g])
        for g in 1:ngroups if shares[g] < 1
    )
end

function citywide_segregation_indices(model::SegregationModel, cutoffs::Vector{Float64})
    ngroups = length(cutoffs) + 1
    counts = zone_group_counts(model, cutoffs)

    return dissimilarity_index(counts, ngroups), exposure_index(counts, ngroups)
end

function model_segregation_row(
        model::SegregationModel,
        lower_tail_cutoffs::Dict{Int, Vector{Float64}},
        cutoffs_three_groups::Vector{Float64},
    )
    # For each p, compares the lower p% of (citywide) income against the
    # remaining (100 - p)%. Not symmetric in p: the lower 10% vs rest and
    # the lower 90% vs rest are different bipartitions (poorest decile vs.
    # richest decile are different groups), so p = 10, 25, 75, 90 are all
    # kept as distinct columns; only p = 50 is its own mirror image.
    d_10, e_10 = citywide_segregation_indices(model, lower_tail_cutoffs[10])
    d_25, e_25 = citywide_segregation_indices(model, lower_tail_cutoffs[25])
    d_50, e_50 = citywide_segregation_indices(model, lower_tail_cutoffs[50])
    d_75, e_75 = citywide_segregation_indices(model, lower_tail_cutoffs[75])
    d_90, e_90 = citywide_segregation_indices(model, lower_tail_cutoffs[90])
    d_three, e_three = citywide_segregation_indices(model, cutoffs_three_groups)

    return (
        step = model.step_count,
        dissimilarity_10 = d_10,
        dissimilarity_25 = d_25,
        dissimilarity_50 = d_50,
        dissimilarity_75 = d_75,
        dissimilarity_90 = d_90,
        dissimilarity_three_groups = d_three,
        exposure_10 = e_10,
        exposure_25 = e_25,
        exposure_50 = e_50,
        exposure_75 = e_75,
        exposure_90 = e_90,
        exposure_three_groups = e_three,
    )
end

function zone_income_row(zone_id::Int, residents::Vector{ResidentAgent}, step::Int)
    incomes = [r.income for r in residents]

    return (
        step = step,
        id = zone_id,
        average_income = mean_or_nan(incomes),
        median_income = median_or_nan(incomes),
        gini = gini(incomes),
    )
end
