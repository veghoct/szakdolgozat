include("../util/imports.jl")

# Source: https://www.ksh.hu/stadat_files/jov/hu/jov0005.html
annual_income_per_person = [564_041, 881_304, 1_057_318, 1_305_526, 1_459_115, 1_666_901, 1_899_253, 2_150_731, 2_562_414, 4_120_089]

# Source: https://www.ksh.hu/stadat_files/jov/hu/jov0022.html
household_size = [3.0, 3.0, 2.9, 2.6, 2.4, 2.4, 2.1, 2.1, 2.0, 1.7]

national_budapest_average_income_ratio = [
    621_898, 611_703, 605_729, 618_512, 667_243, 680_870, 679_378, 696_230, 756_590, 770_183, 766_220, 782_459, 826_282, 840_776, 835_457, 851_211
] ./ [
    493_297, 490_868, 489_577, 501_127, 545_626, 556_825, 557_138, 571_477, 623_005, 634_741, 633_428, 646_635, 680_448, 692_829, 691_228, 705_014
]

# Since the ABM is presuming that a residential agent is a household, we want to work with per household data
# not per person data   
monthly_income_per_household = annual_income_per_person .* 2.3 ./ 12 * mean(national_budapest_average_income_ratio)
monthly_income_per_household = round.(Int, monthly_income_per_household)

println(monthly_income_per_household)

X = hcat(monthly_income_per_household)

deciles = DataFrame(
    decile = ["Decile $i" for i in 1:10],
    monthly_income = monthly_income_per_household
)

# Standardize variables: mean 0, standard deviation 1
Z = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)

# Clustering.jl expects observations in columns
Z_t = permutedims(Z)

# -----------------------------
# 3A. K-means clustering
# -----------------------------

k = 3

km = kmeans(Z_t, k; maxiter = 1000, display = :none)
deciles.kmeans_cluster = assignments(km)

println("K-means clusters:")
sort(deciles[:, [:decile, :kmeans_cluster]], :kmeans_cluster) |> println

println(deciles)
println()

