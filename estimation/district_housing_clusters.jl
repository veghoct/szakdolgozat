include("../util/imports.jl")

df = CSV.read(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), DataFrame, header = true)

zone_cluster = !isempty(ARGS) ? parse(Int, ARGS[1]) : 1

filtered_df = filter(:kmeans_cluster => ==(zone_cluster), df)

# Turn district_moments df into matrix
X = Matrix(filtered_df[:, [:price_per_area]])

# Standardize variables: mean 0, standard deviation 1
Z = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)

# Clustering.jl expects observations in columns
Z_t = permutedims(Z)

# -----------------------------
# 3A. K-means clustering
# -----------------------------

k = 2

km = kmeans(Z_t, k; maxiter = 1000, display = :none)
filtered_df.house_type_cluster = assignments(km)

println("\nMean and median rent by clusters:")

c1_rent = filter(:house_type_cluster => ==(1), filtered_df).price_in_2020_index
println("\nMean rent for cluster 1: ", mean(c1_rent))
println("Median rent for cluster 1: ", median(c1_rent))
println("Size of cluster: ", length(c1_rent))

c2_rent = filter(:house_type_cluster => ==(2), filtered_df).price_in_2020_index
println("\nMean rent for cluster 2: ", mean(c2_rent))
println("Median rent for cluster 2: ", median(c2_rent))
println("Size of cluster: ", length(c2_rent))

c3_rent = filter(:house_type_cluster => ==(3), filtered_df).price_in_2020_index
println("\nMean rent for cluster 3: ", mean(c3_rent))
println("Median rent for cluster 3: ", median(c3_rent))
println("Size of cluster: ", length(c3_rent))


# -----------------------------
# 5. Optional: compare k = 2, 3, 4, 5
# -----------------------------

ks = 2:5
wcss = Float64[]

for kk in ks
    result = kmeans(Z_t, kk; maxiter = 1000, display = :none)
    push!(wcss, result.totalcost)
end

p_elbow = plot(
    ks,
    wcss,
    marker = :circle,
    xlabel = "Number of clusters k",
    ylabel = "Within-cluster sum of squares",
    title = "Elbow plot for k-means",
    legend = false
)

savefig(p_elbow, ".figures/zone_clustering_$zone_cluster.svg")

#df = leftjoin(df, filtered_df[:, [:city, :house_type_cluster]], on = "city")

#CSV.write(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), df, header = true)

#println(first(df, 1))
#println()
