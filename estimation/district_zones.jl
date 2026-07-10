include("../util/imports.jl")

df = CSV.read(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), DataFrame, header = true)

district_moments = combine(groupby(df, :city),
    :price_in_2020_index => (x -> quantile(x, 0.5)) => :p50,
    :price_per_area => (x -> quantile(x, 0.5)) => :p50_m2,
    :price_in_2020_index => (x -> quantile(x, 0.9)) => :p90,
    :price_per_area => (x -> quantile(x, 0.9)) => :p90_m2
)

# Turn district_moments df into matrix
X = Matrix(district_moments[:, [:p50, :p50_m2, :p90, :p90_m2]])

# Standardize variables: mean 0, standard deviation 1
Z = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)

# Clustering.jl expects observations in columns
Z_t = permutedims(Z)

# -----------------------------
# 3A. K-means clustering
# -----------------------------

k = 3

km = kmeans(Z_t, k; maxiter = 1000, display = :none)
district_moments.kmeans_cluster = assignments(km)

println("K-means clusters:")
sort(district_moments[:, [:city, :kmeans_cluster]], :kmeans_cluster) |> println

# -----------------------------
# 3B. Hierarchical clustering
# -----------------------------

# Pairwise Euclidean distance between districts
dist_matrix = pairwise(Euclidean(), Z_t, dims = 2)

# Hierarchical clustering using Ward linkage
hc = hclust(dist_matrix, linkage = :ward)

district_moments.hclust_cluster = cutree(hc, k = k)

println("\nHierarchical clusters:")
sort(district_moments[:, [:city, :hclust_cluster]], :hclust_cluster) |> println

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

#savefig(p_elbow, ".figures/budapest_elbow.svg")

df = leftjoin(df, district_moments[:, [:city, :kmeans_cluster]], on = "city")
    
CSV.write(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), df, header = true)

println(first(df, 1))
println()
