include("../util/imports.jl")

df = CSV.read(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), DataFrame, header = true)

# -----------------------------
# Calculate the beta paremeter, that is the importance of status in the utility function
# -----------------------------

model = lm(@formula(price_in_2020_index ~ city), df)

r2_val = r2(model)

pct_between = r2_val * 100
pct_within = (1.0 - r2_val) * 100

println("\nEstimating the beta parameter:")
println("Variance explained BETWEEN groups (Beta): ", round(pct_between, digits = 2), "%")
println("Variance explained WITHIN groups: ", round(pct_within, digits = 2), "%")

# -----------------------------
# Calculate the alpha paremeter, that is the importance of housing as a whole in the utility function
# -----------------------------

average_monthly_rent = combine(
    groupby(df, :selling_year),
    :price_in_2020_index => median => :median_price_in_2020_index,
    :price_latest_active => median => :median_price,
)

median_monthly_household_income = (341768 + 390438) / 2 / 1000

println("\nThe average monthly rent by years")
println(average_monthly_rent)

println("\nEstimating the alpha parameter:")
println(average_monthly_rent.median_price_in_2020_index ./ median_monthly_household_income)
println("Alpha: ", mean(average_monthly_rent.median_price_in_2020_index ./ median_monthly_household_income))

println("\nEstimating the average change in rent")
println(diff(average_monthly_rent.median_price) ./ average_monthly_rent.median_price[1 : end - 1] .* 100)
println("Change %: ", mean(diff(average_monthly_rent.median_price) ./ average_monthly_rent.median_price[1 : end - 1] .* 100))