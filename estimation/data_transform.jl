include("../util/imports.jl")

raw_df = CSV.read(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado.csv"), DataFrame, header = true)

last_price_cols = [Symbol("latest_price_$y") for y in 2018:2023]
limits = Dict(c => quantile(collect(skipmissing(raw_df[!, c])), [0.005, 0.995]) for c in last_price_cols)

df = filter(raw_df) do row
    active_col = findlast(c -> !ismissing(row[c]), last_price_cols)

    limits[last_price_cols[active_col]][1] <= row[last_price_cols[active_col]] <= limits[last_price_cols[active_col]][2]
end

df_median = combine(df, [
    Symbol("latest_price_$y") => (x -> median(skipmissing(x))) 
    for y in 2018:2023
])

year = 2018:2023
median_values = Vector(df_median[1, :])
median_price_index = (Vector(df_median[1, :]) ./ df_median[1, 3]) * 100


final_df = DataFrame(
    Year = year,
    Median_Rent = median_values,
    Median_2020_Index = median_price_index,
)


active_idx = [findlast(c -> !ismissing(row[c]), last_price_cols) for row in eachrow(df)]
active_prices = [df[i, last_price_cols[active_idx[i]]] for i in 1:nrow(df)]
selling_years = [year[i] for i in active_idx]

new_df = DataFrame(
    ad_id_hash = df.ad_id_hash,
    selling_year = selling_years,
    city = df.city,
    area_size = df.area_size,
    price_latest_active = active_prices,
    price_in_2020_index = active_prices .* 100 ./ final_df.Median_2020_Index[selling_years .- 2017],
    property_type = df.property_type,
    property_subtype = df.property_subtype,
    zone_name = df.zone_name
)

new_df.price_per_area = new_df.price_in_2020_index ./ new_df.area_size

CSV.write(joinpath(@__DIR__, "..", ".localrw", "budapest_ingatlanok_kiado_transzformalt_2020_index.csv"), new_df, header = true)

println(final_df)
println(first(new_df, 2))
println()
