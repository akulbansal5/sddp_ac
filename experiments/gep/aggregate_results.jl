using CSV, DataFrames, Dates

results_dir = joinpath(@__DIR__, "results")
files = filter(f -> startswith(f, "gep_results__") && endswith(f, "_5.csv"), readdir(results_dir))
files = [joinpath(results_dir, f) for f in files]

all_rows = DataFrame[]
timestamp = now()

println("Processing $(length(files)) files:")
for file in files
    println("  Reading: $(basename(file))")
    df = CSV.read(file, DataFrame)
    if nrow(df) > 0
        push!(all_rows, df[end:end, :])
    end
end

if !isempty(all_rows)
    combined = vcat(all_rows...)
    combined.aggregation_timestamp = fill(timestamp, nrow(combined))
    timestamp_str = Dates.format(timestamp, "yyyy-mm-dd_HH-MM-SS")
    output_file = joinpath(results_dir, "gep_results_aggregated_cut_compare_$timestamp_str.csv")
    CSV.write(output_file, combined)
    println("Aggregated $(length(files)) files into $output_file at $timestamp")
end

