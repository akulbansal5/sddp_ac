using CSV, DataFrames, Dates, Statistics

# Function to extract instance ID from filename pattern: gep_results__{id}_{config}_{rep}.csv
function extract_instance_id(filename)
    m = match(r"gep_results__(\d+)_", filename)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

# Aggregate results files for given instance ID range
function aggregate_results_files(id_min::Int = 60, id_max::Int = 74, results_dir::String = joinpath(@__DIR__, "results"))
    # Filter files for instance IDs in range
    files = filter(f -> begin
        if !startswith(f, "gep_results__")
            return false
        end
        id = extract_instance_id(f)
        return id !== nothing && id_min <= id <= id_max
    end, readdir(results_dir))
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
        output_file = joinpath(results_dir, "gep_results_aggregated_$(id_min)_$(id_max)_$timestamp_str.csv")
        CSV.write(output_file, combined)
        println("Aggregated $(length(files)) files into $output_file at $timestamp")
        return output_file
    else
        println("No files found to aggregate")
        return nothing
    end
end

# Average aggregated results across instance IDs and seeds
function average_aggregated_results(input_file::String, results_dir::String = joinpath(@__DIR__, "results"))
    # Read the aggregated CSV file
    df = CSV.read(input_file, DataFrame)
    
    # Compute total_cuts
    df.total_cuts = df.std_cuts .+ df.nonstd_cuts
    
    # Compute eprop and iprop with division by zero handling
    df.eprop = zeros(Float64, nrow(df))
    df.iprop = zeros(Float64, nrow(df))
    
    for i in 1:nrow(df)
        if df.total_cuts[i] == 0
            # Division by zero - set to 0 and print warning
            df.eprop[i] = 0.0
            df.iprop[i] = 0.0
            println("id: $(df.id[i]), stages: $(df.stages[i]), scens: $(df.scens[i]), duality_handler: $(df.duality_handler[i]), backward_pass: $(df.backward_pass[i]), seed: $(df.seed[i])")
        else
            df.eprop[i] = df.nonstd_cuts[i] / df.total_cuts[i]
            df.iprop[i] = df.incomparable[i] / df.total_cuts[i]
        end
    end
    
    # Extract ID range for filename (before filtering out id column)
    id_min = minimum(df.id)
    id_max = maximum(df.id)
    
    # Filter to keep only relevant columns
    filtered_df = select(df, [:id, :stages, :scens, :duality_handler, :backward_pass, 
                              :total_time, :backward_pass_time, :iterations, :gap, 
                              :eprop, :seed, :iprop])
    
    # Group by instance class and method, then compute means
    grouped = groupby(filtered_df, [:stages, :scens, :duality_handler, :backward_pass])
    
    averaged = combine(grouped,
        :total_time => mean => :total_time,
        :backward_pass_time => mean => :backward_pass_time,
        :iterations => mean => :iterations,
        :gap => mean => :gap,
        :eprop => mean => :eprop,
        :iprop => mean => :iprop
    )
    
    # Save to new CSV file
    timestamp = now()
    timestamp_str = Dates.format(timestamp, "yyyy-mm-dd_HH-MM-SS")
    output_file = joinpath(results_dir, "gep_results_averaged_$(id_min)_$(id_max)_$timestamp_str.csv")
    CSV.write(output_file, averaged)
    println("Averaged results saved to $output_file")
    
    return output_file
end

# Call the averaging function for the aggregated CSV file
results_dir = joinpath(@__DIR__, "results")
input_file  = joinpath(results_dir, "gep_results_aggregated_60_74_2025-11-12_18-43-50.csv")
average_aggregated_results(input_file, results_dir)
