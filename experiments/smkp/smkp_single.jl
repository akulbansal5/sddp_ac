"""
Simplified SMKP solver for a single instance
All parameters are specified within the file
"""

using SDDP
using Gurobi
using Serialization
using CSV, DataFrames

# ============================================================================
# PARAMETERS - Modify these to change the instance and solver settings
# ============================================================================

# Instance parameters (ID 254: T=3, rows=10, cols=30, scens=3)
folder = joinpath(@__DIR__, "smkp_data")
id     = 254
inst   = 1
st     = 3        # number of stages (T)
rows   = 10     # number of constraint rows
cols   = 30     # number of decision variables
scens  = 3     # number of scenarios per node

# Solver parameters
threads    = 1
mipgap     = 1e-2
time_limit = 3600   # time limit in seconds (1 hour - liberal backup limit)
iter_limit = 5   # iteration limit (fixed number of iterations to run)
lb         = 0              # lower bound

# Algorithm parameters
duality_handler  = SDDP.LagrangianDuality()           # or SDDP.LaporteLouveauxDuality()
forward_pass     = SDDP.DefaultMultiForwardPass()     # or SDDP.DefaultNestedForwardPass()
backward_pass    = SDDP.AnguloMultiBackwardPass()     # or SDDP.DefaultMultiBackwardPass()
sampling_scheme  = SDDP.InSampleMonteCarloMultiple()  # or SDDP.AllSampleMonteCarloMultiple()

# Algorithm type: 1 = SDDiP, 2 = Nested Benders
fpass_type = 1


# ---------------------------------------------------------------------------
# Stopping Criteria and Sampling Parameters
# ---------------------------------------------------------------------------

delta        = 0.05       # Convergence tolerance for algorithm stopping
type1_prob   = 1.28       # Type I error quantile (e.g., z-value for 10% one-sided)
type2_prob   = 1.28       # Type II error quantile
M            = 2          # Number of scenario paths sampled per iteration
iter_pass    = 1          # Iteration pass type (e.g., 1 = standard, other values = special handling)
final_run    = false      # For SDDiP: true = traverse entire scenario tree for deterministic bounds
seed         = nothing    # Random seed for reproducibility (nothing = do not set seed)

# Simulation parameters (for out-of-sample evaluation)
postSim = 50       # number of simulation replications
simTime = 3600.0   # simulation time limit

# ============================================================================
# FUNCTIONS
# ============================================================================

function getSMKPdata(folder, file)
    """
    Deserializes a single SMKP data file
    """
    out = open(joinpath(folder, file), "r") do f
        deserialize(f)
    end
    return out
end

function getAllSMKPdata(folder, id, inst, st, rows, cols, scens)
    """
    Loads all SMKP data files for a given instance
    Returns: AList, TList, c, d, qDict
    """
    # Load A matrices (one per stage)
    AList = []
    for t in 1:st
        file = "smkp_A"*"_$(id)_$(inst)_$(t)_$(rows)_$(cols)_$(scens).jls"
        A_t = getSMKPdata(folder, file)
        push!(AList, A_t)
    end

    # Load T matrices (one per stage)
    TList = []
    for t in 1:st
        file = "smkp_T"*"_$(id)_$(inst)_$(t)_$(rows)_$(cols)_$(scens).jls"
        T_t = getSMKPdata(folder, file)
        push!(TList, T_t)
    end

    # Load c vector (first stage cost)
    file = "smkp_c"*"_$(id)_$(inst)_1_1_$(cols).jls"
    c = getSMKPdata(folder, file)

    # Load d vector (first stage cost)
    file = "smkp_d"*"_$(id)_$(inst)_1_1_$(cols).jls"
    d = getSMKPdata(folder, file)

    # Load q vectors (scenario costs for stages 2 and beyond)
    qDict = Dict()
    for t in 2:st
        qDict[t] = []
        for s in 1:scens
            file = "smkp_q"*"_$(id)_$(inst)_$(t)_$(s)_$(cols).jls"
            q_t_s = getSMKPdata(folder, file)
            push!(qDict[t], q_t_s)
        end
    end
    
    return AList, TList, c, d, qDict
end

function smkp_gen_ver4(st, lb, rows, cols, A, T, c, d, q, threads)
    """
    Generates the SDDP model for SMKP problem
    """
    model = SDDP.LinearPolicyGraph(
        stages = st,
        sense = :Min,
        lower_bound = lb,
        optimizer = Gurobi.Optimizer,
        solver_threads = threads,
    ) do sp, stage
        @variable(sp, 0 <= x[1:cols], Bin, SDDP.State, initial_value = 0)

        if stage == 1
            @variable(sp, 0 <= z[1:cols], Bin)
        else
            @variable(sp, 0 <= p[1:rows], Int)
        end

        cost_scale = 1e-1
        one_vector = ones(Int, cols)
        h = (3/4)*A[stage]*one_vector + (3/4)*T[stage]*one_vector
        penalty = 200.0

        if stage == 1
            @constraint(
                sp,
                knapsack[i in 1:rows],
                sum((A[stage][i, j]*x[j].out + T[stage][i, j]*z[j]) for j in 1:cols) >= h[i]
            )
        else
            @constraint(
                sp,
                knapsack[i in 1:rows],
                sum((A[stage][i, j]*x[j].out + T[stage][i, j]*x[j].in) for j in 1:cols) + p[i] >= h[i]
            )
        end

        if stage == 1
            @stageobjective(
                sp,
                sum(cost_scale*c[j]*x[j].out for j in 1:cols) + sum(cost_scale*d[j]*z[j] for j in 1:cols)
            )
        else
            SDDP.parameterize(sp, q[stage]) do qs
                @stageobjective(
                    sp,
                    sum(cost_scale*penalty*p[i] for i in 1:rows) + sum(cost_scale*qs[j]*x[j].out for j in 1:cols)
                )
            end
        end
    end

    return model
end

function write_results_to_csv(
    folder::String,
    id::Int,
    rows::Int,
    cols::Int,
    scens::Int,
    stages::Int,
    total_time::Float64,
    backward_pass_time::Float64,
    iterations::Int,
    lower_bound::Float64,
    ci_low::Float64,
    ci_high::Float64,
    std_cuts::Int,
    nonstd_cuts::Int,
    duality_handler::String,
    forward_pass::String,
    backward_pass::String,
    sampling_scheme::String,
    fpass_type::Int,
    mipgap::Float64,
    M::Int,
    gap::Float64
)
    """
    Write results to CSV file
    Appends to existing file or creates new one
    """
    csv_file = joinpath(folder, "smkp_results.csv")
    
    # Create DataFrame with results
    # Column order: instance attributes, algorithm parameters, then results
    results_df = DataFrame(
        id = [id],
        rows = [rows],
        cols = [cols],
        scens = [scens],
        stages = [stages],
        duality_handler = [duality_handler],
        forward_pass = [forward_pass],
        backward_pass = [backward_pass],
        sampling_scheme = [sampling_scheme],
        fpass_type = [fpass_type],
        mipgap = [mipgap],
        M = [M],
        total_time = [total_time],
        backward_pass_time = [backward_pass_time],
        iterations = [iterations],
        lower_bound = [lower_bound],
        ci_low = [ci_low],
        ci_high = [ci_high],
        gap = [gap],
        std_cuts = [std_cuts],
        nonstd_cuts = [nonstd_cuts]
    )
    
    # Append to CSV file (creates file if it doesn't exist)
    if isfile(csv_file)
        CSV.write(csv_file, results_df, append=true)
    else
        CSV.write(csv_file, results_df)
    end
    
    println("Results written to: $csv_file")
    return csv_file
end

function train_method(
    model,
    duality_handler,
    forward_pass,
    backward_pass,
    sampling_scheme,
    time_limit,
    iter_limit,
    mipgap,
    iter_pass,
    M = 1,
    delta = 0.05,
    fpass_type = 1,
    final_run = false,
    type1_prob = 1.28,
    type2_prob = 1.28,
    seed = nothing
)
    """
    Trains the SDDP model
    fpass_type = 1: SDDiP algorithm
    fpass_type = 2: Nested Benders algorithm
    """
    outputs = nothing

    if fpass_type == 2
        # Nested Benders algorithm
        if iter_limit < 1
            outputs = SDDP.train(
                model;
                duality_handler = duality_handler,
                forward_pass = forward_pass,
                backward_pass = backward_pass,
                sampling_scheme = sampling_scheme,
                stopping_rules = [SDDP.TimeLimit(time_limit), SDDP.NBBoundStalling(delta)],
                mipgap = mipgap,
                iter_pass = iter_pass,
                M = M,
                print_level = 2,
                final_run = final_run
            )
        else
            outputs = SDDP.train(
                model,
                duality_handler = duality_handler,
                forward_pass = forward_pass,
                backward_pass = backward_pass,
                sampling_scheme = sampling_scheme,
                stopping_rules = [SDDP.IterationLimit(iter_limit), SDDP.NBBoundStalling(delta), SDDP.TimeLimit(time_limit)],
                mipgap = mipgap,
                iter_pass = iter_pass,
                M = M,
                print_level = 2,
                final_run = final_run
            )
        end
    else
        # SDDiP algorithm
        if iter_limit < 1
            outputs = SDDP.train(
                model;
                duality_handler = duality_handler,
                forward_pass = forward_pass,
                backward_pass = backward_pass,
                sampling_scheme = sampling_scheme,
                stopping_rules = [SDDP.TimeLimit(time_limit), SDDP.TitoStalling(type1_prob, type2_prob, delta)],
                mipgap = mipgap,
                iter_pass = iter_pass,
                M = M,
                print_level = 2,
                final_run = final_run,
                seed = seed
            )
        else
            outputs = SDDP.train(
                model,
                duality_handler = duality_handler,
                forward_pass = forward_pass,
                backward_pass = backward_pass,
                sampling_scheme = sampling_scheme,
                stopping_rules = [SDDP.IterationLimit(iter_limit), SDDP.TitoStalling(type1_prob, type2_prob, delta), SDDP.TimeLimit(time_limit)],
                mipgap = mipgap,
                iter_pass = iter_pass,
                M = M,
                print_level = 2,
                final_run = final_run,
                seed = seed
            )
        end
    end

    sddp_bound = outputs[1].bb
    sddp_simval = outputs[1].ub

    println("Bound attained from SDDP:            $(sddp_bound)")
    println("Simulation value attained from SDDP: $(sddp_simval)")
    println("================== Training Complete ====================")

    return outputs
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

println("=" ^ 60)
println("SMKP Solver - Single Instance")
println("=" ^ 60)
println("Instance ID: $id")
println("Parameters: T=$st, rows=$rows, cols=$cols, scens=$scens")
println("Algorithm: $(fpass_type == 1 ? "SDDiP" : "Nested Benders")")
println("=" ^ 60)

# Load data
println("\nLoading data files...")
build_time = @elapsed begin
    AList, TList, c, d, qDict = getAllSMKPdata(folder, id, inst, st, rows, cols, scens)
end
println("Data loaded in $(build_time) seconds")

# Build model
println("\nBuilding SDDP model...")
build_time = @elapsed begin
    model = smkp_gen_ver4(st, lb, rows, cols, AList, TList, c, d, qDict, threads)
end
println("Model built in $(build_time) seconds")

# Train model
println("\nTraining SDDP model...")
train_time = @elapsed begin
    outputs = train_method(
        model,
        duality_handler,
        forward_pass,
        backward_pass,
        sampling_scheme,
        time_limit,
        iter_limit,
        mipgap,
        iter_pass,
        M,
        delta,
        fpass_type,
        final_run,
        type1_prob,
        type2_prob,
        seed
    )
end
println("Training completed in $(train_time) seconds")

# Run simulation (optional)
println("\nRunning out-of-sample simulation...")
simulation_time = @elapsed begin
    simulations = SDDP.simulate(
        model,
        postSim,
        simTime,
        set_sim_seed = true,
        sim_seed = seed === nothing ? 42 : 2*seed
    )

    objectives = map(simulations) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    μ, ci = SDDP.confidence_interval(objectives)
    sim_lower = μ - ci
    sim_upper = μ + ci
end

println("Simulation completed in $(simulation_time) seconds")
println("\nSimulation Results:")
println("  Mean objective: $(μ)")
println("  Confidence interval: [$sim_lower, $sim_upper]")

# Calculate gap: 100 * (ci_high - lower_bound) / ci_high
gap = if sim_upper != 0.0
    100.0 * (sim_upper - outputs[1].bb) / sim_upper
else
    NaN
end
println("  Gap: $(gap)%")

# Write results to CSV
println("\nWriting results to CSV...")
write_results_to_csv(
    folder,
    id,
    rows,
    cols,
    scens,
    st,  # stages
    outputs[1].time,  # total_time from training
    outputs[1].backward_pass_time,  # backward_pass_time from training outputs
    outputs[1].iter,  # iterations
    outputs[1].bb,    # lower_bound
    sim_lower,        # ci_low
    sim_upper,        # ci_high
    outputs[1].cs,    # std_cuts
    outputs[1].cns,   # nonstd_cuts
    string(nameof(typeof(duality_handler))),   # duality_handler
    string(nameof(typeof(forward_pass))),      # forward_pass
    string(nameof(typeof(backward_pass))),     # backward_pass
    string(nameof(typeof(sampling_scheme))),   # sampling_scheme
    fpass_type,       # fpass_type (1 = SDDiP, 2 = Nested Benders)
    mipgap,           # mipgap
    M,                # M (number of paths sampled)
    gap               # gap: 100 * (ci_high - lower_bound) / ci_high
)

println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Build time:      $(build_time) seconds")
println("Training time:   $(train_time) seconds")
println("Simulation time: $(simulation_time) seconds")
println("Total time:      $(build_time + train_time + simulation_time) seconds")
println("=" ^ 60)