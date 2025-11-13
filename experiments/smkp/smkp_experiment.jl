using SDDP
using Gurobi
using Serialization
using CSV, DataFrames

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

function write_results_to_csv(folder::String, filename::String, id::Int, rows::Int, cols::Int, scens::Int, stages::Int, total_time::Float64, backward_pass_time::Float64, iterations::Int, lower_bound::Float64, ci_low::Float64, ci_high::Float64, std_cuts::Int, nonstd_cuts::Int, duality_handler::String, forward_pass::String, backward_pass::String, sampling_scheme::String, fpass_type::Int, mipgap::Float64, M::Int, gap::Float64, delta::Float64, type1_prob::Float64, type2_prob::Float64, seed::Int, lagrangian_dominant::Int = 0, integer_lshaped_dominant::Int = 0, incomparable::Int = 0)
    mkpath(folder)
    csv_file = joinpath(folder, filename)
    results_df = DataFrame(id = [id], rows = [rows], cols = [cols], scens = [scens], stages = [stages], duality_handler = [duality_handler], forward_pass = [forward_pass], backward_pass = [backward_pass], sampling_scheme = [sampling_scheme], fpass_type = [fpass_type], mipgap = [mipgap], M = [M], total_time = [total_time], backward_pass_time = [backward_pass_time], iterations = [iterations], lower_bound = [lower_bound], ci_low = [ci_low], ci_high = [ci_high], gap = [gap], std_cuts = [std_cuts], nonstd_cuts = [nonstd_cuts], delta = [delta], type1_prob = [type1_prob], type2_prob = [type2_prob], seed = [seed], lagrangian_dominant = [lagrangian_dominant], integer_lshaped_dominant = [integer_lshaped_dominant], incomparable = [incomparable])
    if isfile(csv_file)
        CSV.write(csv_file, results_df, append=true)
    else
        CSV.write(csv_file, results_df)
    end
    return csv_file
end

function train_method(model, duality_handler, forward_pass, backward_pass, sampling_scheme, time_limit, iter_limit, mipgap, iter_pass, M = 1, delta = 0.05, fpass_type = 1, final_run = false, type1_prob = 1.28, type2_prob = 1.28, seed = nothing)
    outputs = nothing
    if fpass_type == 2
        if iter_limit < 1
            outputs = SDDP.train(model; duality_handler = duality_handler, forward_pass = forward_pass, backward_pass = backward_pass, sampling_scheme = sampling_scheme, stopping_rules = [SDDP.TimeLimit(time_limit), SDDP.NBBoundStalling(delta)], mipgap = mipgap, iter_pass = iter_pass, M = M, print_level = 2, final_run = final_run)
        else
            outputs = SDDP.train(model, duality_handler = duality_handler, forward_pass = forward_pass, backward_pass = backward_pass, sampling_scheme = sampling_scheme, stopping_rules = [SDDP.IterationLimit(iter_limit), SDDP.NBBoundStalling(delta), SDDP.TimeLimit(time_limit)], mipgap = mipgap, iter_pass = iter_pass, M = M, print_level = 2, final_run = final_run)
        end
    else
        if iter_limit < 1
            outputs = SDDP.train(model; duality_handler = duality_handler, forward_pass = forward_pass, backward_pass = backward_pass, sampling_scheme = sampling_scheme, stopping_rules = [SDDP.TimeLimit(time_limit), SDDP.TitoStalling(type1_prob, type2_prob, delta)], mipgap = mipgap, iter_pass = iter_pass, M = M, print_level = 2, final_run = final_run, seed = seed)
        else
            outputs = SDDP.train(model, duality_handler = duality_handler, forward_pass = forward_pass, backward_pass = backward_pass, sampling_scheme = sampling_scheme, stopping_rules = [SDDP.IterationLimit(iter_limit), SDDP.TitoStalling(type1_prob, type2_prob, delta), SDDP.TimeLimit(time_limit)], mipgap = mipgap, iter_pass = iter_pass, M = M, print_level = 2, final_run = final_run, seed = seed)
        end
    end
    return outputs
end

# Parse command-line arguments
id         = parse(Int, ARGS[1])   # Instance ID
method_idx = parse(Int, ARGS[2])   # Index for method selection
seed_arg   = parse(Int, ARGS[3])   # Seed for reproducibility

# Instance parameters: Map ID to (st, rows, cols, scens)
instance_params = Dict(
    256 => (3, 10, 30, 3),
    257 => (3, 10, 30, 3),
    258 => (3, 10, 30, 3),
    316 => (3, 10, 30, 5),
    317 => (3, 10, 30, 5),
    318 => (3, 10, 30, 5),
    346 => (4, 10, 30, 3),
    347 => (4, 10, 30, 3),
    348 => (4, 10, 30, 3),
    234 => (4, 10, 30, 10),
    240 => (4, 10, 30, 10),
    246 => (4, 10, 30, 10),
    456 => (5, 15, 40, 3),
    457 => (5, 15, 40, 3),
    458 => (5, 15, 40, 3),
    236 => (5, 15, 40, 10),
    242 => (5, 15, 40, 10),
    248 => (5, 15, 40, 10)
)

methods = [
    (SDDP.LagrangianDuality(), SDDP.DefaultMultiBackwardPass(), "Lagrangian", "Default"),
    (SDDP.LagrangianDuality(), SDDP.AnguloMultiBackwardPass(), "Lagrangian", "Angulo"),
    (SDDP.LaporteLouveauxDuality(), SDDP.DefaultMultiBackwardPass(), "Laporte", "Default"),
    (SDDP.LaporteLouveauxDuality(), SDDP.AnguloMultiBackwardPass(), "Laporte", "Angulo"),
    (SDDP.ContinuousConicDuality(), SDDP.DefaultMultiBackwardPass(), "Benders", "Default"),
    (SDDP.LagrangianDuality(), SDDP.ComparisonMultiBackwardPass(), "Lagrangian", "Comparison")
]

st, rows, cols, scens = instance_params[id]
duality_handler, backward_pass, duality_name, backward_name = methods[method_idx + 1]

folder = joinpath(@__DIR__, "smkp_data", "smkp_subset")
inst = 1
threads = 2
mipgap = 1e-3
time_limit = 3600
iter_limit = 10000
lb = 0
delta = 0.01
type1_prob = 1.28
type2_prob = 1.28
M = 2
iter_pass = 1
final_run = false
fpass_type = 1
# Calculate total number of scenario paths: scens^(st-1) since stage 1 is deterministic
total_paths = scens^(st - 1)
# Use max(25, 5% of all paths) for simulation
postSim = max(25, ceil(Int, 0.05 * total_paths))
simTime = 3600.0
forward_pass = SDDP.DefaultMultiForwardPass()
sampling_scheme = SDDP.InSampleMonteCarloMultiple()

fixed_offset = 1
seed = seed_arg * 1000 + fixed_offset

AList, TList, c, d, qDict = getAllSMKPdata(folder, id, inst, st, rows, cols, scens)
model = smkp_gen_ver4(st, lb, rows, cols, AList, TList, c, d, qDict, threads)
outputs = train_method(model, duality_handler, forward_pass, backward_pass, sampling_scheme, time_limit, iter_limit, mipgap, iter_pass, M, delta, fpass_type, final_run, type1_prob, type2_prob, seed)
simulations = SDDP.simulate(model, postSim, simTime, set_sim_seed = true, sim_seed = 2*seed)
objectives = map(simulations) do simulation
    return sum(stage[:stage_objective] for stage in simulation)
end
μ, ci = SDDP.confidence_interval(objectives)
sim_lower = μ - ci
sim_upper = μ + ci
gap = sim_upper != 0.0 ? 100.0 * (sim_upper - outputs[1].bb) / sim_upper : NaN
results_folder = joinpath(@__DIR__, "results")
csv_filename = "smkp_results__$(id)_$(method_idx)_$(seed_arg).csv"
# Extract cut comparison statistics if available
lagrangian_dominant = hasfield(typeof(outputs[1]), :lagrangian_dominant) ? outputs[1].lagrangian_dominant : 0
integer_lshaped_dominant = hasfield(typeof(outputs[1]), :integer_lshaped_dominant) ? outputs[1].integer_lshaped_dominant : 0
incomparable = hasfield(typeof(outputs[1]), :incomparable) ? outputs[1].incomparable : 0
write_results_to_csv(results_folder, csv_filename, id, rows, cols, scens, st, outputs[1].time, outputs[1].backward_pass_time, outputs[1].iter, outputs[1].bb, sim_lower, sim_upper, outputs[1].cs, outputs[1].cns, duality_name, string(nameof(typeof(forward_pass))), backward_name, string(nameof(typeof(sampling_scheme))), fpass_type, mipgap, M, gap, delta, type1_prob, type2_prob, seed, lagrangian_dominant, integer_lshaped_dominant, incomparable)

