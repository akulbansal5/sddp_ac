using SDDP
using Gurobi
using Serialization
using CSV, DataFrames

function gep_data_load(folder, fixCost_file, varCost_file)
    fix_cost = CSV.read(joinpath(folder, fixCost_file), DataFrame)
    var_cost = CSV.read(joinpath(folder, varCost_file), DataFrame)
    select!(fix_cost, Not(:Time))
    select!(var_cost, Not(:Time))
    fix_cost_matrix = Matrix(fix_cost)
    var_cost_matrix = Matrix(var_cost)
    genMax = [4, 10, 10, 1, 45, 4]
    genCap = [1130.0, 390.0, 380, 1180, 175, 560]
    genHeat = [8844, 7196, 10842, 10400, 0, 8613]
    genEff = [0.4, 0.56, 0.4, 0.45, 1, 0.48]
    omCost = [4.7, 2.11, 3.66, 0.51, 5, 2.98]
    hours = [271.0, 6556.0, 1933.0]
    return fix_cost_matrix, var_cost_matrix, genMax, genCap, genHeat, genEff, omCost, hours
end

function gep(T::Int64, G::Int64, S::Int64, lb::Number, binCoeff::Number, invCost::Matrix{Float64}, genCost::Matrix{Float64}, genMax::Vector{Int64}, genCap::Vector{Float64}, genHeat::Vector{Int64}, genEff::Vector{Float64}, omCost::Vector{Float64}, hours::Vector{Float64}, support::Dict{Int64, Vector{Vector{Any}}}, threads::Int64)
    model = SDDP.LinearPolicyGraph(stages = T, sense = :Min, lower_bound = lb, optimizer = Gurobi.Optimizer, solver_threads = threads) do sp, stage
        @variable(sp, 0 <= x[1:G], integer = true)
        rescale = 10.0
        @variable(sp, 0 <= y[1:G, 1:S])
        @variable(sp, 0 <= u[1:S])
        @variable(sp, 0 <= h[1:G, 1:binCoeff], Bin, SDDP.State, initial_value = 0)
        @variable(sp, demand[1:S])
        @constraint(sp, genCapacity[g in 1:G, s in 1:S], (genCap[g]/rescale)*sum(2^(m-1)*h[g, m].out for m in 1:binCoeff) >= y[g, s])
        @constraint(sp, genLimit[g in 1:G], sum(2^(m-1)*h[g, m].out for m in 1:binCoeff) <= genMax[g])
        @constraint(sp, demSatisfy[s in 1:S], sum(rescale*y[g, s] for g in 1:G) + u[s] == demand[s])
        @constraint(sp, flow[g in 1:G], sum(2^(m-1)*h[g, m].out for m in 1:binCoeff) == x[g] + sum(2^(m-1)*h[g, m].in for m in 1:binCoeff))
        SDDP.parameterize(sp, support[stage]) do sample
            dem_sample = sample[1:end-1]
            gas_price = sample[end]
            rate = 0.08
            if stage != 1
                genCost[stage, 2] = ((gas_price/1000)*(genHeat[2]/genEff[2]) + omCost[2]*1e-6*(1+0.03)^(stage-1))/((1+rate)^(stage-1))
                genCost[stage, 3] = ((gas_price/1000)*(genHeat[3]/genEff[3]) + omCost[3]*1e-6*(1+0.03)^(stage-1))/((1+rate)^(stage-1))
            end
            @stageobjective(sp, sum(invCost[stage, g]*x[g] for g in 1:G) + sum(rescale*hours[s]*genCost[stage, g]*y[g, s] for g in 1:G for s in 1:S) + sum(hours[s]*invCost[stage, G+1]*u[s] for s in 1:S))
            JuMP.fix.(demand, dem_sample)
        end
    end
    return model
end

function write_results_to_csv(folder::String, filename::String, id::Int, rep::Int, st::Int, scens::Int, total_time::Float64, backward_pass_time::Float64, iterations::Int, lower_bound::Float64, ci_low::Float64, ci_high::Float64, std_cuts::Int, nonstd_cuts::Int, duality_handler::String, forward_pass::String, backward_pass::String, sampling_scheme::String, fpass_type::Int, mipgap::Float64, M::Int, gap::Float64, delta::Float64, type1_prob::Float64, type2_prob::Float64, lagrangian_dominant::Int = 0, integer_lshaped_dominant::Int = 0, incomparable::Int = 0)
    mkpath(folder)
    csv_file = joinpath(folder, filename)
    results_df = DataFrame(id = [id], rep = [rep], stages = [st], scens = [scens], duality_handler = [duality_handler], forward_pass = [forward_pass], backward_pass = [backward_pass], sampling_scheme = [sampling_scheme], fpass_type = [fpass_type], mipgap = [mipgap], M = [M], total_time = [total_time], backward_pass_time = [backward_pass_time], iterations = [iterations], lower_bound = [lower_bound], ci_low = [ci_low], ci_high = [ci_high], gap = [gap], std_cuts = [std_cuts], nonstd_cuts = [nonstd_cuts], delta = [delta], type1_prob = [type1_prob], type2_prob = [type2_prob], lagrangian_dominant = [lagrangian_dominant], integer_lshaped_dominant = [integer_lshaped_dominant], incomparable = [incomparable])
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

id = parse(Int, ARGS[1])
method_idx = parse(Int, ARGS[2])

instance_params = Dict(54 => (10, 3), 56 => (10, 5), 57 => (10, 7), 58 => (11, 3), 59 => (12, 3))
methods = [
    (SDDP.LagrangianDuality(), SDDP.DefaultMultiBackwardPass(), "Lagrangian", "Default"),
    (SDDP.LagrangianDuality(), SDDP.AnguloMultiBackwardPass(), "Lagrangian", "Angulo"),
    (SDDP.LaporteLouveauxDuality(), SDDP.DefaultMultiBackwardPass(), "Laporte", "Default"),
    (SDDP.LaporteLouveauxDuality(), SDDP.AnguloMultiBackwardPass(), "Laporte", "Angulo"),
    (SDDP.ContinuousConicDuality(), SDDP.DefaultMultiBackwardPass(), "Benders", "Default"),
    (SDDP.LagrangianDuality(), SDDP.ComparisonMultiBackwardPass(), "Lagrangian", "Comparison")
]

st, scens = instance_params[id]
duality_handler, backward_pass, duality_name, backward_name = methods[method_idx + 1]

folder = joinpath(@__DIR__, "gep_data")
threads = 2
mipgap = 1e-4
time_limit = 3600
iter_limit = 10000
lb = 0
G = 6
S = 3
binCoeff = 6
delta = 0.01
type1_prob = 1.28
type2_prob = 1.28
M = 2
iter_pass = 1
final_run = false
fpass_type = 1
postSim = 2000
simTime = 3600.0
forward_pass = SDDP.DefaultMultiForwardPass()
sampling_scheme = SDDP.InSampleMonteCarloMultiple()

prefix = "gep"
fixCost_file = "gep_fixed_cost_jou.csv"
varCost_file = "gep_varCost_jou.csv"
fix_cost_matrix, var_cost_matrix, genMax, genCap, genHeat, genEff, omCost, hours = gep_data_load(folder, fixCost_file, varCost_file)

rep = 1
seed = id * 1000
inFile = joinpath(folder, prefix*"_$(id)_$(rep)_$(st)_$(scens).jls")
support = open(inFile, "r") do f
    deserialize(f)
end

model = gep(st, G, S, lb, binCoeff, fix_cost_matrix, var_cost_matrix, genMax, genCap, genHeat, genEff, omCost, hours, support, threads)
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
csv_filename = "gep_results__$(id)_$(method_idx).csv"
# Extract cut comparison statistics if available
lagrangian_dominant = hasfield(typeof(outputs[1]), :lagrangian_dominant) ? outputs[1].lagrangian_dominant : 0
integer_lshaped_dominant = hasfield(typeof(outputs[1]), :integer_lshaped_dominant) ? outputs[1].integer_lshaped_dominant : 0
incomparable = hasfield(typeof(outputs[1]), :incomparable) ? outputs[1].incomparable : 0
write_results_to_csv(results_folder, csv_filename, id, rep, st, scens, outputs[1].time, outputs[1].backward_pass_time, outputs[1].iter, outputs[1].bb, sim_lower, sim_upper, outputs[1].cs, outputs[1].cns, duality_name, string(nameof(typeof(forward_pass))), backward_name, string(nameof(typeof(sampling_scheme))), fpass_type, mipgap, M, gap, delta, type1_prob, type2_prob, lagrangian_dominant, integer_lshaped_dominant, incomparable)


