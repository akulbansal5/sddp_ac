#  Copyright (c) 2017-23, Oscar Dowson and SDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    DefaultForwardPass(; include_last_node::Bool = true)

The default forward pass.

If `include_last_node = false` and the sample terminated due to a cycle, then
the last node (which forms the cycle) is omitted. This can be useful option to
set when training, but it comes at the cost of not knowing which node formed the
cycle (if there are multiple possibilities).
"""
struct DefaultForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultForwardPass,
) where {T}
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        scenario_path, terminated_due_to_cycle =
            sample_scenario(model, options.sampling_scheme)
    end
    final_node = scenario_path[end]
    if terminated_due_to_cycle && !pass.include_last_node
        pop!(scenario_path)
    end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    #storage for objective function on forward pass
    costtogo = Dict{Int64, Float64}()
    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Tuple{Int,Dict{T,Float64}}[]
    current_belief = initialize_belief(model)
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Objective state interpolation.
    objective_state_vector, N =
        initialize_objective_state(model[scenario_path[1][1]])
    objective_states = NTuple{N,Float64}[]
    # Iterate down the scenario.
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        # Objective state interpolation.
        objective_state_vector = update_objective_state(
            node.objective_state,
            objective_state_vector,
            noise,
        )
        if objective_state_vector !== nothing
            push!(objective_states, objective_state_vector)
        end
        # Update belief state, etc.
        if node.belief_state !== nothing
            belief = node.belief_state::BeliefState{T}
            partition_index = belief.partition_index
            current_belief = belief.updater(
                belief.belief,
                current_belief,
                partition_index,
                noise,
            )
            push!(belief_states, (partition_index, copy(current_belief)))
        end
        # ===== Begin: starting state for infinite horizon =====
        starting_states = options.starting_states[node_index]
        if length(starting_states) > 0
            # There is at least one other possible starting state. If our
            # incoming state is more than δ away from the other states, add it
            # as a possible starting state.
            if distance(starting_states, incoming_state_value) >
               options.cycle_discretization_delta
                push!(starting_states, incoming_state_value)
            end
            # TODO(odow):
            # - A better way of randomly sampling a starting state.
            # - Is is bad that we splice! here instead of just sampling? For
            #   convergence it is probably bad, since our list of possible
            #   starting states keeps changing, but from a computational
            #   perspective, we don't want to keep a list of discretized points
            #   in the state-space δ distance apart...
            incoming_state_value =
                splice!(starting_states, rand(1:length(starting_states)))
        end
        # ===== End: starting state for infinite horizon =====
        # Solve the subproblem, note that `duality_handler = nothing`.
        TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path[1:depth],
                duality_handler = nothing,
            )
        end
        # Cumulate the stage_objective.
        cumulative_value += subproblem_results.stage_objective
        # Set the outgoing state value as the incoming state value for the next
        # node.
        incoming_state_value = copy(subproblem_results.state)
        # Add the outgoing state variable to the list of states we have sampled
        # on this forward pass.
        push!(sampled_states, incoming_state_value)
        costtogo[node_index] = JuMP.value(node.bellman_function.global_theta.theta)
    end
    if terminated_due_to_cycle
        # We terminated due to a cycle. Here is the list of possible starting
        # states for that node:
        starting_states = options.starting_states[final_node[1]]
        # We also need the incoming state variable to the final node, which is
        # the outgoing state value of the second to last node:
        incoming_state_value = if pass.include_last_node
            sampled_states[end-1]
        else
            sampled_states[end]
        end
        # If this incoming state value is more than δ away from another state,
        # add it to the list.
        if distance(starting_states, incoming_state_value) >
           options.cycle_discretization_delta
            push!(starting_states, incoming_state_value)
        end
    end
    # ===== End: drop off starting state if terminated due to cycle =====
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        objective_states = objective_states,
        belief_states = belief_states,
        cumulative_value = cumulative_value,
        costtogo = costtogo,
    )
end

# ==================================================================================== #
"""
    DefaultMultiForwardPass(; include_last_node::Bool = true)

The default multiple path forward pass.

If `include_last_node = false` and the sample terminated due to a cycle, then
the last node (which forms the cycle) is omitted. This can be useful option to
set when training, but it comes at the cost of not knowing which node formed the
cycle (if there are multiple possibilities).
"""

struct DefaultMultiForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultMultiForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end


function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultMultiForwardPass,
) where {T}
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.


    # println("Inside forward pass")
    # println(options.sampling_scheme)
    # println("----")


    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        scenario_paths, scenario_paths_noises, terminated_due_to_cycle =
            sample_scenario(model, options.sampling_scheme, options.M)
    end
    
    
    # final_node = scenario_path[end]
    # if terminated_due_to_cycle && !pass.include_last_node
    #     pop!(scenario_path)
    # end

    # println("===== forward pass started successfully")
    #number of scenario paths
    M = length(scenario_paths)
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict(i => Dict{Symbol,Float64}[] for i in 1:M)
    #storage for objective function on forward pass
    costtogo = Dict(i => Dict{Int64, Float64}() for i in 1:M)

    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Dict(i => Tuple{Int,Dict{T,Float64}}[] for i in 1:M)
    
    # Our initial incoming state.

    # A cumulator for the stage-objectives.
    cumulative_values = Dict(i => 0.0 for i in 1:M)

    # NOTE: No objective state interpolation here
    items = ForwardPassItems(T)

    objective_states = NTuple{0,Float64}[]

    #Iterate down the scenario paths
    for i in 1:M
        incoming_state_value = copy(options.initial_state)
        
        
        scenario_path = scenario_paths[i]
        scenario_path_noises = scenario_paths_noises[i]
        
        # Iterate down the scenario.
        for (depth, (node_index, noise)) in enumerate(scenario_path)
            
            
            node    = model[node_index]
            noiseid = scenario_path_noises[depth]


            # NOTE: No objective state interpolation here
            # NOTE: No update in belief state etc.
            # NOTE: No infinite horizon problem here
            # NOTE: No termination due to cycle over here
            # println("Inside forward pass -> Dpeth: $(depth), Node index: $(node_index), Noise ID: $(noiseid)")
            #Takes care of the overlapping scenario paths
            if haskey(items.cached_solutions, (node_index, noiseid))
                sol_index               = items.cached_solutions[(node_index, noiseid)]
                cumulative_values[i]     = cumulative_values[i] + items.stage_objective[sol_index]
                push!(sampled_states[i], copy(items.incoming_state_value[sol_index]))
                costtogo[i][node_index] = items.costtogo[sol_index]
            else
                # println("   =========== executing solve subproblem")
                # Solve the subproblem, note that `duality_handler = nothing`.
                TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        node,
                        incoming_state_value,
                        noise,
                        scenario_path[1:depth],
                        duality_handler = nothing,
                    )
                end

                # println("   =========== ended solve subproblem")
                # Cumulate the stage_objective.
                
                cumulative_values[i] = cumulative_values[i] + subproblem_results.stage_objective

                # Set the outgoing state value as the incoming state value for the next #node.
                incoming_state_value = copy(subproblem_results.state)

                # Add the outgoing state variable to the list of states we have sampled
                # on this forward pass.
                push!(sampled_states[i], incoming_state_value)
                costtogo[i][node_index] = JuMP.value(node.bellman_function.global_theta.theta)
                #update items.cached_solutions

                push!(items.stage_objective, subproblem_results.stage_objective)
                push!(items.incoming_state_value, incoming_state_value)
                push!(items.costtogo, costtogo[i][node_index])
                items.cached_solutions[(node_index, noiseid)] = length(items.costtogo)
            end
        end
    end
    
    # cumulative_value = Dict(i => 0.0 for i in 1:M)
    cum_paths =  [cumulative_values[i] for i in 1:M]
    std_cost  =  Statistics.std(cum_paths)
    avg_cost  =  Statistics.mean(cum_paths)
    stat_ub   =  avg_cost + 1.6*std_cost

    # println(" ======== successfully executed the multi-forward pass ======== ")

    return (
        scenario_paths   = scenario_paths,
        sampled_states   = sampled_states,
        objective_states = objective_states,
        belief_states    = belief_states,
        cumulative_value = stat_ub,
        costtogo         = costtogo,
    )
end





# ==================================================================================== #
"""
    DefaultNestedForwardPass(; include_last_node::Bool = true)

The default nested multiple path forward pass.

If `include_last_node = false` and the sample terminated due to a cycle, then
the last node (which forms the cycle) is omitted. This can be useful option to
set when training, but it comes at the cost of not knowing which node formed the
cycle (if there are multiple possibilities).
"""

struct DefaultNestedForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultNestedForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end


function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultNestedForwardPass,
) where {T}
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.


    println("   ==forward pass")
    # println(options.sampling_scheme)
    # println("----")

    iterations = length(options.log)
    TimerOutputs.@timeit model.timer_output "sample_scenario" begin
        
        if iterations < 1
            scenario_paths, scenario_paths_noises, scenario_paths_prob =
                sample_scenario(model, options.sampling_scheme)

            model.scenario_paths          = scenario_paths
            model.scenario_paths_noises   = scenario_paths_noises
            model.scenario_paths_prob     = scenario_paths_prob
        else
            scenario_paths                = model.scenario_paths
            scenario_paths_noises         = model.scenario_paths_noises
            scenario_paths_prob           = model.scenario_paths_prob
        end
    end


    
    # println("")
    # final_node = scenario_path[end]
    # if terminated_due_to_cycle && !pass.include_last_node
    #     pop!(scenario_path)
    # end

    # println("===== forward pass started successfully")
    #number of scenario paths
    M              = length(scenario_paths)
    path_len       = length(scenario_paths[1])
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Tuple{T,Int}, Dict{Symbol,Float64}}()
    #storage for objective function on forward pass
    costtogo       = Dict(i => Dict{Int, Float64}() for i in 1:path_len)

    
    scenario_trajectory = Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}}()

    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Dict(i => Tuple{Int,Dict{T,Float64}}[] for i in 1:M)
    
    # Our initial incoming state.

    # A cumulator for the stage-objectives.
    cumulative_values = Dict(i => 0.0 for i in 1:M)

    # NOTE: No objective state interpolation here
    items = ForwardPassItems(T)

    objective_states = NTuple{0,Float64}[]

    #Iterate down the scenario paths
    for i in 1:M

        incoming_state_value = copy(options.initial_state)
        scenario_path        = scenario_paths[i]
        scenario_path_noises = scenario_paths_noises[i]
        
        # Iterate down the scenario.
        for (depth, (node_index, noise)) in enumerate(scenario_path)
            
            
            node    = model[node_index]
            noiseid = scenario_path_noises[depth]
            
            # println("verifying: depth $(depth), node_index: $(node_index)")
            

            # NOTE: No objective state interpolation here
            # NOTE: No update in belief state etc.
            # NOTE: No infinite horizon problem here
            # NOTE: No termination due to cycle over here
            # println("Inside forward pass -> Dpeth: $(depth), Node index: $(node_index), Noise ID: $(noiseid)")
            # Takes care of the overlapping scenario paths

            
            if haskey(items.cached_solutions, (node_index, noiseid))
                sol_index               = items.cached_solutions[(node_index, noiseid)]
                stage_OBJ               = items.stage_objective[sol_index]
                cumulative_values[i]     = cumulative_values[i] + stage_OBJ
                incoming_state_value     = items.incoming_state_value[sol_index]

                # push!(sampled_states[i], copy(items.incoming_state_value[sol_index]))
                # costtogo[i][(node_index, noiseid)] = items.costtogo[sol_index]
            else
                # println("   =========== executing solve subproblem")
                # Solve the subproblem, note that `duality_handler = nothing`.


                old_noise_id = 0
                if depth > 1
                    old_noise_id = scenario_path_noises[depth-1]

                end


                TimerOutputs.@timeit model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        node,
                        incoming_state_value,
                        noise,
                        scenario_path[1:depth],
                        duality_handler = nothing,
                        incoming_noise_id = old_noise_id,
                        current_noise_id = noiseid,
                        current_node_index = node_index,
                        write_sub = true, 
                        write_string = "forward_$(iterations)_",
                    )
                end


                # incoming_noise_id = incoming_noise_id,
                # current_noise_id = noise.id,
                # current_node_index = child_node.index, 
                # write_sub = write_sub,
                # write_string = "backward"

                # incoming_noise_id::Number = 1,
                # current_noise_id::Number = 1,
                # current_node_index::Number = 1,
                # write_sub::Bool = false,

                # println("   =========== ended solve subproblem")
                # Cumulate the stage_objective.
                

                # theta_val = JuMP.value(node.bellman_function.global_theta.theta)
                # total_obj = subproblem_results.objective
                

                # @assert total_obj - theta_val == stage_OBJ
                # @assert isapprox(total_obj - theta_val, stage_OBJ, atol=1e-1)


                # println("stage_obj: $(stage_OBJ), calc_obj: $(total_obj - theta_val)")

                stage_OBJ            = subproblem_results.stage_objective
                cumulative_values[i] = cumulative_values[i] + stage_OBJ
                
                
                
                
                # Set the outgoing state value as the incoming state value for the *next* #node.
                incoming_state_value = copy(subproblem_results.state)

                # Add the outgoing state variable to the list of states we have sampled
                # on this forward pass.
                sampled_states[(node_index, noiseid)] = incoming_state_value
                cost_to_go                            = JuMP.value(node.bellman_function.global_theta.theta)

                
                # println("cost to go nested dictionary: $(costtogo)")

                # println("dictionary accessed at nodeindex: $(node_index), noiseid: $(noiseid)")

                costtogo[node_index][noiseid]         = cost_to_go

                # println("cost to go access sucessful")
                
                scenario_trajectory[(node_index, noiseid)] = scenario_path[1:depth]

                # println("scenarion trajectory and scenario path successful")
                
                push!(items.stage_objective, stage_OBJ)
                push!(items.incoming_state_value, incoming_state_value)
                push!(items.costtogo, cost_to_go)
                items.cached_solutions[(node_index, noiseid)] = length(items.stage_objective)
            end
            println("           path: $(i), stage: $(depth), node: $(node_index), noise: $(noiseid), st_obj: $(stage_OBJ), cost-to-go: $(costtogo[node_index][noiseid]), prob: $(scenario_paths_prob[i])")
        end
        println("       path: $(i), cumm_value: $(cumulative_values[i])")
    end
    
    # cumulative_value = Dict(i => 0.0 for i in 1:M)
    stat_ub =  sum([cumulative_values[i]*scenario_paths_prob[i] for i in 1:M])

    # cum_all = [cumulative_values[i] for i in 1:M]
    # println("Iter: $(iterations), stat_ub: $(stat_ub)")


    # std_cost  =  Statistics.std(cum_all)
    # avg_cost  =  Statistics.mean(cum_all)
    # stat_ub   =  avg_cost + 1.6*std_cost
    # stat_ub   =  Statistics.quantile(cum_paths, 0.95)

    # println(" ======== successfully executed the multi-forward pass ======== ")

    return (
        scenario_paths   = scenario_paths,
        sampled_states   = sampled_states,
        objective_states = objective_states,
        belief_states    = belief_states,
        cumulative_value = stat_ub,
        costtogo         = costtogo,
        scenario_trajectory = scenario_trajectory,
    )
end








# ==================================================================================== #
mutable struct RevisitingForwardPass <: AbstractForwardPass
    period::Int
    sub_pass::AbstractForwardPass
    archive::Vector{Any}
    last_index::Int
    counter::Int
end

"""
    RevisitingForwardPass(
        period::Int = 500;
        sub_pass::AbstractForwardPass = DefaultForwardPass(),
    )

A forward pass scheme that generate `period` new forward passes (using
`sub_pass`), then revisits all previously explored forward passes. This can
be useful to encourage convergence at a diversity of points in the
state-space.

Set `period = typemax(Int)` to disable.

For example, if `period = 2`, then the forward passes will be revisited as
follows: `1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, ...`.
"""
function RevisitingForwardPass(
    period::Int = 500;
    sub_pass::AbstractForwardPass = DefaultForwardPass(),
)
    @assert period > 0
    return RevisitingForwardPass(period, sub_pass, Any[], 0, 0)
end

function forward_pass(
    model::PolicyGraph,
    options::Options,
    fp::RevisitingForwardPass,
)
    fp.counter += 1
    if fp.counter - fp.period > fp.last_index
        fp.counter = 1
        fp.last_index = length(fp.archive)
    end
    if fp.counter <= length(fp.archive)
        return fp.archive[fp.counter]
    else
        pass = forward_pass(model, options, fp.sub_pass)
        push!(fp.archive, pass)
        return pass
    end
end

mutable struct RiskAdjustedForwardPass{F,T} <: AbstractForwardPass
    forward_pass::F
    risk_measure::T
    resampling_probability::Float64
    rejection_count::Int
    objectives::Vector{Float64}
    nominal_probability::Vector{Float64}
    adjusted_probability::Vector{Float64}
    archive::Vector{Any}
    resample_count::Vector{Int}
end

"""
    RiskAdjustedForwardPass(;
        forward_pass::AbstractForwardPass,
        risk_measure::AbstractRiskMeasure,
        resampling_probability::Float64,
        rejection_count::Int = 5,
    )

A forward pass that resamples a previous forward pass with
`resampling_probability` probability, and otherwise samples a new forward pass
using `forward_pass`.

The forward pass to revisit is chosen based on the risk-adjusted (using
`risk_measure`) probability of the cumulative stage objectives.

Note that this objective corresponds to the _first_ time we visited the
trajectory. Subsequent visits may have improved things, but we don't have the
mechanisms in-place to update it. Therefore, remove the forward pass from
resampling consideration after `rejection_count` revisits.
"""
function RiskAdjustedForwardPass(;
    forward_pass::AbstractForwardPass,
    risk_measure::AbstractRiskMeasure,
    resampling_probability::Float64,
    rejection_count::Int = 5,
)
    if !(0 < resampling_probability < 1)
        throw(ArgumentError("Resampling probability must be in `(0, 1)`"))
    end
    return RiskAdjustedForwardPass{typeof(forward_pass),typeof(risk_measure)}(
        forward_pass,
        risk_measure,
        resampling_probability,
        rejection_count,
        Float64[],
        Float64[],
        Float64[],
        Any[],
        Int[],
    )
end

function forward_pass(
    model::PolicyGraph,
    options::Options,
    fp::RiskAdjustedForwardPass,
)
    if length(fp.archive) > 0 && rand() < fp.resampling_probability
        r = rand()
        for i in 1:length(fp.adjusted_probability)
            r -= fp.adjusted_probability[i]
            if r > 1e-8
                continue
            end
            pass = fp.archive[i]
            if fp.resample_count[i] >= fp.rejection_count
                # We've explored this pass too many times. Kick it out of the
                # archive.
                splice!(fp.objectives, i)
                splice!(fp.nominal_probability, i)
                splice!(fp.adjusted_probability, i)
                splice!(fp.archive, i)
                splice!(fp.resample_count, i)
            else
                fp.resample_count[i] += 1
            end
            return pass
        end
    end
    pass = forward_pass(model, options, fp.forward_pass)
    push!(fp.objectives, pass.cumulative_value)
    push!(fp.nominal_probability, 0.0)
    fill!(fp.nominal_probability, 1 / length(fp.nominal_probability))
    push!(fp.adjusted_probability, 0.0)
    push!(fp.archive, pass)
    push!(fp.resample_count, 1)
    adjust_probability(
        fp.risk_measure,
        fp.adjusted_probability,
        fp.nominal_probability,
        fp.objectives,
        fp.objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    return pass
end

"""
    RegularizedForwardPass(;
        rho::Float64 = 0.05,
        forward_pass::AbstractForwardPass = DefaultForwardPass(),
    )

A forward pass that regularizes the outgoing first-stage state variables with an
L-infty trust-region constraint about the previous iteration's solution.
Specifically, the bounds of the outgoing state variable `x` are updated from
`(l, u)` to `max(l, x^k - rho * (u - l)) <= x <= min(u, x^k + rho * (u - l))`,
where `x^k` is the optimal solution of `x` in the previous iteration. On the
first iteration, the value of the state at the root node is used.

By default, `rho` is set to 5%, which seems to work well empirically.

Pass a different `forward_pass` to control the forward pass within the
regularized forward pass.

This forward pass is largely intended to be used for investment problems in
which the first stage makes a series of capacity decisions that then influence
the rest of the graph. An error is thrown if the first stage problem is not
deterministic, and states are silently skipped if they do not have finite
bounds.
"""
mutable struct RegularizedForwardPass{T<:AbstractForwardPass} <:
               AbstractForwardPass
    forward_pass::T
    trial_centre::Dict{Symbol,Float64}
    ρ::Float64

    function RegularizedForwardPass(;
        rho::Float64 = 0.05,
        forward_pass::AbstractForwardPass = DefaultForwardPass(),
    )
        centre = Dict{Symbol,Float64}()
        return new{typeof(forward_pass)}(forward_pass, centre, rho)
    end
end

function forward_pass(
    model::PolicyGraph,
    options::Options,
    fp::RegularizedForwardPass,
)
    if length(model.root_children) != 1
        error(
            "RegularizedForwardPass cannot be applied because first-stage is " *
            "not deterministic",
        )
    end
    node = model[model.root_children[1].term]
    if length(node.noise_terms) > 1
        error(
            "RegularizedForwardPass cannot be applied because first-stage is " *
            "not deterministic",
        )
    end
    old_bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for (k, v) in node.states
        if has_lower_bound(v.out) && has_upper_bound(v.out)
            old_bounds[k] = (l, u) = (lower_bound(v.out), upper_bound(v.out))
            x = get(fp.trial_centre, k, model.initial_root_state[k])
            set_lower_bound(v.out, max(l, x - fp.ρ * (u - l)))
            set_upper_bound(v.out, min(u, x + fp.ρ * (u - l)))
        end
    end
    pass = forward_pass(model, options, fp.forward_pass)
    for (k, (l, u)) in old_bounds
        fp.trial_centre[k] = pass.sampled_states[1][k]
        set_lower_bound(node.states[k].out, l)
        set_upper_bound(node.states[k].out, u)
    end
    return pass
end
