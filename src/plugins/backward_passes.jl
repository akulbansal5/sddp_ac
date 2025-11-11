
# using LinearAlgebra: dot
struct DefaultBackwardPass <: AbstractBackwardPass end
struct AnguloBackwardPass <:AbstractBackwardPass end
struct SBendersBackwardPass <:AbstractBackwardPass end
struct DefaultMultiBackwardPass <:AbstractBackwardPass end
struct AnguloMultiBackwardPass <:AbstractBackwardPass end
struct ComparisonMultiBackwardPass <: AbstractBackwardPass end


function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultBackwardPass,     
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}},
    costtogo::Dict{Int64, Float64},
) where {T,NoiseType,N}

    
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end
    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    


    cuts_std    = 0            #standard Benders cuts
    cuts_nonstd = 0            #cuts other than the Benders cuts

    for index in length(scenario_path):-1:1
        
        outgoing_state = sampled_states[index]
       
        objective_state = get(objective_states, index, nothing)
        
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        
        items = BackwardPassItems(T, Noise)
        

        if belief_state !== nothing
            # Update the cost-to-go function for partially observable model.
            for (node_index, belief) in belief_state
                if iszero(belief)
                    continue
                end
                solve_all_children(
                    model,
                    model[node_index],
                    items,
                    belief,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],
                    options.duality_handler,
                    options.mipgap,
                )
            end
            # We need to refine our estimate at all nodes in the partition.
            for node_index in model.belief_partition[partition_index]
                node = model[node_index]
                # Update belief state, etc.
                current_belief = node.belief_state::BeliefState{T}
                for (idx, belief) in belief_state
                    current_belief.belief[idx] = belief
                end
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    options.duality_handler,
                    node.bellman_function,
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability .* items.belief,
                    items.objectives,
                )
                cuts_std += 1
                push!(cuts[node_index], new_cuts)
            end
        else
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if length(node.children) == 0
                continue
            end
            
            
            
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
                options.mipgap,
            )

            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)

            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    options.duality_handler,
                    node.bellman_function,
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,                                #dual_variables
                    items.supports,                             
                    items.probability,                         
                    items.objectives,
                    objofchildren_lp,                           # in order for Laporte and Laouvex cuts to work the input includes mip objective
                )
                cuts_std += 1                     
                push!(cuts[node_index], new_cuts)

                iter = length(options.log)

                if options.refine_at_similar_nodes
                    # Refine the bellman function at other nodes with the same
                    # children, e.g., in the same stage of a Markovian policy graph.
                    for other_index in options.similar_children[node_index]
                        copied_probability = similar(items.probability)
                        other_node = model[other_index]
                        for (idx, child_index) in enumerate(items.nodes)
                            copied_probability[idx] =
                                get(options.Φ, (other_index, child_index), 0.0) *
                                items.supports[idx].probability
                        end
                        new_cuts = refine_bellman_function(
                            model,
                            other_node,
                            options.duality_handler,
                            other_node.bellman_function,
                            options.risk_measures[other_index],
                            outgoing_state,                     #outgoing state
                            items.duals,                
                            items.supports,
                            copied_probability,
                            items.objectives,
                        )
                        push!(cuts[other_index], new_cuts)
                        cuts_std += 1
                    end
                end
            end
        end
    end
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end
    return cuts, cuts_std, cuts_nonstd
end

function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::AnguloBackwardPass,
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}},
    costtogo::Dict{Int64, Float64},
) where {T,NoiseType,N}

    #duality handler changed to continuous conic duality
    continuous_duality = SDDP.ContinuousConicDuality()

    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    

    cuts_std = 0            #standard Benders cuts
    cuts_nonstd = 0         #cuts other than the Benders cuts

    for index in length(scenario_path):-1:1
        
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, Noise)
        if belief_state !== nothing                                            
            # Update the cost-to-go function for partially observable model.    
            for (node_index, belief) in belief_state
                if iszero(belief)
                    continue
                end
                solve_all_children(
                    model,
                    model[node_index],
                    items,
                    belief,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],
                    options.duality_handler,
                    options.mipgap,
                )
            end
            # We need to refine our estimate at all nodes in the partition.
            for node_index in model.belief_partition[partition_index]
                node = model[node_index]
                # Update belief state, etc.
                current_belief = node.belief_state::BeliefState{T}
                for (idx, belief) in belief_state
                    current_belief.belief[idx] = belief
                end
                new_cuts = refine_bellman_function(
                    model,   
                    node,
                    options.duality_handler,                                                       #TODO (akul): check duality handler in refine_bellman_function when belief_state != nothing
                    node.bellman_function,              
                    options.risk_measures[node_index],
                    outgoing_state,                                     
                    items.duals,
                    items.supports,
                    items.probability .* items.belief,
                    items.objectives,                                                              
                )
                push!(cuts[node_index], new_cuts)
            end
        else
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if length(node.children) == 0
                continue
            end
            
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                restore_duality = prepare_backward_pass_node(model, node, continuous_duality, options) 
            end

            #duality handler changed to continuous duality
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                continuous_duality,
                options.mipgap,
            )
            
            objofchildren_lp = bounds_on_actual_costtogo(items, continuous_duality)
            
            
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #restores integer/binary constraints for all children subproblems
                restore_duality()
            end

            #main step of Angulo et al's alternate cutting plane criterion
            flag = 0                                                            #indicates no cut
            objofchildren_mip = nothing
            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0
                new_cuts = refine_bellman_function(
                    model,
                    node,                               #continuous conic duality from lp subproblems
                    continuous_duality,
                    node.bellman_function,              
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability,
                    items.objectives,
                )
                flag = 1            #indicates benders cut
                cuts_std += 1
                push!(cuts[node_index], new_cuts)
            else                  
                #reinitiate items for loading new information
                items = BackwardPassItems(T, Noise)                           

                solve_all_children(
                    model,
                    node,
                    items,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],                      
                    options.duality_handler,
                    options.mipgap
                    )
                
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)

                if options.sense_signal*(costtogo[node_index] - objofchildren_mip) < 0

                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        options.duality_handler,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,
                        items.supports,
                        items.probability,
                        items.objectives,
                        objofchildren_mip,
                    )
                    flag = 2                                #indicates non-benders cut
                    cuts_nonstd += 1
                    push!(cuts[node_index], new_cuts)
                end
            end


            if options.refine_at_similar_nodes
                # Refine the bellman function at other nodes with the same
                # children, e.g., in the same stage of a Markovian policy graph.
                for other_index in options.similar_children[node_index]
                    copied_probability = similar(items.probability)
                    other_node = model[other_index]
                    for (idx, child_index) in enumerate(items.nodes)
                        copied_probability[idx] =
                            get(options.Φ, (other_index, child_index), 0.0) *
                            items.supports[idx].probability
                    end

                    if flag > 0
                        if flag == 1
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,                        
                                continuous_duality,            
                                other_node.bellman_function,        
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            cuts_std += 1
                        else
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                                objofchildren_mip,
                            )
                            cuts_nonstd += 1
                        push!(cuts[other_index], new_cuts)
                        end
                    end
                end
            end
        end
    end
    return cuts, cuts_std, cuts_nonstd
end

function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::SBendersBackwardPass,
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}},
    costtogo::Dict{Int64, Float64},
) where {T,NoiseType,N}

    #duality handler changed to continuous conic duality
    str_continuous_duality = SDDP.StrengthenedConicDuality()


    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    

    cuts_std = 0            #standard Benders cuts
    cuts_nonstd = 0         #cuts other than the Benders cuts

    for index in length(scenario_path):-1:1
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, Noise)
        if belief_state !== nothing
            # Update the cost-to-go function for partially observable model.    
            for (node_index, belief) in belief_state
                if iszero(belief)
                    continue
                end
                solve_all_children(
                    model,
                    model[node_index],
                    items,
                    belief,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],
                    options.duality_handler,
                    options.mipgap,
                )
            end
            # We need to refine our estimate at all nodes in the partition.
            for node_index in model.belief_partition[partition_index]
                node = model[node_index]
                # Update belief state, etc.
                current_belief = node.belief_state::BeliefState{T}
                for (idx, belief) in belief_state
                    current_belief.belief[idx] = belief
                end
                new_cuts = refine_bellman_function(
                    model,   
                    node,
                    options.duality_handler,                                                       #TODO (akul): check duality handler in refine_bellman_function when belief_state != nothing
                    node.bellman_function,              
                    options.risk_measures[node_index],
                    outgoing_state,                                     
                    items.duals,
                    items.supports,
                    items.probability .* items.belief,
                    items.objectives,                                                              
                )
                push!(cuts[node_index], new_cuts)
            end
        else
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if length(node.children) == 0
                continue
            end
            
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                restore_duality = prepare_backward_pass_node(model, node, str_continuous_duality, options) 
            end
            #duality handler changed to continuous duality
            
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                str_continuous_duality,
                options.mipgap,
            )
            
            objofchildren_lp = bounds_on_actual_costtogo(items, str_continuous_duality)
            
            
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #in strenghthened conic duality restore function will return nothing
                restore_duality()
            end


            #main step of Angulo et al's alternate cutting plane criterion
            flag = 0                                                                #indicates no cut
            objofchildren_mip = nothing
            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0

                new_cuts = refine_bellman_function(
                    model,
                    node,                               #continuous conic duality from lp subproblems
                    str_continuous_duality,
                    node.bellman_function,              
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability,
                    items.objectives,
                )
                flag = 1            #indicates benders cut
                cuts_std += 1
                push!(cuts[node_index], new_cuts)
            else                  
                #reinitiate items for loading new information
                items = BackwardPassItems(T, Noise)                           

                #solving mip subproblems
                solve_all_children(
                    model,
                    node,
                    items,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[1:index],                      
                    options.duality_handler,
                    options.mipgap
                    )
                
                
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)

                if options.sense_signal*(costtogo[node_index] - objofchildren_mip) < 0

                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        options.duality_handler,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,
                        items.supports,
                        items.probability,
                        items.objectives,
                        objofchildren_mip,
                    )
                    flag = 2                                #indicates non-benders cut
                    cuts_nonstd += 1
                    push!(cuts[node_index], new_cuts)
                end
            end


            if options.refine_at_similar_nodes
                # Refine the bellman function at other nodes with the same
                # children, e.g., in the same stage of a Markovian policy graph.
                for other_index in options.similar_children[node_index]
                    copied_probability = similar(items.probability)
                    other_node = model[other_index]
                    for (idx, child_index) in enumerate(items.nodes)
                        copied_probability[idx] =
                            get(options.Φ, (other_index, child_index), 0.0) *
                            items.supports[idx].probability
                    end

                    if flag > 0
                        if flag == 1
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,                        
                                str_continuous_duality,            
                                other_node.bellman_function,        
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            cuts_std += 1
                        else
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                                objofchildren_mip,
                            )
                            cuts_nonstd += 1
                        push!(cuts[other_index], new_cuts)
                        end
                    end
                end
            end
        end
    end
    return cuts, cuts_std, cuts_nonstd
end

function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultMultiBackwardPass,     
    scenario_paths::Dict{Int, Vector{Tuple{T, Any}}},
    sampled_states::Dict{Tuple{T,Int}, Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Dict{Int, Vector{Tuple{Int,Dict{T,Float64}}}},
    costtogo::Dict{Int, Dict{Int, Float64}},
    scenario_trajectory::Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}},
    tolerance::Float64 = 1e-6,
) where {T,N}


    iterations = length(options.log)

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end

    # TODO(odow): improve storage type.

    # creating cuts object for storing cuts
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    

    
    
    path_len    = length(scenario_paths[1])
    cuts_std    = 0                                        
    cuts_nonstd = 0
    
    for index in path_len:-1:1

        #note node_index is same as index in case of linear policy gtaphs
        node_index = index
        node = model[node_index]
        if length(node.children) == 0
            continue
        end


        # unique_outgoing_states
        states_visited       = Dict{Int, Dict{Symbol,Float64}}()
        unique_noise_indices = []
        noiseids             = keys(costtogo[node_index])

        for noise_id in noiseids
            

            outgoing_state = sampled_states[(node_index, noise_id)]
            
            TimerOutputs.@timeit model.timer_output "hashing" begin
                visited_flag = false
                for h in unique_noise_indices
                    if outgoing_state == states_visited[h]
                        visited_flag = true
                        break
                    end
                end
                
                if visited_flag == true
                    continue
                else
                    states_visited[noise_id] = outgoing_state
                    push!(unique_noise_indices, noise_id)
                end
            end

            objective_state = nothing
            partition_index, belief_state = (0, nothing)
            items = BackwardPassItems(T, Noise)

            time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
            
            #based on duality_handler mip or lp subproblems are solved below
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_trajectory[(node_index, noise_id)],
                options.duality_handler,
                options.mipgap,
                noise_id,
                false,
                iterations,
                time_left
            )


            
            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)
            cost_to_go       = costtogo[node_index][noise_id]

            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                TimerOutputs.@timeit model.timer_output "cut_addition" begin
                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        options.duality_handler,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,                                #dual_variables
                        items.supports,                             
                        items.probability,                         
                        items.objectives,
                        objofchildren_lp,                           # in order for Laporte and Laouvex cuts to work the input includes mip objective
                    )
                    cuts_std += 1                     
                    push!(cuts[node_index], new_cuts)

                    if options.refine_at_similar_nodes
                        # Refine the bellman function at other nodes with the same
                        # children, e.g., in the same stage of a Markovian policy graph.
                        for other_index in options.similar_children[node_index]
                            copied_probability = similar(items.probability)
                            other_node = model[other_index]
                            for (idx, child_index) in enumerate(items.nodes)
                                copied_probability[idx] =
                                    get(options.Φ, (other_index, child_index), 0.0) *
                                    items.supports[idx].probability
                            end
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,                     #outgoing state
                                items.duals,                
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            push!(cuts[other_index], new_cuts)
                            cuts_std += 1
                        end
                    end
                end
            end
        end
    end

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end

    return cuts, cuts_std, cuts_nonstd
end

#also takes noise_tree as an argument
function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultMultiBackwardPass,     
    scenario_paths::Dict{Int, Vector{Tuple{T, Any}}},
    sampled_states::Dict{Tuple{T,Int}, Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Dict{Int, Vector{Tuple{Int,Dict{T,Float64}}}},
    costtogo::Dict{Int, Dict{Int, Float64}},
    scenario_trajectory::Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}},
    noise_tree::Any,
    tolerance::Float64 = 1e-6,
) where {T,N}

    iterations = length(options.log)
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end

    # TODO(odow): improve storage type.
    cuts        = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))    
    path_len    = length(scenario_paths[1])
    cuts_std    = 0                # benders cuts                        
    cuts_nonstd = 0                # Lagrangian or Integer L-shaped cuts 
    
    for index in path_len:-1:1

        #note node_index is same as index in case of linear policy graphs
        #TODO: remove hardcoding for linear policy graphs (or stage-wise dependence) 
        node_index = index
        node       = model[node_index]
        if length(node.children) == 0
            continue
        end

        # unique_outgoing_states
        #TODO: unique state filtering only works for linear policy graphs (or stage-wise independence)
        states_visited       = Dict{Int, Dict{Symbol,Float64}}()
        unique_noise_indices = []                                  
        noise_nodes          = noise_tree.stageNodes[index]         #noise_tree.stageNodes -> Dict{Int, Vector{ScenarioNode}}
        noise_node_counter   = 1

        for noise_node in noise_nodes
            
            outgoing_state = noise_node.sampled_states
            
            TimerOutputs.@timeit model.timer_output "hashing" begin
                visited_flag = false
                for h in unique_noise_indices
                    if outgoing_state == states_visited[h]
                        visited_flag = true
                        break
                    end
                end
                
                if visited_flag == true
                    #note with stage-wise independence irrespective of the noise term if the outgoing state is same
                    #then we do not need to solve again. So we choose to move on in the for loop.
                    continue
                else
                    states_visited[noise_node_counter] = outgoing_state
                    push!(unique_noise_indices, noise_node_counter)
                    noise_node_counter   += 1
                end
            end

            objective_state               = nothing
            partition_index, belief_state = (0, nothing)
            items                         = BackwardPassItems(T, Noise)
            
            time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
            


            #NOTE: In the scenario trajectory each value is just a dummy empty vector of tuples representing scenario path Tuple{T, Any}[]
            #we do this because the scenario path argument has no use in the solve_all_children and later in solve_subproblem
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_trajectory[(node_index, noise_node.noise_id)],
                options.duality_handler,
                options.mipgap,
                noise_node.noise_id,
                false,
                iterations,
                time_left
            )

            
            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)
            cost_to_go       = noise_node.cost_to_go

            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance

                TimerOutputs.@timeit model.timer_output "cut_addition" begin
                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        options.duality_handler,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,                                #dual_variables
                        items.supports,                             
                        items.probability,                         
                        items.objectives,
                        objofchildren_lp,                           # in order for Laporte and Laouvex cuts to work the input includes mip objective
                    )

                    
                    # Determine if cut is Benders (standard) or non-standard based on duality handler
                    if isa(options.duality_handler, SDDP.ContinuousConicDuality) || 
                       isa(options.duality_handler, SDDP.StrengthenedConicDuality)
                        cuts_std += 1
                    else
                        cuts_nonstd += 1
                    end
                    push!(cuts[node_index], new_cuts)

                    
                    if options.refine_at_similar_nodes
                        # Refine the bellman function at other nodes with the same
                        # children, e.g., in the same stage of a Markovian policy graph.
                        for other_index in options.similar_children[node_index]
                            copied_probability = similar(items.probability)
                            other_node = model[other_index]
                            for (idx, child_index) in enumerate(items.nodes)
                                copied_probability[idx] =
                                    get(options.Φ, (other_index, child_index), 0.0) *
                                    items.supports[idx].probability
                            end
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,                     #outgoing state
                                items.duals,                
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            push!(cuts[other_index], new_cuts)
                            # Determine if cut is Benders (standard) or non-standard based on duality handler
                            if isa(options.duality_handler, SDDP.ContinuousConicDuality) || 
                               isa(options.duality_handler, SDDP.StrengthenedConicDuality)
                                cuts_std += 1
                            else
                                cuts_nonstd += 1
                            end
                        end
                    end
                end
            end
        end
    end

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end

    return cuts, cuts_std, cuts_nonstd
end

function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::AnguloMultiBackwardPass,     
    scenario_paths::Dict{Int, Vector{Tuple{T, Any}}},
    sampled_states::Dict{Tuple{T,Int}, Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Dict{Int, Vector{Tuple{Int,Dict{T,Float64}}}},
    costtogo::Dict{Int, Dict{Int, Float64}},
    scenario_trajectory::Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}},
    tolerance::Float64 = 1e-3,
) where {T,N}

    
    continuous_duality = SDDP.ContinuousConicDuality()

    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    

    
    path_len    = length(scenario_paths[1])
    cuts_std    = 0           
    cuts_nonstd = 0

    for index in path_len:-1:1


        #note node_index is same as index in case of linear policy gtaphs
        node_index = index
        node = model[node_index]
        if length(node.children) == 0
            continue
        end


        # unique_outgoing_states = []
        states_visited      = Dict{Int, Dict{Symbol,Float64}}()
        unique_path_indices = []
        noiseids            = keys(costtogo[node_index])

        for noise_id in noiseids

            outgoing_state = sampled_states[(node_index, noise_id)]
            TimerOutputs.@timeit model.timer_output "hashing" begin
                visited_flag = false
                for h in unique_path_indices
                    if outgoing_state == states_visited[h]
                        visited_flag = true
                        break
                    end
                end
                
                if visited_flag == true
                    continue
                else
                    states_visited[noise_id] = outgoing_state
                    push!(unique_path_indices, noise_id)
                end
            end
            
            objective_state               = nothing
            partition_index, belief_state = (0, nothing)
            items                         = BackwardPassItems(T, Noise)

            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                restore_duality = prepare_backward_pass_node(model, node, continuous_duality, options) 
            end

            
            

            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_trajectory[(node_index, noise_id)],
                continuous_duality,
                options.mipgap,
            )

            objofchildren_lp = bounds_on_actual_costtogo(items, continuous_duality)
            cost_to_go       = costtogo[node_index][noise_id]

            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #restores integer/binary constraints for all children subproblems
                restore_duality()
            end

            flag              = 0
            objofchildren_mip = nothing
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                TimerOutputs.@timeit model.timer_output "benders_cut_addition" begin
                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        continuous_duality,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,                                #dual_variables
                        items.supports,                             
                        items.probability,                         
                        items.objectives,
                        objofchildren_lp,                           # in order for Laporte and Laouvex cuts to work the input includes mip objective
                    )
                end
                flag = 1
                cuts_std += 1                     
                push!(cuts[node_index], new_cuts)
            else
                items = BackwardPassItems(T, Noise) 
                time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
                solve_all_children(
                    model,
                    node,
                    items,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_trajectory[(node_index, noise_id)],                      
                    options.duality_handler,
                    options.mipgap,
                    1,
                    false,
                    1,
                    time_left
                    )

                
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)

                if options.sense_signal*(cost_to_go - objofchildren_mip) < -tolerance

                    TimerOutputs.@timeit model.timer_output "tight_cut_addition" begin
                        new_cuts = refine_bellman_function(
                            model,
                            node,
                            options.duality_handler,
                            node.bellman_function,
                            options.risk_measures[node_index],
                            outgoing_state,
                            items.duals,
                            items.supports,
                            items.probability,
                            items.objectives,
                            objofchildren_mip,
                        )
                    end
                    flag = 2                                #indicates non-benders cut
                    cuts_nonstd += 1
                    push!(cuts[node_index], new_cuts)
                end
            end


            if options.refine_at_similar_nodes
                # Refine the bellman function at other nodes with the same
                # children, e.g., in the same stage of a Markovian policy graph.
                for other_index in options.similar_children[node_index]
                    copied_probability = similar(items.probability)
                    other_node = model[other_index]
                    for (idx, child_index) in enumerate(items.nodes)
                        copied_probability[idx] =
                            get(options.Φ, (other_index, child_index), 0.0) *
                            items.supports[idx].probability
                    end

                    if flag > 0
                        if flag == 1
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,                        
                                continuous_duality,            
                                other_node.bellman_function,        
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            cuts_std += 1
                        else
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                                objofchildren_mip,
                            )
                            cuts_nonstd += 1
                        push!(cuts[other_index], new_cuts)
                        end
                    end
                end
            end
        end
    end
    return cuts, cuts_std, cuts_nonstd
end

#also takes noise_tree as an argument
function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::AnguloMultiBackwardPass,     
    scenario_paths::Dict{Int, Vector{Tuple{T, Any}}},
    sampled_states::Dict{Tuple{T,Int}, Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Dict{Int, Vector{Tuple{Int,Dict{T,Float64}}}},
    costtogo::Dict{Int, Dict{Int, Float64}},
    scenario_trajectory::Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}},
    noise_tree::Any,
    tolerance::Float64 = 1e-3,
) where {T,N}

    iterations = length(options.log)
    
    continuous_duality = SDDP.ContinuousConicDuality()

    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    

    
    path_len    = length(scenario_paths[1])
    cuts_std    = 0           
    cuts_nonstd = 0

    for index in path_len:-1:1


        #note node_index is same as index in case of linear policy gtaphs
        node_index = index
        node = model[node_index]
        if length(node.children) == 0
            continue
        end


        # unique_outgoing_states = []
        states_visited       = Dict{Int, Dict{Symbol,Float64}}()
        unique_noise_indices = []
        noise_nodes          = noise_tree.stageNodes[index]
        noise_node_counter   = 1

        for noise_node in noise_nodes

            outgoing_state = noise_node.sampled_states

            TimerOutputs.@timeit model.timer_output "hashing" begin
                visited_flag = false
                for h in unique_noise_indices
                    if outgoing_state == states_visited[h]
                        visited_flag = true
                        break
                    end
                end
                
                if visited_flag == true
                    continue
                else
                    states_visited[noise_node_counter] = outgoing_state
                    push!(unique_noise_indices, noise_node_counter)
                    noise_node_counter += 1
                end
            end
            
            objective_state               = nothing
            partition_index, belief_state = (0, nothing)
            items                         = BackwardPassItems(T, Noise)

            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                restore_duality = prepare_backward_pass_node(model, node, continuous_duality, options) 
            end

            
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_trajectory[(node_index, noise_node.noise_id)],
                continuous_duality,
                options.mipgap,
                noise_node.noise_id
            )

            objofchildren_lp = bounds_on_actual_costtogo(items, continuous_duality)
            cost_to_go       = noise_node.cost_to_go

            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #restores integer/binary constraints for all children subproblems
                restore_duality()
            end
            
            flag              = 0
            objofchildren_mip = nothing
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                TimerOutputs.@timeit model.timer_output "benders_cut_addition" begin
                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        continuous_duality,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,                                #dual_variables
                        items.supports,                             
                        items.probability,                         
                        items.objectives,
                        objofchildren_lp,                           # in order for Laporte and Laouvex cuts to work the input includes mip objective
                    )
                end
                flag = 1
                cuts_std += 1                     
                push!(cuts[node_index], new_cuts)
            else
                items = BackwardPassItems(T, Noise) 
                time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
                solve_all_children(
                    model,
                    node,
                    items,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_trajectory[(node_index, noise_node.noise_id)],                      
                    options.duality_handler,
                    options.mipgap,
                    noise_node.noise_id,
                    false,
                    iterations,
                    time_left
                    )

                
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)


                if options.sense_signal*(cost_to_go - objofchildren_mip) < -tolerance

                    TimerOutputs.@timeit model.timer_output "tight_cut_addition" begin
                        new_cuts = refine_bellman_function(
                            model,
                            node,
                            options.duality_handler,
                            node.bellman_function,
                            options.risk_measures[node_index],
                            outgoing_state,
                            items.duals,
                            items.supports,
                            items.probability,
                            items.objectives,
                            objofchildren_mip,
                        )
                    end
                    flag = 2                                #indicates non-benders cut
                    cuts_nonstd += 1
                    push!(cuts[node_index], new_cuts)
                end
            end


            if options.refine_at_similar_nodes
                # Refine the bellman function at other nodes with the same
                # children, e.g., in the same stage of a Markovian policy graph.
                for other_index in options.similar_children[node_index]
                    copied_probability = similar(items.probability)
                    other_node = model[other_index]
                    for (idx, child_index) in enumerate(items.nodes)
                        copied_probability[idx] =
                            get(options.Φ, (other_index, child_index), 0.0) *
                            items.supports[idx].probability
                    end

                    if flag > 0
                        if flag == 1
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,                        
                                continuous_duality,            
                                other_node.bellman_function,        
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                            )
                            cuts_std += 1
                        else
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,
                                items.duals,
                                items.supports,
                                copied_probability,
                                items.objectives,
                                objofchildren_mip,
                            )
                            cuts_nonstd += 1
                        push!(cuts[other_index], new_cuts)
                        end
                    end
                end
            end
        end
    end
    return cuts, cuts_std, cuts_nonstd
end
# ========================= ComparisonMultiBackwardPass ========================= #

#also takes noise_tree as an argument
function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::ComparisonMultiBackwardPass,     
    scenario_paths::Dict{Int, Vector{Tuple{T, Any}}},
    sampled_states::Dict{Tuple{T,Int}, Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Dict{Int, Vector{Tuple{Int,Dict{T,Float64}}}},
    costtogo::Dict{Int, Dict{Int, Float64}},
    scenario_trajectory::Dict{Tuple{T,Int}, Vector{Tuple{T, Any}}},
    noise_tree::Any,
    tolerance::Float64 = 1e-6,
) where {T,N}

    iterations = length(options.log)
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end

    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    path_len    = length(scenario_paths[1])
    cuts_std    = 0
    cuts_nonstd = 0
    
    # Statistics for cut comparison
    lagrangian_dominant_count = 0
    integer_lshaped_dominant_count = 0
    incomparable_count = 0
    
    for index in path_len:-1:1
        node_index = index
        node = model[node_index]
        if length(node.children) == 0
            continue
        end

        states_visited = Dict{Int, Dict{Symbol,Float64}}()
        unique_noise_indices = []
        noise_nodes = noise_tree.stageNodes[index]
        noise_node_counter = 1

        for noise_node in noise_nodes
            outgoing_state = noise_node.sampled_states
            
            TimerOutputs.@timeit model.timer_output "hashing" begin
                visited_flag = false
                for h in unique_noise_indices
                    if outgoing_state == states_visited[h]
                        visited_flag = true
                        break
                    end
                end
                
                if visited_flag == true
                    continue
                else
                    states_visited[noise_node_counter] = outgoing_state
                    push!(unique_noise_indices, noise_node_counter)
                    noise_node_counter += 1
                end
            end

            objective_state = nothing
            partition_index, belief_state = (0, nothing)
            time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
            
            # Check if we should do comparison (only for LagrangianDuality and single-cut)
            should_compare = isa(options.duality_handler, SDDP.LagrangianDuality) && 
                            node.bellman_function.cut_type == SINGLE_CUT
            
            if should_compare
                # First, compute Integer L-shaped cut
                items_ll = BackwardPassItems(T, Noise)
                ll_duality = SDDP.LaporteLouveauxDuality()
                
                TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                    restore_duality_ll = prepare_backward_pass(model, ll_duality, options)
                end
                
                solve_all_children(
                    model,
                    node,
                    items_ll,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_trajectory[(node_index, noise_node.noise_id)],
                    ll_duality,
                    options.mipgap,
                    noise_node.noise_id,
                    false,
                    iterations,
                    time_left
                )
                
                objofchildren_ll = bounds_on_actual_costtogo(items_ll, ll_duality)
                cost_to_go = noise_node.cost_to_go
                
                TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                    restore_duality_ll()
                end
                
                # Compute Integer L-shaped cut info (without adding to model)
                model_theta = JuMP.owner_model(node.bellman_function.global_theta.theta)
                if JuMP.objective_sense(model_theta) == MOI.MIN_SENSE
                    L = JuMP.lower_bound(node.bellman_function.global_theta.theta)
                else
                    L = JuMP.upper_bound(node.bellman_function.global_theta.theta)
                end
                
                integer_lshaped_cut = _compute_laplou_average_cut_info(
                    node,
                    outgoing_state,
                    objofchildren_ll,
                    L,
                )
                
                # Now compute Lagrangian cut
                items_lag = BackwardPassItems(T, Noise)
                solve_all_children(
                    model,
                    node,
                    items_lag,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_trajectory[(node_index, noise_node.noise_id)],
                    options.duality_handler,
                    options.mipgap,
                    noise_node.noise_id,
                    false,
                    iterations,
                    time_left
                )
                
                objofchildren_lag = bounds_on_actual_costtogo(items_lag, options.duality_handler)
                
                if options.sense_signal*(cost_to_go - objofchildren_lag) < -tolerance
                    TimerOutputs.@timeit model.timer_output "cut_addition" begin
                        lagrangian_cut = refine_bellman_function(
                            model,
                            node,
                            options.duality_handler,
                            node.bellman_function,
                            options.risk_measures[node_index],
                            outgoing_state,
                            items_lag.duals,
                            items_lag.supports,
                            items_lag.probability,
                            items_lag.objectives,
                            objofchildren_lag,
                        )
                        cuts_nonstd += 1
                        push!(cuts[node_index], lagrangian_cut)
                        
                        # Compare cuts
                        comparison_result = compare_cuts(
                            lagrangian_cut,
                            integer_lshaped_cut,
                            model.objective_sense
                        )
                        
                        if comparison_result == :lagrangian_dominant
                            lagrangian_dominant_count += 1
                        elseif comparison_result == :integer_lshaped_dominant
                            integer_lshaped_dominant_count += 1
                        else
                            incomparable_count += 1
                        end
                    end
                    
                    if options.refine_at_similar_nodes
                        for other_index in options.similar_children[node_index]
                            copied_probability = similar(items_lag.probability)
                            other_node = model[other_index]
                            for (idx, child_index) in enumerate(items_lag.nodes)
                                copied_probability[idx] =
                                    get(options.Φ, (other_index, child_index), 0.0) *
                                    items_lag.supports[idx].probability
                            end
                            new_cuts = refine_bellman_function(
                                model,
                                other_node,
                                options.duality_handler,
                                other_node.bellman_function,
                                options.risk_measures[other_index],
                                outgoing_state,
                                items_lag.duals,
                                items_lag.supports,
                                copied_probability,
                                items_lag.objectives,
                            )
                            push!(cuts[other_index], new_cuts)
                            cuts_nonstd += 1
                        end
                    end
                end
            else
                # For non-LagrangianDuality or multi-cut, behave like DefaultMultiBackwardPass
                items = BackwardPassItems(T, Noise)
                solve_all_children(
                    model,
                    node,
                    items,
                    1.0,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_trajectory[(node_index, noise_node.noise_id)],
                    options.duality_handler,
                    options.mipgap,
                    noise_node.noise_id,
                    false,
                    iterations,
                    time_left
                )
                
                objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)
                cost_to_go = noise_node.cost_to_go
                
                if options.sense_signal*(cost_to_go - objofchildren_lp) < -tolerance
                    TimerOutputs.@timeit model.timer_output "cut_addition" begin
                        new_cuts = refine_bellman_function(
                            model,
                            node,
                            options.duality_handler,
                            node.bellman_function,
                            options.risk_measures[node_index],
                            outgoing_state,
                            items.duals,
                            items.supports,
                            items.probability,
                            items.objectives,
                            objofchildren_lp,
                        )
                        
                        if isa(options.duality_handler, SDDP.ContinuousConicDuality) || 
                           isa(options.duality_handler, SDDP.StrengthenedConicDuality)
                            cuts_std += 1
                        else
                            cuts_nonstd += 1
                        end
                        push!(cuts[node_index], new_cuts)
                        
                        if options.refine_at_similar_nodes
                            for other_index in options.similar_children[node_index]
                                copied_probability = similar(items.probability)
                                other_node = model[other_index]
                                for (idx, child_index) in enumerate(items.nodes)
                                    copied_probability[idx] =
                                        get(options.Φ, (other_index, child_index), 0.0) *
                                        items.supports[idx].probability
                                end
                                new_cuts = refine_bellman_function(
                                    model,
                                    other_node,
                                    options.duality_handler,
                                    other_node.bellman_function,
                                    options.risk_measures[other_index],
                                    outgoing_state,
                                    items.duals,
                                    items.supports,
                                    copied_probability,
                                    items.objectives,
                                )
                                push!(cuts[other_index], new_cuts)
                                if isa(options.duality_handler, SDDP.ContinuousConicDuality) || 
                                   isa(options.duality_handler, SDDP.StrengthenedConicDuality)
                                    cuts_std += 1
                                else
                                    cuts_nonstd += 1
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality()
    end

    return cuts, cuts_std, cuts_nonstd, lagrangian_dominant_count, integer_lshaped_dominant_count, incomparable_count
end
