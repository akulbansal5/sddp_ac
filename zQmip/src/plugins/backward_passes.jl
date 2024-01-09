
# using LinearAlgebra: dot
struct DefaultBackwardPass <: AbstractBackwardPass end
struct AnguloBackwardPass <:AbstractBackwardPass end
struct SBendersBackwardPass <:AbstractBackwardPass end
struct DefaultMultiBackwardPass <:AbstractBackwardPass end
struct AnguloMultiBackwardPass <:AbstractBackwardPass end


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

    # println("--starting backward pass--")
    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end
    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    # println("starting for loop")


    cuts_std    = 0            #standard Benders cuts
    cuts_nonstd = 0            #cuts other than the Benders cuts

    for index in length(scenario_path):-1:1
        # println("       Index in scenario_path $index")
        outgoing_state = sampled_states[index]
        # println("step 1")
        objective_state = get(objective_states, index, nothing)
        # println("step 2")
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        # println("step 3")
        items = BackwardPassItems(T, Noise)
        # println("formalities finished in for loop")

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
            
            
            # println("solving all children")
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
            # println("finished solving all children")
            # objofchildren = dot(items.probability, items.objectives)
            # println("At node: $node_index objective: $objofchildren")
            # println("adding cuts")

            # println("dual variables: ", items.duals)
            #note this may not necessarily be a 
            # objofchildren_lp = dot(items.probability, items.objectives)
            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)

            # println("refine bellman function")
            # println("obj of node $(node_index)'s children: $(objofchildren_lp)")

            # println("       node: $(node_index), costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")

            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
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
                # JuMP.write_to_file(node.subproblem, "subprob_mpo_$(node.index)_$(iter).lp")
                # println("   printed backward subproblem at node $(node.index) and iteration $(iter).")
                

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
        # println("       Index of scenario path: $index")
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, Noise)
        if belief_state !== nothing                                             #TODO (akul) : note in belief state we are not preparing the backward pass
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
            # println("       prepare backward pass")
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                # println("       prepare backward pass")
                restore_duality = prepare_backward_pass_node(model, node, continuous_duality, options) 
            end
            #duality handler changed to continuous duality
            # println("       solve all children at node: $(node.index), path index: $(index) ")
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
            # objofchildren_lp = dot(items.probability, items.objectives)
            objofchildren_lp = bounds_on_actual_costtogo(items, continuous_duality)
            
            # println("       solved all children")
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #restores integer/binary constraints for all children subproblems
                restore_duality()
            end


            #main step of Angulo et al's alternate cutting plane criterion
            # println(" check if benders cut can be applied")
            # println("       costtogo: $(costtogo[node_index]), obj of children: $(objofchildren_lp)")

            flag = 0 #indicates no cut
            objofchildren_mip = nothing
            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
                # println("       adding standard Benders cut")
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

                # println("       solving MIP subproblems--")
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
                
                
                # objofchildren_mip = dot(items.probability, items.objectives)
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)

                # println("       At scenario path index: $index MIP objective: $objofchildren_mip")
                # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                if options.sense_signal*(costtogo[node_index] - objofchildren_mip) < 0
                    # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                    # println("       add LapLov cut/Lag cut procedure")
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
        # println("       Index of scenario path: $index")
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, Noise)
        if belief_state !== nothing                                             #TODO (akul) : note in belief state we are not preparing the backward pass
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
            # println("       prepare backward pass")
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                # println("       prepare backward pass")
                restore_duality = prepare_backward_pass_node(model, node, str_continuous_duality, options) 
            end
            #duality handler changed to continuous duality
            # println("       solve all children at node: $(node.index), path index: $(index) ")
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
            # objofchildren_lp = dot(items.probability, items.objectives)
            objofchildren_lp = bounds_on_actual_costtogo(items, str_continuous_duality)
            
            # println("       solved all children")
            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #in strenghthened conic duality restore function will return nothing
                restore_duality()
            end


            #main step of Angulo et al's alternate cutting plane criterion
            # println(" check if benders cut can be applied")
            # println("       costtogo: $(costtogo[node_index]), obj of children: $(objofchildren_lp)")

            flag = 0 #indicates no cut
            objofchildren_mip = nothing
            if options.sense_signal*(costtogo[node_index] -  objofchildren_lp) < 0
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
                # println("       adding standard Benders cut")
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

                # println("       solving MIP subproblems--")
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
                
                
                # objofchildren_mip = dot(items.probability, items.objectives)
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)

                # println("       At scenario path index: $index MIP objective: $objofchildren_mip")
                # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                if options.sense_signal*(costtogo[node_index] - objofchildren_mip) < 0
                    # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                    # println("       add LapLov cut/Lag cut procedure")
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
    # println("   >>starting backward pass at iteration: $(iterations)")

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end

    # TODO(odow): improve storage type.
    # println("creating cuts object for storing cuts")
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
                    # println("       node_index: $(node_index), noise_id: $(noise_id) already cached")
                    continue
                else
                    states_visited[noise_id] = outgoing_state
                    push!(unique_noise_indices, noise_id)
                end
            end

            objective_state = nothing
            partition_index, belief_state = (0, nothing)
            items = BackwardPassItems(T, Noise)


            # log[end].time
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
                noise_id,
                false,
                iterations,
                time_left
            )


            

            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)
            cost_to_go       = costtogo[node_index][noise_id]

            # println("       node: $(node_index), costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)"
            # println("       node_index: $(node_index), noise_id: $(noise_id), costtogo: $(cost_to_go), objofchilds: $(objofchildren_lp)")
            
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")

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

                    
                    # JuMP.write_to_file(node.subproblem, "subprob_mpo_$(node.index)_$(iter).lp")
                    # println("   printed backward subproblem at node $(node.index) and iteration $(iter).")
                    

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

    # println("   number of cuts added: $(cuts_std)")
    # println("=============== finished backward pass ================== ")
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
    noise_tree::Any,
    tolerance::Float64 = 1e-6,
) where {T,N}


    iterations = length(options.log)
    # println("   >>starting backward pass at iteration: $(iterations)")

    TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
        restore_duality =
            prepare_backward_pass(model, options.duality_handler, options)
    end

    # TODO(odow): improve storage type.
    # println("creating cuts object for storing cuts")
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
        # noiseids             = keys(costtogo[node_index])
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
                    # println("       node_index: $(node_index), noise_id: $(noise_id) already cached")
                    continue
                else
                    states_visited[noise_node_counter] = outgoing_state
                    push!(unique_noise_indices, noise_node_counter)
                end
            end

            objective_state = nothing
            partition_index, belief_state = (0, nothing)
            items = BackwardPassItems(T, Noise)
            

            
            # log[end].time
            time_left = length(options.log) > 0 ? options.time_limit - options.log[end].time : nothing
            


            #NOTE: In the scenario trajectory each value is just a dummy empty vector of tuples representing scenario path Tuple{T, Any}[]
            #we do this because the scenario path argument has no use in the solve_all_children and later in solve_subproblem
            # println("       solving all children")
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

            # println("       successfully solved all children")

            

            objofchildren_lp = bounds_on_actual_costtogo(items, options.duality_handler)
            cost_to_go       = noise_node.cost_to_go

            # println("       node: $(node_index), costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)"
            # println("       node_index: $(node_index), noise_id: $(noise_id), costtogo: $(cost_to_go), objofchilds: $(objofchildren_lp)")
            
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")

                # println("   refining bellman function")
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
                    # println("   refined bellman function")
                    
                    # JuMP.write_to_file(node.subproblem, "subprob_mpo_$(node.index)_$(iter).lp")
                    # println("   printed backward subproblem at node $(node.index) and iteration $(iter).")
                    
                    
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

    # println("   number of cuts added: $(cuts_std)")
    # println("=============== finished backward pass ================== ")
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

    # println("--starting backward pass--")
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

            # log[end].time
            

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

            # println("       node: $(node_index), costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
            
            flag              = 0
            objofchildren_mip = nothing
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
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


                # JuMP.write_to_file(node.subproblem, "subprob_mpo_$(node.index)_$(iter).lp")
                # println("   printed backward subproblem at node $(node.index) and iteration $(iter).")
                
                if options.sense_signal*(cost_to_go - objofchildren_mip) < -tolerance
                    # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                    # println("       add LapLov cut/Lag cut procedure")
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
    noise_tree:Any,
    tolerance::Float64 = 1e-3,
) where {T,N}

    iterations = length(options.log)
    # println("--starting backward pass--")
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
                end
            end
            
            objective_state               = nothing
            partition_index, belief_state = (0, nothing)
            items                         = BackwardPassItems(T, Noise)

            TimerOutputs.@timeit model.timer_output "prepare_backward_pass" begin
                #only relaxes subproblems for children node
                restore_duality = prepare_backward_pass_node(model, node, continuous_duality, options) 
            end

            # log[end].time
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

            # println("       node: $(node_index), costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
            
            flag              = 0
            objofchildren_mip = nothing
            if options.sense_signal*(cost_to_go -  objofchildren_lp) < -tolerance
                # println("       costtogo: $(costtogo[node_index]), obj of children lp: $(objofchildren_lp)")
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
                    noise_node.noise_id,
                    false,
                    iterations,
                    time_left
                    )

                
                objofchildren_mip = bounds_on_actual_costtogo(items, options.duality_handler)


                # JuMP.write_to_file(node.subproblem, "subprob_mpo_$(node.index)_$(iter).lp")
                # println("   printed backward subproblem at node $(node.index) and iteration $(iter).")
                
                if options.sense_signal*(cost_to_go - objofchildren_mip) < -tolerance
                    # println("       costtogo: $(costtogo[node_index]), obj of children mip: $(objofchildren_mip)")
                    # println("       add LapLov cut/Lag cut procedure")
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