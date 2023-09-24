

struct DefaultBackwardPass <: AbstractBackwardPass end
struct AnguloBackwardPass <:AbstractBackwardPass end


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
            )
            # println("finished solving all children")
            # objofchildren = dot(items.probability, items.objectives)
            # println("At node: $node_index objective: $objofchildren")
            # println("adding cuts")

            # println("dual variables: ", items.duals)
            objofchildren_lp = dot(items.probability, items.objectives)

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