# Call Tree Analysis for SDDP.jl

This document lists all functions called in nested order when `SDDP.train` and `SDDP.LinearPolicyGraph` are invoked.

## 1. SDDP.LinearPolicyGraph Call Chain

### Entry Point: `SDDP.LinearPolicyGraph(builder::Function; stages::Int, kwargs...)`
**Location:** `src/user_interface.jl:884`

**Call Tree (in nested order):**

1. **`SDDP.LinearPolicyGraph`** (`src/user_interface.jl:884`)
   - 1.1. **`LinearGraph(stages)`** (`src/user_interface.jl:425`)
     - 1.1.1. **`Graph(root_node, nodes, edges)`** (constructor)
   - 1.2. **`PolicyGraph(builder, LinearGraph(stages); kwargs...)`** (`src/user_interface.jl:1003`)
     - 1.2.1. **`_validate_graph(graph)`** (`src/user_interface.jl:144`)
     - 1.2.2. **`PolicyGraph(sense, graph.root_node, solver_threads)`** (constructor)
     - 1.2.3. **`BellmanFunction(lower_bound, upper_bound, ...)`** (`src/plugins/bellman_functions.jl:278`)
       - 1.2.3.1. **`InstanceFactory{BellmanFunction}(...)`** (constructor)
     - 1.2.4. **`construct_subproblem(optimizer, direct_mode)`** (`src/user_interface.jl:832`)
       - 1.2.4.1. **`JuMP.direct_model(...)`** or **`JuMP.Model()`**
     - 1.2.5. **`Node(...)`** (constructor, creates node structure)
     - 1.2.6. **`builder(subproblem, node_index)`** (user-provided function)
     - 1.2.7. **`JuMP.set_objective_sense(subproblem, policy_graph.objective_sense)`**
     - 1.2.8. **`initialize_bellman_function(bellman_function, policy_graph, node)`** (`src/plugins/bellman_functions.jl:296`)
       - 1.2.8.1. **`@variable(node.subproblem)`** (creates Θᴳ variable)
       - 1.2.8.2. **`JuMP.set_lower_bound(...)`** / **`JuMP.set_upper_bound(...)`**
       - 1.2.8.3. **`_add_initial_bounds(node.objective_state, Θᴳ)`** (if objective_state exists)
       - 1.2.8.4. **`ConvexApproximation(...)`** (constructor)
       - 1.2.8.5. **`BellmanFunction(...)`** (constructor)
     - 1.2.9. **`initialize_belief_states(policy_graph, graph)`** (`src/user_interface.jl:1136`) [if belief_partition exists]
       - 1.2.9.1. **`construct_belief_update(policy_graph, Set.(graph.belief_partition))`**

## 2. SDDP.train Call Chain

### Entry Point: `SDDP.train(model::PolicyGraph; ...)`
**Location:** `src/algorithm.jl:1204`

**Call Tree (in nested order):**

1. **`SDDP.train`** (`src/algorithm.jl:1204`)
   - 1.1. **`TimerOutputs.reset_timer!(model.timer_output)`**
   - 1.2. **`open(log_file, "a")`**
   - 1.3. **`print_helper(print_banner, log_file_handle)`** (`src/print.jl`)
   - 1.4. **`print_helper(print_problem_statistics, ...)`** (`src/print.jl`)
     - 1.4.1. **`print_problem_statistics(...)`**
   - 1.5. **`numerical_stability_report(io, model, print = print_level > 0)`** (if `run_numerical_stability_report`)
     - 1.5.1. Various model validation functions
   - 1.6. **`print_helper(print_iteration_header, log_file_handle)`** (`src/print.jl`)
   - 1.7. **`convert(Vector{AbstractStoppingRule}, stopping_rules)`**
   - 1.8. **`IterationLimit(iteration_limit)`** (if `iteration_limit !== nothing`)
   - 1.9. **`TimeLimit(time_limit)`** (if `time_limit !== nothing`)
   - 1.10. **`SimulationStoppingRule()`** (if no stopping rules exist)
   - 1.11. **`launch_dashboard()`** (if `dashboard == true`)
   - 1.12. **`Options(...)`** (constructor) (`src/algorithm.jl:1367`)
   - 1.13. **`Random.seed!(seed)`** (if `seed !== nothing`)
   - 1.14. **`master_loop(parallel_scheme, model, options)`** (`src/plugins/parallel_schemes.jl:37` or `236`)
     
     **1.14.1. Serial master_loop** (`src/plugins/parallel_schemes.jl:37`)
     - 1.14.1.1. **`_initialize_solver(model; throw_error = false)`**
     - 1.14.1.2. **`_add_threads_solver(model, threads = model.solver_threads)`** (if `solver_threads !== nothing`)
     - 1.14.1.3. **`iteration(model, options, options.iter_pass)`** (LOOP) (`src/algorithm.jl:931`)
       
       **1.14.1.3.1. iteration function** (`src/algorithm.jl:931`)
       - 1.14.1.3.1.1. **`forward_pass(model, options, options.forward_pass)`** (`src/plugins/forward_passes.jl:23` or `177`)
         
         **1.14.1.3.1.1.1. DefaultForwardPass** (`src/plugins/forward_passes.jl:23`)
         - 1.14.1.3.1.1.1.1. **`sample_scenario(model, options.sampling_scheme)`** (`src/plugins/sampling_schemes.jl:433`)
           - 1.14.1.3.1.1.1.1.1. **`get_root_children(sampling_scheme, graph)`**
           - 1.14.1.3.1.1.1.1.2. **`sample_noise(...)`** (samples root child)
           - 1.14.1.3.1.1.1.1.3. **`get_noise_terms(sampling_scheme, node, node_index)`**
           - 1.14.1.3.1.1.1.1.4. **`get_children(sampling_scheme, node, node_index)`**
           - 1.14.1.3.1.1.1.1.5. **`sample_noise(noise_terms)`** (samples noise)
           - 1.14.1.3.1.1.1.1.6. **`sample_noise(children)`** (samples next node)
         - 1.14.1.3.1.1.1.2. **`initialize_objective_state(model[scenario_path[1][1]])`**
         - 1.14.1.3.1.1.1.3. **`update_objective_state(...)`** (for each node in path)
         - 1.14.1.3.1.1.1.4. **`belief.updater(...)`** (if belief_state exists)
         - 1.14.1.3.1.1.1.5. **`distance(starting_states, incoming_state_value)`** (if infinite horizon)
         - 1.14.1.3.1.1.1.6. **`solve_subproblem(...)`** (for each node) (`src/algorithm.jl:447`)
           - 1.14.1.3.1.1.1.6.1. **`_initialize_solver(node; throw_error = false)`**
           - 1.14.1.3.1.1.1.6.2. **`_add_threads_solver(node, threads = model.solver_threads)`** (if solver_threads exists)
           - 1.14.1.3.1.1.1.6.3. **`_add_mipgap_solver(node, mipgap, duality_handler)`**
           - 1.14.1.3.1.1.1.6.4. **`set_incoming_state(node, state)`** (`src/algorithm.jl:173`)
             - 1.14.1.3.1.1.1.6.4.1. **`JuMP.fix(...)`** (fixes state variable values)
           - 1.14.1.3.1.1.1.6.5. **`parameterize(node, noise)`** (`src/algorithm.jl:297`)
             - 1.14.1.3.1.1.1.6.5.1. **`node.parameterize(noise)`** (user-provided function)
             - 1.14.1.3.1.1.1.6.5.2. **`set_objective(node)`** (`src/algorithm.jl:220`)
               - 1.14.1.3.1.1.1.6.5.2.1. **`get_objective_state_component(node)`**
               - 1.14.1.3.1.1.1.6.5.2.2. **`get_belief_state_component(node)`**
               - 1.14.1.3.1.1.1.6.5.2.3. **`JuMP.set_objective_function(...)`**
           - 1.14.1.3.1.1.1.6.6. **`node.pre_optimize_hook(...)`** (if exists)
           - 1.14.1.3.1.1.1.6.7. **`JuMP.optimize!(node.subproblem)`**
           - 1.14.1.3.1.1.1.6.8. **`JuMP.objective_value(node.subproblem)`**
           - 1.14.1.3.1.1.1.6.9. **`JuMP.primal_status(node.subproblem)`**
           - 1.14.1.3.1.1.1.6.10. **`attempt_numerical_recovery(model, node)`** (if not feasible)
           - 1.14.1.3.1.1.1.6.11. **`get_outgoing_state(node)`**
           - 1.14.1.3.1.1.1.6.12. **`stage_objective_value(node.stage_objective)`**
           - 1.14.1.3.1.1.1.6.13. **`get_dual_solution(node, duality_handler, ...)`**
           - 1.14.1.3.1.1.1.6.14. **`node.post_optimize_hook(...)`** (if exists)
         - 1.14.1.3.1.1.1.7. **`JuMP.value(node.bellman_function.global_theta.theta)`**
       
       - 1.14.1.3.1.2. **`options.forward_pass_callback(forward_trajectory)`**
       - 1.14.1.3.1.3. **`backward_pass(model, options, options.backward_pass, ...)`** (`src/plugins/backward_passes.jl:10`)
         
         **1.14.1.3.1.3.1. DefaultBackwardPass** (`src/plugins/backward_passes.jl:10`)
         - 1.14.1.3.1.3.1.1. **`prepare_backward_pass(model, options.duality_handler, options)`**
         - 1.14.1.3.1.3.1.2. **`solve_all_children(...)`** (for each node in reverse) (`src/algorithm.jl:722`)
           - 1.14.1.3.1.3.1.2.1. **`sample_backward_noise_terms(backward_sampling_scheme, child_node)`** (for each child)
           - 1.14.1.3.1.3.1.2.2. **`belief.updater(...)`** (if belief_state exists)
           - 1.14.1.3.1.3.1.2.3. **`update_objective_state(...)`** (if objective_state exists)
           - 1.14.1.3.1.3.1.2.4. **`solve_subproblem(...)`** (for each child noise combination) - see 1.14.1.3.1.1.1.6 for details
           - 1.14.1.3.1.3.1.2.5. Cache management (stores/retrieves cached solutions)
         - 1.14.1.3.1.3.1.3. **`bounds_on_actual_costtogo(items, options.duality_handler)`**
         - 1.14.1.3.1.3.1.4. **`refine_bellman_function(...)`** (`src/plugins/bellman_functions.jl:402`)
           - 1.14.1.3.1.3.1.4.1. **`adjust_probability(risk_measure, ...)`** (adjusts probabilities based on risk measure)
           - 1.14.1.3.1.3.1.4.2. **`_add_average_cut(...)`** (if SINGLE_CUT) or
           - 1.14.1.3.1.3.1.4.3. **`_add_locals_if_necessary(...)`** (if MULTI_CUT, creates local theta variables)
           - 1.14.1.3.1.3.1.4.4. **`_add_multi_cut(...)`** (if MULTI_CUT)
             - 1.14.1.3.1.3.1.4.4.1. **`@constraint(...)`** (adds Benders cut constraint to subproblem)
             - 1.14.1.3.1.3.1.4.4.2. **`_maybe_delete_cuts(...)`** (if cut deletion is enabled)
         - 1.14.1.3.1.3.1.5. **`restore_duality()`** (cleanup)
       
       - 1.14.1.3.1.4. **`calculate_bound(model)`** (`src/algorithm.jl:848`)
         - 1.14.1.3.1.4.1. **`initialize_belief(model)`**
         - 1.14.1.3.1.4.2. For each root child and noise term:
           - 1.14.1.3.1.4.2.1. **`update_objective_state(...)`** (if objective_state exists)
           - 1.14.1.3.1.4.2.2. **`belief.updater(...)`** (if belief_state exists)
           - 1.14.1.3.1.4.2.3. **`solve_subproblem(...)`** (solves child node)
         - 1.14.1.3.1.4.3. **`adjust_probability(risk_measure, ...)`** (adjusts probabilities)
         - 1.14.1.3.1.4.4. Computes weighted sum of objective values
         - 1.14.1.3.1.4.5. Returns bound value
       
       - 1.14.1.3.1.5. **`Log(...)`** (constructor)
       - 1.14.1.3.1.6. **`push!(options.log, ...)`**
       - 1.14.1.3.1.7. **`convergence_test(model, options.log, options.stopping_rules)`** (`src/plugins/stopping_rules.jl`)
         - 1.14.1.3.1.7.1. **`IterationLimit.convergence_test(...)`** or
         - 1.14.1.3.1.7.2. **`TimeLimit.convergence_test(...)`** or
         - 1.14.1.3.1.7.3. **`SimulationStoppingRule.convergence_test(...)`** or
         - 1.14.1.3.1.7.4. Other stopping rule tests
       
       - 1.14.1.3.1.8. **`IterationResult(...)`** (constructor)
     
     - 1.14.1.4. **`options.post_iteration_callback(result)`**
     - 1.14.1.5. **`log_iteration(options)`** (`src/plugins/parallel_schemes.jl:15`)
       - 1.14.1.5.1. **`options.dashboard_callback(options.log[end], false)`**
       - 1.14.1.5.2. **`_should_log(options, options.log_frequency)`**
       - 1.14.1.5.3. **`print_helper(print_iteration, options.log_file_handle, options.log[end])`**
         - 1.14.1.5.3.1. **`print_iteration(...)`** (`src/print.jl`)
     
     - 1.14.1.6. **Return if `result.has_converged == true`**
   
   - 1.15. **`final_forward_pass(model, options, final_run)`** (`src/algorithm.jl:1418`)
     - 1.15.1. **`forward_pass(...)`** (final forward pass)
     - 1.15.2. **`simulate(...)`** (if needed)
   
   - 1.16. **`count_first_stage_changes(options.log)`** (`src/algorithm.jl:1072`)
   - 1.17. **`TrainingResults(status, log)`** (constructor)
   - 1.18. **`confidence_interval(...)`** (for statistics)
   - 1.19. **`print_helper(print_footer, log_file_handle, training_results)`** (`src/print.jl`)
   - 1.20. **`TimerOutputs.print_timer(...)`** (if `print_level > 1`)
   - 1.21. **`close(log_file_handle)`**

## Summary of Key Functions

### Core Algorithm Functions:
- `master_loop`: Main training loop
- `iteration`: Single SDDP iteration (forward + backward pass)
- `forward_pass`: Samples scenario and solves subproblems forward
- `backward_pass`: Generates cuts and refines value functions backward
- `solve_subproblem`: Solves a single node's optimization problem
- `refine_bellman_function`: Adds cuts to approximate cost-to-go function
- `calculate_bound`: Computes lower bound by solving root node

### Model Construction Functions:
- `LinearPolicyGraph`: Creates linear policy graph
- `PolicyGraph`: Main policy graph constructor
- `initialize_bellman_function`: Sets up value function approximation
- `_validate_graph`: Validates graph structure

### Utility Functions:
- `convergence_test`: Checks stopping criteria
- `sample_scenario`: Samples a scenario path through the graph
- `log_iteration`: Logs iteration information
- Various print and logging functions

---

## 3. 2-Day Intensive Learning Plan

This plan prioritizes understanding the critical path over completeness. Focus on the core algorithm; details can come later.

### Day 1: Core Algorithm Flow (8-10 hours)

#### Morning (3-4 hours): Architecture and Model Construction

**Hour 1: High-Level Overview**
- [x] Read `CALL_TREE_ANALYSIS.md` (this document) thoroughly (30 min)
- [x] Read `SDDP.jl` - module structure (15 min)
- [x] Read `plugins/headers.jl` - abstract types and interfaces (45 min)
  - Focus: `AbstractForwardPass`, `AbstractBackwardPass`, `AbstractRiskMeasure`, `AbstractDualityHandler`
  - Goal: Understand the plugin architecture

**Hour 2: Model Construction (Breadth-First)**
- [x] Read `user_interface.jl:884-1130` - `LinearPolicyGraph` and `PolicyGraph` (60 min)
  - Focus: How models are built, not every detail
  - Key: `PolicyGraph` constructor, `Node` creation, `BellmanFunction` initialization
- [x] Skim `plugins/bellman_functions.jl:278-344` - `initialize_bellman_function` (30 min)
  - Goal: Understand how value functions are set up

**Hour 3: Data Structures**
- [x] Find and read type definitions:
  - `PolicyGraph` struct
  - `Node` struct
  - `BellmanFunction` struct
  - `Options` struct
  - `Log` struct
- [x] Goal: Know what data is stored, not how it's used yet

**Hour 4: Training Entry Point**
- [x] Read `algorithm.jl:1204-1392` - `train` function (60 min)
  - Focus: Setup, not the loop yet
  - Understand: Options creation, stopping rules, initialization
- [x] Read `plugins/parallel_schemes.jl:37-62` - `master_loop` (Serial version) (30 min)
  - Goal: Understand the iteration loop structure

#### Afternoon (4-5 hours): Forward Pass and Subproblem Solving

**Hour 5: Forward Pass Overview**
- [x] Read `plugins/forward_passes.jl:23-154` - `DefaultForwardPass` (90 min)
  - Focus: High-level flow
  - Understand: Scenario sampling → state updates → solve subproblems

**Hour 6: Scenario Sampling**
- [x] Read `plugins/sampling_schemes.jl:433-485` - `sample_scenario` (60 min)
  - Goal: How scenarios are generated
  - Skip: Other sampling scheme variants for now

**Hours 7-8: Solve Subproblem (Depth-First)**
- [x] Read `algorithm.jl:447-544` - `solve_subproblem` (25 min)
  - Read every line carefully
  - Understand: State fixing, parameterization, optimization, result extraction
- [x] Read `algorithm.jl:173-216` - `set_incoming_state` and `get_outgoing_state` (25 min)
- [x] Read `algorithm.jl:297-301` - `parameterize` (10 min)
- [x] Read `algorithm.jl:220-295` - `set_objective` (15 min)
- [ ] Read get_dual_solution
  - Goal: Understand how subproblems are solved

**Hour 9: Integration Check**
- [x] Trace through one complete forward pass mentally
- [x] Write notes: What happens from `train` → `forward_pass` → `solve_subproblem`
- [x] Identify gaps in understanding

---

### Day 2: Backward Pass and Cut Generation (8-10 hours)

#### Morning (4-5 hours): Backward Pass

**Hour 1: Backward Pass Overview**
- [x] Read `plugins/backward_passes.jl:10-166` - `DefaultBackwardPass` (90 min)
  - Focus: High-level flow
  - Understand: Reverse traversal, solving children, generating cuts

- [x] Read `plugins/backward_passes.jl:10-166` - `AnguloBackwardPass` (90 min)

**Hour 2: Solve All Children**
- [x] Read `algorithm.jl:722-821` - `solve_all_children` (90 min)
  - Understand: How child nodes are solved in backward pass
  - Note: Similar to forward pass but in reverse

**Hour 3: Cut Generation (Depth-First)**
- [-] Read `plugins/bellman_functions.jl:402-452` - `refine_bellman_function` (60 min)
  - Understand: How cuts are created
- [-] Find and read `_add_average_cut` or `_add_multi_cut` (30 min)
  - Goal: Understand the actual cut constraint being added

**Hour 4: Bound Calculation**
- [-] Read `algorithm.jl:848-920` - `calculate_bound` (60 min)
  - Understand: How lower bounds are computed
  - Goal: Complete the iteration picture

- [-] Read bounds_on_actual_costtogo

**Hour 5: Complete Iteration**
- [ ] Re-read `algorithm.jl:931-1053` - `iteration` function (60 min)
  - Now with context from forward/backward passes
  - Goal: Understand the complete iteration

#### Afternoon (3-4 hours): Integration and Testing

**Hour 6: Full Trace**
- [ ] Trace through one complete iteration:
  - `train` → `master_loop` → `iteration` → `forward_pass` → `backward_pass` → `calculate_bound`
- [ ] Draw a diagram of the flow
- [ ] Write a summary of what happens in each step

**Hour 7: Convergence and Logging**
- [ ] Read `plugins/stopping_rules.jl` - key stopping rules (30 min)
  - Focus: `IterationLimit`, `TimeLimit`, `SimulationStoppingRule`
- [ ] Read `plugins/parallel_schemes.jl:15-24` - `log_iteration` (15 min)
- [ ] Understand: How training terminates

**Hour 8: Fill Gaps and Test Understanding**
- [ ] Review your notes
- [ ] Identify what you still don't understand
- [ ] Pick 2-3 specific functions you're unclear on and read them
- [ ] Try to explain the algorithm to yourself (or write it down)

**Hour 9: Optional - Specific Areas**
- [ ] If time: Read one plugin variant (e.g., different forward pass or backward pass)
- [ ] If time: Understand `get_dual_solution` in `plugins/duality_handlers.jl`
- [ ] If time: Review `calculate_bound` more carefully

---

### What to Skip (For Now)

- Visualization code (`visualization/`)
- Experimental features (`Experimental.jl`)
- Alternative implementations (`alternative_forward.jl`)
- Biobjective code (`biobjective.jl`)
- MSP format (`MSPFormat.jl`)
- Most plugin variants (focus on defaults)
- Printing utilities (skim only)
- Parallel schemes (focus on Serial)

### Success Criteria

By end of Day 2, you should be able to:
1. ✅ Explain the SDDP algorithm flow in your own words
2. ✅ Trace through one iteration: forward pass → backward pass → bound calculation
3. ✅ Understand how subproblems are solved
4. ✅ Understand how cuts are generated and added
5. ✅ Know where to look when you need to modify something

### Tips for Success

1. **Use a debugger**: Set breakpoints and step through execution
2. **Take notes**: Write what each function does in 1-2 sentences
3. **Draw diagrams**: Visualize the call flow
4. **Don't get stuck**: If something is unclear after 30 min, note it and move on
5. **Test understanding**: After each major section, explain it to yourself

### Realistic Expectations

**In 2 days you CAN:**
- ✅ Understand the core algorithm flow
- ✅ Know where key functions are located
- ✅ Understand the main data structures
- ✅ Be able to make simple modifications
- ✅ Have a roadmap for deeper understanding later

**You will NOT:**
- ❌ Understand every edge case
- ❌ Know all plugin variants
- ❌ Understand all optimization details
- ❌ Be an expert (but you'll have a solid foundation)

---

## 4. Learning Strategy: Breadth-First → Depth-First → Breadth-First

### Phase 1: Breadth-First Overview (Day 1 Morning)
**Goal**: Understand the architecture and data flow

**Why breadth-first first:**
- Avoids getting lost in details
- Shows how components connect
- Reveals the critical path
- Helps prioritize what to study deeply

### Phase 2: Depth-First on Critical Path (Day 1 Afternoon - Day 2 Morning)
**Goal**: Understand the core algorithm execution

**Why depth-first here:**
- The algorithm is sequential - you need to follow the flow
- Dependencies are clear (forward_pass → backward_pass → bound)
- Builds a complete mental model of one path
- Easier to debug and modify later

### Phase 3: Breadth-First for Plugins (Day 2 Afternoon - Optional)
**Goal**: Understand optional/extensible components

**Why breadth-first here:**
- Plugins are mostly independent
- You already understand the core - now see variations
- Faster to compare than to go deep into each

---

## 5. Quick Reference: File Locations

### Core Algorithm Files:
- `src/algorithm.jl` - Main training loop, subproblem solving, bound calculation
- `src/plugins/forward_passes.jl` - Forward pass implementations
- `src/plugins/backward_passes.jl` - Backward pass implementations
- `src/plugins/bellman_functions.jl` - Value function approximation and cut generation
- `src/plugins/sampling_schemes.jl` - Scenario sampling
- `src/plugins/parallel_schemes.jl` - Parallel execution (focus on Serial)

### Model Construction Files:
- `src/user_interface.jl` - Policy graph construction
- `src/plugins/headers.jl` - Abstract types and interfaces

### Utility Files:
- `src/plugins/stopping_rules.jl` - Convergence criteria
- `src/plugins/duality_handlers.jl` - Dual variable extraction
- `src/plugins/risk_measures.jl` - Risk measure implementations
- `src/print.jl` - Logging and printing



## 6. Things to double check later

- what function stubs do I need to include in headers.jl file?
  check what additional functions I defined that needs to be added to headers.jl?
