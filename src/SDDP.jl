#  Copyright (c) 2017-23, Oscar Dowson and SDDP.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module SDDP

import Reexport
Reexport.@reexport using JuMP

import Distributed
import HTTP
import JSON
import MutableArithmetics
import Printf
import Random
import SHA
import Statistics
import TimerOutputs

# Work-around for https://github.com/JuliaPlots/RecipesBase.jl/pull/55
# Change this back to `import RecipesBase` once the fix is tagged.
using RecipesBase
export @stageobjective

#NOTE: The order in which files are included matters

# ============================================================================
# FILE DEPENDENCY STRUCTURE (nested call hierarchy)
# ============================================================================
# SDDP.jl (main module)
#   │
#   ├─ user_interface.jl (model construction)
#   │   └─ Uses: JuMP.jl, bellman_functions.jl, cyclic.jl
#   │
#   ├─ modeling_aids.jl (helper functions)
#   │
#   ├─ plugins/headers.jl (abstract types/interfaces)
#   │   └─ Used by: all plugin files
#   │
#   ├─ binary_expansion.jl (binary expansion utilities)
#   │
#   ├─ JuMP.jl (JuMP extensions)
#   │
#   ├─ cyclic.jl (cycle detection)
#   │   └─ Used by: user_interface.jl, algorithm.jl
#   │
#   ├─ print.jl (printing/logging)
#   │   └─ Called by: algorithm.jl
#   │
#   ├─ algorithm.jl (CORE ALGORITHM - main entry point)
#   │   ├─ Calls: print.jl, cyclic.jl
#   │   ├─ Calls: plugins/parallel_schemes.jl
#   │   │   └─ Calls: plugins/forward_passes.jl
#   │   │       └─ Calls: plugins/sampling_schemes.jl
#   │   │       └─ Calls: algorithm.jl (solve_subproblem)
#   │   ├─ Calls: plugins/backward_passes.jl
#   │   │   ├─ Calls: plugins/backward_sampling_schemes.jl
#   │   │   ├─ Calls: plugins/bellman_functions.jl
#   │   │   │   └─ Calls: plugins/risk_measures.jl
#   │   │   ├─ Calls: plugins/duality_handlers.jl
#   │   │   └─ Calls: algorithm.jl (solve_subproblem, solve_all_children)
#   │   └─ Calls: plugins/stopping_rules.jl
#   │
#   ├─ plugins/risk_measures.jl
#   │   └─ Used by: bellman_functions.jl, backward_passes.jl
#   │
#   ├─ plugins/sampling_schemes.jl
#   │   └─ Called by: forward_passes.jl
#   │
#   ├─ plugins/stopping_rules.jl
#   │   └─ Called by: algorithm.jl (master_loop)
#   │
#   ├─ plugins/local_improvement_search.jl
#   │   └─ Used by: duality_handlers.jl
#   │
#   ├─ plugins/duality_handlers.jl
#   │   └─ Called by: backward_passes.jl, algorithm.jl
#   │
#   ├─ plugins/bellman_functions.jl
#   │   ├─ Called by: user_interface.jl (initialization)
#   │   ├─ Called by: backward_passes.jl (cut generation)
#   │   └─ Calls: plugins/risk_measures.jl
#   │
#   ├─ plugins/parallel_schemes.jl
#   │   └─ Calls: algorithm.jl (iteration)
#   │
#   ├─ plugins/backward_sampling_schemes.jl
#   │   └─ Called by: backward_passes.jl
#   │
#   ├─ plugins/forward_passes.jl
#   │   ├─ Calls: plugins/sampling_schemes.jl
#   │   └─ Calls: algorithm.jl (solve_subproblem)
#   │
#   ├─ plugins/backward_passes.jl
#   │   ├─ Calls: plugins/backward_sampling_schemes.jl
#   │   ├─ Calls: plugins/bellman_functions.jl
#   │   ├─ Calls: plugins/duality_handlers.jl
#   │   └─ Calls: algorithm.jl (solve_subproblem, solve_all_children)
#   │
#   ├─ visualization/*.jl (independent visualization modules)
#   │
#   ├─ deterministic_equivalent.jl
#   │
#   ├─ biobjective.jl
#   │
#   ├─ alternative_forward.jl
#   │   └─ Wraps: plugins/forward_passes.jl (uses different model)
#   │
#   ├─ Experimental.jl
#   │
#   └─ MSPFormat.jl
# ============================================================================

# Modelling interface.
# Core data structures for policy graphs, nodes, and scenario trees
include("user_interface.jl")
# Helper functions for model construction (e.g., lattice approximation)
include("modeling_aids.jl")

# Default definitions for SDDP related modular utilities.
# Abstract types and interfaces for plugins (risk measures, sampling schemes, etc.)
include("plugins/headers.jl")

# Tools for overloading JuMP functions
# Binary expansion utilities for integer/continuous state variables
include("binary_expansion.jl")
# Extensions to JuMP for SDDP-specific variable types (e.g., State variables)
include("JuMP.jl")

# Printing utilities.
# Cycle detection in policy graphs using Tarjan's algorithm
include("cyclic.jl")
# Printing and logging functionality for SDDP iterations
include("print.jl")

# The core SDDP code.
# Main SDDP algorithm implementation (forward/backward passes, training loop)
include("algorithm.jl")

# Specific plugins.
# Risk measure implementations (Expectation, WorstCase, CVaR, etc.)
include("plugins/risk_measures.jl")

# Sampling scheme implementations (Monte Carlo, Historical, PSR, etc.)
include("plugins/sampling_schemes.jl")

# Stopping rule implementations (IterationLimit, TimeLimit, Statistical, TitoHypothesisTesting etc.)
include("plugins/stopping_rules.jl")

# Local improvement or subgradient method for solving the Lagrangian dual
include("plugins/local_improvement_search.jl")

# Duality handlers for integer/mixed-integer problems (Lagrangian, Conic, etc.)
# What dual to we solve
include("plugins/duality_handlers.jl")

# Bellman function representations and cut management
include("plugins/bellman_functions.jl")

# Parallel execution schemes (Serial, Asynchronous, Synchronous)
include("plugins/parallel_schemes.jl")

# Backward sampling schemes for sampling noise terms during backward pass
include("plugins/backward_sampling_schemes.jl")

# Forward pass implementations (Default, Alternative, etc.)
include("plugins/forward_passes.jl")

# Backward pass implementations for different cut generation approaches (Default, Angulo, Strengthened Benders, etc.)
include("plugins/backward_passes.jl")

# Visualization related code.
# Publication-quality quantile plots of simulation results
include("visualization/publication_plot.jl")

# Spaghetti plots showing individual scenario trajectories
include("visualization/spaghetti_plot.jl")

# Interactive web-based dashboard for monitoring SDDP training
include("visualization/dashboard.jl")

# Value function visualization and extraction utilities
include("visualization/value_functions.jl")

# Deterministic equivalent formulation of stochastic programs
include("deterministic_equivalent.jl")

# Biobjective optimization support (Pareto frontier generation)
include("biobjective.jl")

# Alternative forward pass using a different model than the backward pass
include("alternative_forward.jl")

# Experimental features (StochOptFormat I/O, validation scenarios)
include("Experimental.jl")

# MSPFormat file I/O support for reading/writing stochastic programs
include("MSPFormat.jl")

end
