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

# Modelling interface.
include("user_interface.jl")
include("modeling_aids.jl")

# Default definitions for SDDP related modular utilities.
include("plugins/headers.jl")

# Tools for overloading JuMP functions
include("binary_expansion.jl")
include("JuMP.jl")

# Printing utilities.
include("cyclic.jl")
include("print.jl")

# The core SDDP code.
include("algorithm.jl")

# println(">>>>>>>>>>>>>>>>>> algorithm.jl included")

# Specific plugins.
include("plugins/risk_measures.jl")
# println(">>>>>>>>>>>>>>>>>> risk_measures.jl included")

include("plugins/sampling_schemes.jl")
# println(">>>>>>>>>>>>>>>>>> sampling_schemes.jl included")

include("plugins/stopping_rules.jl")
# println(">>>>>>>>>>>>>>>>>> stopping_rules.jl included")

include("plugins/local_improvement_search.jl")
# println(">>>>>>>>>>>>>>>>>> local_improvement_search.jl included")

include("plugins/duality_handlers.jl")
# println(">>>>>>>>>>>>>>>>>> duality_handlers.jl included")

# include("plugins/duality_specific.jl")
include("plugins/bellman_functions.jl")
#println(">>>>>>>>>>>>>>>>>> bellman_functions.jl included")


include("plugins/parallel_schemes.jl")
#println(">>>>>>>>>>>>>>>>>> parallel_schemes.jl included")

include("plugins/backward_sampling_schemes.jl")
#println(">>>>>>>>>>>>>>>>>> backward_sampling_schemes.jl included")


include("plugins/forward_passes.jl")
#println(">>>>>>>>>>>>>>>>>> forward_passes.jl included")


include("plugins/backward_passes.jl")
#println(">>>>>>>>>>>>>>>>>> backward_passes.jl included")

# Visualization related code.
include("visualization/publication_plot.jl")
#println(">>>>>>>>>>>>>>>>>> publication_plot.jl included")

include("visualization/spaghetti_plot.jl")
#println(">>>>>>>>>>>>>>>>>> spaghetti_plot.jl included")

include("visualization/dashboard.jl")
#println(">>>>>>>>>>>>>>>>>> dashboard.jl included")

include("visualization/value_functions.jl")
#println(">>>>>>>>>>>>>>>>>> value_functions.jl included")

include("deterministic_equivalent.jl")
#println(">>>>>>>>>>>>>>>>>> deterministic_equivalent.jl included")

include("biobjective.jl")
#println(">>>>>>>>>>>>>>>>>> biobjective.jl included")

include("alternative_forward.jl")
#println(">>>>>>>>>>>>>>>>>> alternative_forward.jl included")

include("Experimental.jl")
#println(">>>>>>>>>>>>>>>>>> Experimental.jl included")

include("MSPFormat.jl")
#println(">>>>>>>>>>>>>>>>>> MSPFormat.jl included")

end
