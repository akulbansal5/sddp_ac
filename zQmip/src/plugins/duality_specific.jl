"""
this file lists functions that are duality specific
"""


using LinearAlgebra: dot


"""
    _add_mipgap_solver(node::Node; duality_handler::Union{Nothing,ContinuousConicDuality}, mipgap::Number)

Adds the mipgap to the node subproblem if the


    
"""
function _add_mipgap_solver(node::Node; duality_handler::Union{Nothing,ContinuousConicDuality}, mipgap::Number)
    #does nothing for these types of duality handlers
end


"""
    _add_mipgap_solver(node::Node; mipgap::Number)

Adds the mipgap to the node subproblem if the


"""
function _add_mipgap_solver(node::Node; duality_handler::Union{LaporteLouveauxDuality,LagrangianDuality}, mipgap::Number)
    #set the solver gap here
    set_optimizer_attribute(node.subproblem, "mip_gap", mipgap)
end

"""
    Computes the lower bound on  actual cost to go by solving the problems exactly of with gap
"""
function bounds_on_actual_costtogo(items::BackwardPassItems, duality_handler::Union{ContinuousConicDuality, LagrangianDuality, StrengthenedConicDuality, Nothing})

    return dot(items.probability, items.objectives)

end


function bounds_on_actual_costtogo(items::BackwardPassItems, duality_handler::LaporteLouveauxDuality)

    return dot(items.probability, items.bounds)

end
