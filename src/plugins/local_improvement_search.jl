#  Copyright (c) 2017-23, Oscar Dowson and SDDP.jl contributors and contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module LocalImprovementSearch

_norm(x) = sqrt(sum(xi^2 for xi in x))

abstract type AbstractSearchMethod end

function minimize(f::Function, x₀::Vector{Float64})
    return minimize(f, BFGS(1000), x₀)
end

mutable struct BFGS <: AbstractSearchMethod
    evaluation_limit::Int
    ftol::Float64
    gtol::Float64
    gaptol::Float64
    bound::Union{Float64,Nothing}
    function BFGS(evaluation_limit::Int, ftol::Float64 = 1e-4, gtol::Float64 = 1e-5, gaptol::Float64 = 1e-2, bound::Union{Float64,Nothing} = nothing)
        new(evaluation_limit, ftol, gtol)
    end
end

"""
    minimize(f::Function, x₀::Vector{Float64})

Minimizes a convex function `f` using first-order information.

The algorithm is a modified version of BFGS, with a specialized back-tracking
inexact line-search.

Compared to off-the-shelf implementations, it has a number of features tailored
to this purpose:

 * Infeasibility is indicated by the function returning `nothing`. No other
   constraint information is given.
 * Sub-optimal solutions are okay, so we should focus on improving the feasible
   starting point, instead of finding the global minimizer.
 * `f` can be piecewise-linear convex with non-differentiable points.

## Arguments

 * `f(::Vector{Float64})`: takes a vector `x` and returns one of the following:
   * `nothing` if `x` is infeasible
   * `(f, Δf)::Tuple{Float64,Vector{Float64}`:  a tuple of the function
     evaluation and first-order gradient information.
 * `x₀::Vector{Float64}`: a feasible starting point.
"""
function minimize(f::F, bfgs::BFGS, x₀::Vector{Float64}, time_left::Union{Number,Nothing}=nothing, curr_bound::Union{Number, Nothing} = nothing) where {F<:Function}
    # Initial estimte for the Hessian matrix in BFGS
    B = zeros(length(x₀), length(x₀))
    start_time = time()

    for i in 1:size(B, 1)
        B[i, i] = 1.0
    end

    # We assume that the initial iterate is feasible
    xₖ = x₀
    fₖ, ∇fₖ = f(xₖ)::Tuple{Float64,Vector{Float64}}  #the gradient comes through f(x_k)
       
    
    # Initial step-length
    αₖ = 1.0

    # Evaluation counter
    # evals = Ref(0)
    evals = Ref(bfgs.evaluation_limit)
    curr_gap = 1
    while true
        pₖ = B \ -∇fₖ
        # Run line search in direction `pₖ`
        αₖ, fₖ₊₁, ∇fₖ₊₁ = _line_search(f, fₖ, ∇fₖ, xₖ, pₖ, αₖ, evals)
        
        norm_value     = _norm(αₖ * pₖ)
        step           = norm_value / max(1.0, _norm(xₖ))

        
        if curr_bound !== nothing
            curr_gap = abs(curr_bound - fₖ₊₁)/abs(curr_bound + 1e-11)
        end
        
        if step < bfgs.ftol
            # Small steps! Probably at the edge of the feasible region.
            # Return the current iterate.
            return fₖ, xₖ
        elseif _norm(∇fₖ₊₁) < bfgs.gtol
            # Zero(ish) gradient. Return what must be a local maxima.
            # println("             local_imprv: zero gradient with gradient $(_norm(∇fₖ₊₁)) and evals: $(evals[])")
            return fₖ₊₁, xₖ + αₖ * pₖ
        elseif evals[] <= 0
            # We have evaluated the function too many times. Return our current
            # best.
            return fₖ₊₁, xₖ + αₖ * pₖ
        elseif curr_gap < bfgs.gaptol
            # termination due to gap
            return fₖ₊₁, xₖ + αₖ * pₖ
        elseif time_left !== nothing && time() - start_time > time_left
            # hit the time limit
            return fₖ₊₁, xₖ + αₖ * pₖ
        end
        # BFGS update.
        sₖ = αₖ * pₖ
        yₖ = ∇fₖ₊₁ - ∇fₖ
        # A slight tweak to normal BFGS: because we're dealing with non-smooth
        # problems, ||yₖ|| might be 0.0, i.e., we just moved along a facet from
        # from an interior point to a vertex, so the gradient stays the same.
        if _norm(yₖ) > 1e-12
            B .=
                B .+ (yₖ * yₖ') / (yₖ' * sₖ) -
                (B * sₖ * sₖ' * B') / (sₖ' * B * sₖ)
        end
        fₖ, ∇fₖ, xₖ = fₖ₊₁, ∇fₖ₊₁, xₖ + sₖ
    end
    
    
end

function _line_search(
    f::F,
    fₖ::Float64,
    ∇fₖ::Vector{Float64},
    x::Vector{Float64},
    p::Vector{Float64},
    α::Float64,
    evals::Ref{Int},
) where {F<:Function}
    
    while _norm(α * p) > 1e-6 * max(1.0, _norm(x))
        xₖ = x + α * p
        ret = f(xₖ)
        evals[] -= 1
        
        if ret === nothing
            α /= 2  # Infeasible. So take a smaller step
            continue
        end
        fₖ₊₁, ∇fₖ₊₁ = ret
        if p' * ∇fₖ₊₁ < 1e-6
            # Still a descent direction, so take a step.
            
            return α, fₖ₊₁, ∇fₖ₊₁
        elseif isapprox(fₖ + α * p' * ∇fₖ, fₖ₊₁; atol = 1e-8)
            # Step is onto a kink
            
            return α, fₖ₊₁, ∇fₖ₊₁
        end
        #  Step is an ascent, so use Newton's method to find the intersection
        α = (fₖ₊₁ - fₖ - p' * ∇fₖ₊₁ * α) / (p' * ∇fₖ - p' * ∇fₖ₊₁)
        
    end
    #termination due to tolerance
    return 0.0, fₖ, ∇fₖ
end

end