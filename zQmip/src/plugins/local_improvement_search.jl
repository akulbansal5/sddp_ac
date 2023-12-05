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

###
### BFGS
###
struct BFGS <: AbstractSearchMethod
    evaluation_limit::Int
    ftol::Float64
    gtol::Float64
    function BFGS(evaluation_limit::Int, ftol::Float64 = 1e-3, gtol::Float64 = 1e-6)
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
function minimize(f::F, bfgs::BFGS, x₀::Vector{Float64}) where {F<:Function}
    # Initial estimte for the Hessian matrix in BFGS
    B = zeros(length(x₀), length(x₀))


    for i in 1:size(B, 1)
        B[i, i] = 1.0
    end

    # We assume that the initial iterate is feasible
    xₖ = x₀
    fₖ, ∇fₖ = f(xₖ)::Tuple{Float64,Vector{Float64}}                             #the gradient comes through f(x_k)
    # println("               local_improv: gradient of the function $(∇fₖ)")    
    
    # Initial step-length
    αₖ = 1.0
    # Evaluation counter
    # evals = Ref(0)
    evals = Ref(bfgs.evaluation_limit)
    # println("               local_imprv: allowed number of evals: $(bfgs.evaluation_limit)")
    while true
        # Search direction. We could be clever here and maintain B⁻¹, but we're
        # only ever going to be solving this for very small |x| << 100 problems,
        # so taking the linear solve every time is okay. (The MIP solve is much
        # more of a bottleneck.)
        pₖ = B \ -∇fₖ
        # Run line search in direction `pₖ`
        αₖ, fₖ₊₁, ∇fₖ₊₁ = _line_search(f, fₖ, ∇fₖ, xₖ, pₖ, αₖ, evals)
        # println("                           local imprv iter: $(evals[]), total: $(evals)")
        norm_value     = _norm(αₖ * pₖ)
        step           = norm_value / max(1.0, _norm(xₖ)) 
        if step < bfgs.ftol
            # Small steps! Probably at the edge of the feasible region.
            # Return the current iterate.
            #
            # Note that "1e-3" isn't thaaaat small. But we hit a very annoying
            # feature of solvers: their feasibility checks are only approximate.
            # This tolerance is needed to pass the `test_kelleys_ip_xxx` tests.
            # Decreasing the tolerance leads to a _worse_ estimate for the dual,
            # because we abuse the solvers feasibility tolerance, and end up
            # returning a solution that is on the edge of numerical dual
            # feasibility.
            println("             local_imprv: at edge with # of lg dual evals: $(evals[]), ftol: $(bfgs.ftol), step: $(step), alpha_k: $(αₖ), norm_value: $(norm_value)")
            return fₖ, xₖ
        elseif _norm(∇fₖ₊₁) < bfgs.gtol
            # Zero(ish) gradient. Return what must be a local maxima.
            println("             local_imprv: zero gradient with gradient $(_norm(∇fₖ₊₁)) and evals: $(evals[])")
            return fₖ₊₁, xₖ + αₖ * pₖ
        elseif evals[] <= 0
            # We have evaluated the function too many times. Return our current
            # best.
            println("             local_imprv: termination with number of lg dual evals: $(evals[])")
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
    while _norm(α * p) > 1e-3 * max(1.0, _norm(x))
        xₖ = x + α * p
        ret = f(xₖ)
        evals[] -= 1
        # println("             line search: $(evals[]), alpha value: $(α), ret: $(ret), p_k: $(p)")
        if ret === nothing
            α /= 2  # Infeasible. So take a smaller step
            continue
        end
        fₖ₊₁, ∇fₖ₊₁ = ret
        if p' * ∇fₖ₊₁ < 1e-6
            # Still a descent direction, so take a step.
            # println("           line search: termination but still a descent direction")
            return α, fₖ₊₁, ∇fₖ₊₁
        elseif isapprox(fₖ + α * p' * ∇fₖ, fₖ₊₁; atol = 1e-8)
            # Step is onto a kink
            # println("           line search: termination by entering into a kink")
            return α, fₖ₊₁, ∇fₖ₊₁
        end
        #  Step is an ascent, so use Newton's method to find the intersection
        α = (fₖ₊₁ - fₖ - p' * ∇fₖ₊₁ * α) / (p' * ∇fₖ - p' * ∇fₖ₊₁)
        
    end
    return 0.0, fₖ, ∇fₖ
end

end