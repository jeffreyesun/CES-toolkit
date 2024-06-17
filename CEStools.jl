"""
A toolkit for working with constant elasticity of substitution production functions.

Released under MIT License

Copyright (c) 2024 Jeffrey Sun (jeffreyesun@gmail.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

struct ScalarCESLayer{T<:Tuple, P}
    α::T
    σ::P
end

struct VectorCESLayer{P}
    α::Matrix{P}
    σ::P
end

############
# Solution #
############

# CES Layer #
#-----------#
# Prices #
# Scalar Layer
function solveCES_pout_scalar(p_in::Tuple, weights::Tuple, eos)
    p_out = mapreduce((a,b)->a.+b, zip(weights,p_in)) do (α,p_in)
        @. α*p_in^(1-eos)
    end
    return @. p_out^(1/(1-eos))
end
solveCES_pout(p_in, layer::ScalarCESLayer) = solveCES_pout_scalar(p_in, layer.α, layer.σ)

# Vector Layer
function solveCES_pout_vector(p_in, weights, eos)
    p_in_pow = p_in.^(1-eos)
    p_out = weights * p_in_pow
    #mul!(p_out, weights, p_in_pow)
    return p_out .^ (1/(1-eos))
end
solveCES_pout(p_in, layer::VectorCESLayer) = solveCES_pout_vector(p_in, layer.α, layer.σ)

# Stacked Layer
function solveCES_pout(p_in, layers::Vector{<:VectorCESLayer})
    n_markets = length(layers)
    res = zeros(eltype(p_in), (size(first(layers).α, 2), n_markets))
    Threads.@threads for i=1:n_markets
        res[:,i] .= @views solveCES_pout(p_in[:,i], layers[i])
    end
    return res
end

# Expenditures #
# Scalar Layer
function solveCES_revenue_scalar(budget, p_in::Tuple, p_out, weights::Tuple, eos)
    return Tuple((@. α*budget*(p_in./p_out)^(1-eos)) for (α,p_in)=zip(weights,p_in))
end
solveCES_revenue(budget, p_in, p_out, layer::ScalarCESLayer) = solveCES_revenue_scalar(budget, p_in, p_out, layer.α, layer.σ)

# Vector Layer
function solveCES_revenue_vector(budget, p_in, p_out, weights, eos)
    y_out_div = @. budget/p_out^(1-eos)
    y_in = weights'*y_out_div
    return @. y_in *= p_in^(1-eos)
end
solveCES_revenue(budget, p_in, p_out, layer::VectorCESLayer) = solveCES_revenue_vector(budget, p_in, p_out, layer.α, layer.σ)

# Stacked Layer
function solveCES_revenue(budget, p_in, p_out, layers::Vector{<:VectorCESLayer})
    n_markets = length(layers)
    res = zeros(eltype(p_in), (size(first(layers).α, 1), n_markets))
    Threads.@threads for i=1:n_markets
        res[:,i] .= @views solveCES_revenue(budget[:,i], p_in[:,i], p_out[:,i], layers[i])
    end
    return res
end
