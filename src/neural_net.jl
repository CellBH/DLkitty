"""
    DistOutputLayer{D <: Distribution{Univariate, Continuous}}

A Lux layer which produces an output that is a instance of the distribution `D`
It takes 1 argument: the size of the layer below.
(as well as the type parameter for the distribution type)

It does a learnt dense projection from the layer below to the parameters of the distribution.
"""
struct DistOutputLayer{D <: ContinuousDistribution} <: LuxCore.AbstractLuxLayer
    in_dims::Int
end

"""
first step and all parameters/state of DistOutputLayer is as per Dense
this function constructions one so we can delgate to it.
It entirely optimizes away.
"""
function _backing_Dense(l::DistOutputLayer{D}) where D
    out_dims = n_dist_parameters(D)
    return Dense(l.in_dims=>out_dims)
end

n_dist_parameters(::Type{D}) where D = length(parameter_functions(D))

"ensures the parameter is a legal value"
function parameter_functions end
parameter_functions(::Type{<:Union{Normal,LogNormal}}) = (identity, abs)
parameter_functions(::Type{<:Truncated{D}}) where D = parameter_functions(D)
parameter_functions(::Type{<:Gamma}) = (abs, abs)

function LuxCore.initialparameters(rng::AbstractRNG, l::DistOutputLayer{D}) where D
    return LuxCore.initialparameters(rng, _backing_Dense(l))
end

function LuxCore.initialstates(rng::AbstractRNG, l::DistOutputLayer)
    return LuxCore.initialstates(rng, _backing_Dense(l))
end

LuxCore.parameterlength(l::DistOutputLayer) = LuxCore.parameterlength(_backing_Dense(l))
LuxCore.statelength(l::DistOutputLayer) = LuxCore.statelength(_backing_Dense(l))

function (l::DistOutputLayer{D})(x::AbstractMatrix, ps, st) where D
    y, st = _backing_Dense(l)(x, ps, st)
    act_funs = parameter_functions(D)
    ps = broadcast.(act_funs, y)  #It's a broadcasted broadcast
    dists = D.(eachrow(ps)...)
    return dists, st
end


struct DistributionLoss <: Lux.AbstractLossFunction
    # TODO: put relative weights for different loss types in here
end

(loss::DistributionLoss)(ŷs, ys) = mean(((ŷ, y),)->loss1(loss, ŷ, y), zip(ŷs, ys))

# for samples
loss1(::DistributionLoss, dist::Distribution, y::Number) = -loglikelihood(dist, y)

# for mean + std-dev
function loss1(::DistributionLoss, dist::Distribution, y::NamedTuple{(:mean, :std)})
    return abs2(mean(dist) - y.mean) + abs2(std(dist) - y.std)
end