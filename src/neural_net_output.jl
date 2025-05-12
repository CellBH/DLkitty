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
this function constructions one so we can delegate to it.
It entirely optimizes away.
"""
function _backing_Dense(l::DistOutputLayer{D}) where D
    out_dims = n_dist_parameters(D)
    return Dense(l.in_dims=>out_dims)
end

n_dist_parameters(::Type{D}) where D = length(parameter_functions(D))

"ensures the parameter is a legal value"
function parameter_functions end
parameter_functions(::Type{<:Normal}) = (identity, abs)
parameter_functions(::Type{<:LogNormal}) = (identity, abs∘log∘abs)
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

@kwdef struct DistributionLoss <: Lux.AbstractLossFunction
    sample_weight::Float64 = 1.0
    mean_std_weight::Float64 = 2.0  # 2x as much information as a single point
end

(loss::DistributionLoss)(ŷs, ys) = mean(((ŷ, y),)->loss1(loss, ŷ, y), zip(ŷs, ys))

# for samples

loss1(d::DistributionLoss, dist::Distribution, y::Number) = d.sample_weight * abs2(mean(dist) - mean(y))
# for mean + std-dev
function loss1(d::DistributionLoss, dist::Distribution, y::NamedTuple{(:mean, :std)})
    return d.mean_std_weight * (abs2(mean(dist) - y.mean) + abs2(std(dist) - y.std))
end

# for either: (redispatch)
function loss1(d::DistributionLoss, dist::Distribution, datum)
    if ismissing(datum.StandardDeviation)
        return loss1(d, dist, datum.Value)
    else
        return loss1(d, dist, (;mean=datum.Value, std=datum.StandardDeviation))
    end
end

using ChainRulesCore: RuleConfig, HasReverseMode, rrule_via_ad, ProjectTo, NoTangent, unthunk

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.sum), f, xs::AbstractArray)
    fx_and_pullbacks = ThreadsX.map(x->rrule_via_ad(config, f, x), xs)
    y = ThreadsX.sum(first, fx_and_pullbacks)

    pullbacks = ThreadsX.map(last, fx_and_pullbacks)

    project = ProjectTo(xs)

    function sum_pullback(ȳ)
        call(f, x) = f(x)
        # if dims is :, then need only left-handed only broadcast
        # broadcast_ȳ = dims isa Colon  ? (ȳ,) : ȳ
        broadcast_ȳ = ȳ
        f̄_and_x̄s = ThreadsX.map(f->f(ȳ), pullbacks)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            ThreadsX.sum(first, f̄_and_x̄s)
        end
        x̄s = ThreadsX.map(unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return NoTangent(), f̄, project(x̄s)
    end
    return y, sum_pullback
end

struct PointEstimateLoss <: Lux.AbstractLossFunction
end

function (self::PointEstimateLoss)(model::Lux.AbstractLuxLayer, ps, st, data::AbstractArray)
    base = ThreadsX.sum(1:length(data)) do i
        x, y = data[i]
        y_pred, _ = model(x, ps, st)
        Lux.LossFunctionImpl.l2_distance_loss(y_pred, y)
    end
    base /= length(data)
    return base, st, (;)
end