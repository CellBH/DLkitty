
l2_term(x::AbstractArray{<:Number}) = sum(abs2, x)
function l2_term(ps::NamedTuple)
    # We could make this more fancy if we wanted to exclude the embedding or attention layers
    return sum(l2_term, Functors.fleaves(ps))
end


struct L2RegLoss{F<:Lux.AbstractLossFunction} <: Lux.AbstractLossFunction
    base_loss::F
    l2_coefficient::Float64
end

function (self::L2RegLoss)(model::Lux.AbstractLuxLayer, ps, st, (x, y))
    base, new_st, _ = self.base_loss(model, ps, st, (x, y))
    reg = self.l2_coefficient * l2_term(ps)
    # @show base reg
    # uncomment above during training can tune to aim for `reg`
    # to be a few orders of magnitude below `base` training loss
    return (base + reg), new_st, (;)
end