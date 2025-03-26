l2_term(x::AbstractArray{<:Number}) = sum(abs2, x)
function l2_term(ps::NamedTuple)
    total = 0.0
    for (fname, fval) in pairs(ps)
        if fname ∈ (:atomic_num_embed, :bond_embedding, :fingerprint_embedding, :embed, :bias)
            continue
        else
            total += l2_term(fval)
        end
    end
    return total
end


struct L2RegLoss{F<:Lux.AbstractLossFunction} <: Lux.AbstractLossFunction
    base_loss::F
    l2_coefficient::Float64
end

function (self::L2RegLoss)(model::Lux.AbstractLuxLayer, ps, st, (x, y))
    base, new_st, _ = self.base_loss(model, ps, st, (x, y))
    reg = self.l2_coefficient * l2_term(ps)
    #@show base reg
    # uncomment above during training can tune to aim for `reg`
    # to be a few orders of magnitude below `base` training loss
    return (base + reg), new_st, (;)
end