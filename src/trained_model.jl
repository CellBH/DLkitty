"Helper to bundle a Lux Model with its parameters and state"
struct TrainedModel{M, P, S}
    model::M
    ps::P
    st::S
end

function TrainedModel(
    rng::AbstractRNG,
    model::M,
    ps::P = LuxCore.initialparameters(rng, model),
    st::S = LuxCore.initialstates(rng, model),
    ) where {M, P, S}
    return TrainedModel{M, P, S}(model, ps, st)
end
TrainedModel(args...; rng=Xoshiro(0)) = TrainedModel(rng, args...)