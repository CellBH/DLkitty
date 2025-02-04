
function AttentionCNN(num_unique_ngrams; window=11, hdim=20)
    return @compact(;
        embed=Dense(num_unique_ngrams => hdim),
        convs=ntuple(3) do _
            Conv(
                (2*window + 1, 2*window + 1), (1 => 1), relu;
                pad=window, cross_correlation=true
            )
        end,
        attention=Dense(hdim=>hdim, relu),
    ) do (substrate_vector, ngrams_onehot_matrix)
        @assert eltype(ngrams_onehot_matrix) == Bool
        x1 =  embed(ngrams_onehot_matrix)
        x1_wide = reshape(x1, Val{4}())
        xs_wide = foldl(|>, convs; init=x1_wide)
        xs = dropdims(xs_wide, dims=(3, 4))
        h = attention(substrate_vector)
        hs = attention(xs)
        weights = tanh.(hs'*h)
        ys = hs .* weights'
        @return dropdims(mean(ys, dims=2), dims=2)
    end
end


const MAX_ATOMIC_NUM = 100  # plutonium is 94, anything much larger we are not worrying about ever
const BOND_TYPES = 22  # This is determined by the BondType enum in RDkit

# TODO: rewrite as a LuxCore.AbstractLuxContainerLayer
struct SubstrateGNN <: LuxCore.AbstractLuxLayer
    hdim::Int
end

function components(m::SubstrateGNN)
    return (;            
        atomic_num_embed = Embedding(MAX_ATOMIC_NUM=>m.hdim),
        bond_embedding = Embedding(BOND_TYPES=>m.hdim),
        input_net = CGConv((m.hdim,m.hdim) => m.hdim, relu; residual=false),
        output_net = GNNChain(            
            CGConv(m.hdim => m.hdim, relu; residual=true),
            CGConv(m.hdim => m.hdim, relu; residual=true),
            x -> mean(x, dims=2),  # combine all node data
        )
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SubstrateGNN)
    return Lux.fmap(x->LuxCore.initialparameters(rng, x), components(l))
end

function LuxCore.initialstates(rng::AbstractRNG, l::SubstrateGNN)
    return Lux.fmap(x->LuxCore.initialstates(rng, x), components(l))
end

function (l::SubstrateGNN)(g::GNNGraph, ps, st)
    c = @ignore_derivatives components(l)
    xn, _ = c.atomic_num_embed(g.ndata.atomic_num, ps.atomic_num_embed, st.atomic_num_embed)
    xe, _ = c.bond_embedding(g.edata.bond_type, ps.bond_embedding, st.bond_embedding)
    h, _ = c.input_net(g, xn, xe, ps.input_net, st.input_net)
    y, _ = c.output_net(g, h, ps.output_net, st.output_net)
    return y, st
end

struct DLkittyModel{S,P,M,O} <: LuxCore.AbstractLuxContainerLayer{(:substrate_net, :protein_net, :merge_net, :output_layer)}
    # TODO add other fields we need for preprocessing here?
    substrate_net::S
    protein_net::P
    merge_net::M
    output_layer::O
end

function DLkittyModel(; hdim=20, num_unique_ngrams)
    return DLkittyModel(
        SubstrateGNN(hdim),
        AttentionCNN(num_unique_ngrams; hdim),
        Dense((2*hdim + 2)=>hdim, relu),  # TODO: this should maybe be deeper (and others shallower?)
        DistOutputLayer{LogNormal}(hdim),
    )
end

function (m::DLkittyModel)((substrate_graphs, protein_1hots_seqs, temperature, ph), ps, st)
    # This sum is a DeepSet operation
    substrate_h = sum(substrate_graphs) do g
        # drop state (its empty anyway)
        first(m.substrate_net(g, ps.substrate_net, st.substrate_net))
    end

    # This sum is also a DeepSet operation
    protein_h = sum(protein_1hots_seqs) do seq
        # drop state (its empty anyway)
        first(m.protein_net((substrate_h, seq), ps.protein_net, st.protein_net))
    end

    # TODO: other misc features, than tempurature and ph
    # TODO consider just putting in the number of substrates and proteins as a features (esp as sume subtrates are Missing)
    # TODO consider putting in just the total length of the proteins
    h2, _ = m.merge_net([substrate_h; protein_h; temperature; ph], ps.merge_net, st.merge_net)
    y, _ = m.output_layer(h2, ps.output_layer, st.output_layer)
    return y, st
end
