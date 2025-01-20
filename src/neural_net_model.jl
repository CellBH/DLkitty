
# TODO: consider using Kmers.jl for this
function ngrams(sequence, ngram_len)
    @assert ngram_len >= 1
    sequence = "-" * sequence * "="
    return map(1 : length(sequence) - ngram_len + 1) do start
        sequence[start : start + ngram_len - 1]
    end
end

const UNKNOWN_NGRAM = "@"
function all_sequence_ngrams(df, ngram_len)
    sequences = identity.(skipmissing(Iterators.flatten(skipmissing(df.ProteinSequences))))
    seen = mapreduce(s->ngrams(s, ngram_len), union!, sequences; init=Set{String}())
    return push!(collect(seen), UNKNOWN_NGRAM)
end


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


function DLkittyModel(; hdim=20, num_unique_ngrams)
    return @compact(;
        substrate_net = SubstrateGNN(hdim),
        protein_net = AttentionCNN(num_unique_ngrams; hdim),
        merge_net = Dense((2*hdim + 2)=>hdim, relu),  # TODO: this should maybe be deeper (and others shallower?)
        output_layer = DistOutputLayer{LogNormal}(hdim),
    ) do (substrate_graphs, protein_1hots_seqs, temperature, ph)
        # This sum is a DeepSet operation
        @show len(substrate_graphs)
        substrate_h = sum(substrate_graphs) do g
            substrate_net(g)
        end
        @show size(substrate_h)

        # This sum is also a DeepSet operation
        protein_h = sum(protein_1hots_seqs) do seq
            protein_net((substrate_h, seq))
        end
        @show size(protein_h)

        # TODO: other misc features, than tempurature and ph
        # TODO consider just putting in the number of substrates and proteins as a features (esp as sume subtrates are Missing)
        # TODO consider putting in just the total length of the proteins
        h2 = merge_net([substrate_h; protein_h; temperature; ph])
        y = output_layer(h2)
        @return y
    end
end


function  predict_kcat_dist((;model, ps, st), all_ngrams, datum)
    ngram_len = length(first(all_ngrams))

    substrate_graphs = map(skipmissing(datum.SubstrateSMILES)) do smiles
        mol = mol_from_smiles(smiles)   
        gnn_graph(mol)
    end

    protein_1hots_seqs = map(skipmissing(datum.ProteinSequences)) do seq
        onehotbatch(ngrams(seq, ngram_len), all_ngrams, UNKNOWN_NGRAM)
    end

    # TODO: strongly consider z-normalzing temperature and pH
    temperature = datum.Temperature
    ph = datum.pH
    y, _ = model((substrate_graphs, protein_1hots_seqs, temperature, ph), ps, st)
    return y
end