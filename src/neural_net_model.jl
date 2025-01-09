
# TODO: consider using Kmers.jl for this
function ngrams(sequence, ngram_len)
    @assert ngram_len > 1
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


function AttentionCNN(num_unique_ngrams; window=11, dim=20)
    return @compact(;
        embed=Dense(num_unique_ngrams => dim),
        convs=ntuple(3) do _
            Conv(
                (2*window + 1, 2*window + 1), (1 => 1), relu;
                pad=window, cross_correlation=true
            )
        
        end,
        attention=Dense(dim=>dim, relu),
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
