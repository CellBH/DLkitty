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

function save_all_sequence_ngrams(df, ngram_len)
    seqs = all_sequence_ngrams(df, ngram_len)
    open(joinpath(dirname(@__DIR__), "data", "all_ngrams_$(ngram_len).txt"), "w") do fh
        for seq in seqs
            println(fh, seq)
        end
    end
end

function load_all_sequence_ngrams(ngram_len)
    open(collect∘eachline, joinpath(dirname(@__DIR__), "data", "all_ngrams_$(ngram_len).txt"))
end