@testset "ngrams" begin
    @test DLkitty.ngrams("ABCD", 3) == ["-AB", "ABC", "BCD", "CD="]
end


@testset "Protien Sequence Input" begin
    df = kcat_table_train_and_valid()
    ngram_len = 3
    all_ngrams = DLkitty.all_sequence_ngrams(df, ngram_len)

    # These are all single protien cases:
    seqs_batch1 = only.(df.ProteinSequences[1:5])
    ngrams_seqs_batch1 = DLkitty.ngrams.(seqs_batch1, ngram_len)
    onehots_seqs_batch1 = onehotbatch.(ngrams_seqs_batch1, (all_ngrams,), DLkitty.UNKNOWN_NGRAM)
    @test onehots_seqs_batch1 isa Vector{<:OneHotMatrix}
    @test onecold.(onehots_seqs_batch1, (all_ngrams,)) == ngrams_seqs_batch1


    num_unique_ngrams = length(all_ngrams)
    attention_cnn = DLkitty.AttentionCNN(num_unique_ngrams)

    substrate_vector = randn(20)
    ps, st = Lux.setup(Xoshiro(0), attention_cnn)
    for onehots_seq in onehots_seqs_batch1
        z, _= attention_cnn((substrate_vector, onehots_seq), ps, st)
        @test size(z)== (20,)
    end

end
