@testset "ngrams" begin
    @test DLkitty.ngrams("ABCD", 3) == ["-AB", "ABC", "BCD", "CD="]
end

@testset "load" begin
    ngrams = DLkitty.all_sequence_ngrams(kcat_table_train(), 3)
    @test length(ngrams) > 8_000
    @test all(x==DLkitty.UNKNOWN_NGRAM || length(x)==3 for x in ngrams)
end
