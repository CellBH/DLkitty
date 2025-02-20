

@testset "predict_kcat_dist" begin
    df = kcat_table_train_and_valid()
    ngram_len = 3
    all_ngrams = load_all_sequence_ngrams(ngram_len)

    model = DLkittyModel(; num_unique_ngrams=length(all_ngrams))
    trained_model = TrainedModel(model)  # actually untrained

    datum = first(Tables.namedtupleiterator(df))
    dist = predict_kcat_dist(trained_model, all_ngrams, datum)
    @test dist isa LogNormal
end


if !haskey(ENV, "CI")
    @testset "full train" begin
        df = kcat_table_train_and_valid()
        all_ngrams = load_all_sequence_ngrams(3)
        trained_model = train(df, all_ngrams; n_samples=3, n_epochs=2)
    end
end