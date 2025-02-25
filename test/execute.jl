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

@testset "evaluation" begin
    all_ngrams = load_all_sequence_ngrams(3)
    model = DLkittyModel(; num_unique_ngrams=length(all_ngrams))
    trained_model = TrainedModel(model)  # actually untrained

    eval_df = filter(is_complete, kcat_table_valid())[1:100, :]
    eval_df.predicted_kcat_dists = map(Tables.namedtupleiterator(eval_df)) do datum
        predict_kcat_dist(trained_model, all_ngrams, datum)
    end

    eval_df.loglikelyhoods = loglikelihood.(eval_df.predicted_kcat_dists, eval_df.Value)
    log_likelyhood_of_eval_set = sum(eval_df.loglikelyhoods)  # an extremely small number
    @test isfinite(log_likelyhood_of_eval_set)

    eval_df.ae_to_mode = abs.(mode.(eval_df.predicted_kcat_dists) .- eval_df.Value)
    mean_ae_to_mode = mean(eval_df.ae_to_mode)
    @test isfinite(mean_ae_to_mode)
    @test mean_ae_to_mode > 0
end


@testset "train" begin
    rows_to_use = 100
    df = kcat_table_train_and_valid()
    df = df[shuffle(1:nrow(df))[1:rows_to_use], :]
    all_ngrams = load_all_sequence_ngrams(3)
    trained_model = train(df, all_ngrams; n_samples=2, n_epochs=1)
end