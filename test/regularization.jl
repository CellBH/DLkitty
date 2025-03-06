@testset "regularization.jl" begin
    preprocessor = load_preprocessor()
    model = DLkittyModel(preprocessor)
    trained_model = TrainedModel(model)  # actually untrained

    unit_ps = Functors.fmap(x->0.0.*x .+ 1.0, trained_model.ps)
    num_l2_terms = DLkitty.l2_term(unit_ps)
    @test 0 < num_l2_terms

    total_terms = sum(Iterators.flatten(Functors.fleaves(unit_ps)))
    @test num_l2_terms < 0.05 * total_terms  # Most total terms are embeddings which should not be L2 reg'ed
    @test isinteger(num_l2_terms)

    total_basic_weights = 0
    Functors.fmap_with_path(trained_model.ps) do path, x
        total_basic_weights += (
            path[end] == :weight && 
            (
                length(path)>=2 && path[end-1] ∈ (:dense_f, :dense_s) ||
                length(path)>=3 && path[end-2] == :convs
            )
        ) * length(x)
    end
    @test total_basic_weights > 0.75 * num_l2_terms  # most L2 terms should be basic weights
end