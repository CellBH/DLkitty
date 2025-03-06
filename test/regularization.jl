@testset "regularization.jl" begin
    preprocessor = load_preprocessor()
    model = DLkittyModel(preprocessor)
    trained_model = TrainedModel(model)  # actually untrained

    unit_ps = Functors.fmap(x->0.0.*x .+ 1.0, trained_model.ps)
    @test DLkitty.l2_term(unit_ps) == 8467  # note when sturcture changes this will also need to be changed
end