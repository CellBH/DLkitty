

@testset "distribution_output" begin
    # Simplified model that just uses our output layer
    # So we can make sure that works right
    model = DLkitty.DistOutputLayer{LogNormal}(3)
    
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model) |> Lux.f64
    input_data = randn(rng, 3, 100)

    output_data, _ = Lux.apply(model, input_data, ps, st)
    @test output_data isa AbstractVector{<:LogNormal}
    @test length(output_data) == 100
end

@testset "training from sample" begin
    model = DLkitty.DistOutputLayer{LogNormal}(1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model) |> Lux.f64


    n_samples = 2_000
    target_dist = LogNormal(0.1, 0.5)
    training_inputs = randn(rng, 1, n_samples)
    training_outputs = rand(rng, target_dist, n_samples)
    @assert isapprox(mean(training_outputs), mean(target_dist), atol=0.05)
    @assert isapprox(std(training_outputs), std(target_dist), atol=0.05)
    @assert isapprox(median(training_outputs), median(target_dist), atol=0.1)

    
    train_state = Lux.Training.TrainState(model, ps, st, Adam())
    losses = Float64[]
    for _ in 1:10
        gs, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), DLkitty.DistributionLoss(), (training_inputs, training_outputs), train_state
        )
        push!(losses, loss)
    end

    # check our losess are roughly monotonic, never overshooting to be much worse than before
    @test all(diff(losses)./losses[1:end-1] .< 3)
end


@testset "training from mean + stddev" begin
    model = DLkitty.DistOutputLayer{LogNormal}(1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model) |> Lux.f64


    n_samples = 2_000
    target_dist = LogNormal(0.1, 0.5)
    training_inputs = randn(rng, 1, n_samples)
    training_outputs = map(1:n_samples) do _
        data = rand(rng, target_dist, 100)
        return (; mean=mean(data), std=std(data))
    end

    
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001))
    losses = Float64[]
    for _ in 1:10
        gs, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), DLkitty.DistributionLoss(), (training_inputs, training_outputs), train_state
        )
        push!(losses, loss)
    end

    # check our losess are roughly monotonic, never overshooting to be much worse than before
    @test all(diff(losses)./losses[1:end-1] .< 1)
end


