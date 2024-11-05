using DLkitty
using Test
using Distributions
using Random
using Lux
using Optimisers
using Zygote

@testset "DLkitty.jl" begin
    @testset "neural_net.jl" include("neural_net.jl")
end
