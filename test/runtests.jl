using DLkitty
using Test
using Random
using DataFrames
using Distributions
using Random
using Lux
using Optimisers
using Zygote

@testset "DLkitty.jl" begin
    @testset "data.jl" include("data.jl")
    @testset "neural_net.jl" include("neural_net.jl")
    @testset "resampling.jl" include("resampling.jl")
end
