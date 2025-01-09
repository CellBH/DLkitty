using DLkitty
using Test
using Random
using DataFrames
using Distributions
using Random
using Lux
using Optimisers
using Zygote
using OneHotArrays

@testset "DLkitty.jl" begin
    @testset "data.jl" include("data.jl")
    @testset "neural_net_output.jl" include("neural_net_output.jl")
    @testset "neural_net_model.jl" include("neural_net_model.jl")
    @testset "resampling.jl" include("resampling.jl")
end
