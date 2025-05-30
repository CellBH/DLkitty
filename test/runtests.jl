using DLkitty
using Test
using Random
using DataFrames
using Distributions
using Random
using Lux
using Optimisers
using Zygote
using PythonCall
using Statistics
using Tables
using Graphs
using OneHotArrays
using Statistics
using GNNLux
using Lux.Training
using Functors

@testset "DLkitty.jl" begin
    @testset "standardizer.jl" include("standardizer.jl")
    @testset "data.jl" include("data.jl")
    @testset "ngrams.jl" include("ngrams.jl")
    @testset "neural_net_output.jl" include("neural_net_output.jl")
    @testset "neural_net_model.jl" include("neural_net_model.jl")
    @testset "resampling.jl" include("resampling.jl")
    @testset "substrate.jl" include("substrate.jl")
    @testset "execute.jl" include("execute.jl")
    @testset "regularization.jl" include("regularization.jl")
end
