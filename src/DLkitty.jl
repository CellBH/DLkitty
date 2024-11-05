module DLkitty
using DataDeps
using CSV
using DataFrames
using Distributions
using Lux
using LuxCore
using Random

include("data.jl")
include("neural_net.jl")

function __init__()
    init_data()
end

end
