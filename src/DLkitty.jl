module DLkitty
using DataDeps
using CSV
using DataFrames
using JSONTables
using Distributions
using Lux
using LuxCore
using Random
using Statistics
using OneHotArrays

export kcat_table_train_and_valid, is_usable, resample

include("data.jl")
include("neural_net_output.jl")
include("neural_net_model.jl")
include("resampling.jl")

function __init__()
    init_data()
end

end
