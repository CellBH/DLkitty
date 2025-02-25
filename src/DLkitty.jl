module DLkitty
using DataDeps
using CSV
using DataFrames
using JSONTables
using Distributions
using Lux
using LuxCore
using Lux.Training
using Optimisers
using Random
using Statistics
using OneHotArrays
using PythonCall
using GNNGraphs
using Tables
using ChainRulesCore
using GNNLux
using Printf
using Functors

export kcat_table_train_and_valid, is_usable, resample, load_all_sequence_ngrams
export mol_from_smiles, gnn_graph
export TrainedModel, DLkittyModel, predict_kcat_dist, train, load

include("data.jl")
include("ngrams.jl")
include("neural_net_output.jl")
include("neural_net_model.jl")
include("resampling.jl")
include("substrate.jl")
include("trained_model.jl")
include("execute.jl")
include("regularization.jl")

function __init__()
    init_data()
end

end
