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
using Graphs: Graphs
using LinearAlgebra
using ProgressMeter
using Serialization
using ThreadsX
using Logging
using TensorBoardLogger
using ParameterHandling
using OhMyThreads

export kcat_table_train_and_valid, kcat_table_train, kcat_table_valid
export is_usable, is_complete, resample, load_all_sequence_ngrams, prep_data
export mol_from_smiles, gnn_graph
export TrainedModel, DLkittyModel, Preprocessor, load_preprocessor, predict_kcat_dist, train, load

include("standardizer.jl")
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