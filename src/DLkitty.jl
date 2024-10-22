module DLkitty
using DataDeps
using CSV
using DataFrames

include("data.jl")

function __init__()
    init_data()
end

end
