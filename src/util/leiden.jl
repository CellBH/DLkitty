#!/usr/bin/env julia
using PyCall
using SparseArrays
using HDF5, H5Zzstd
ROOT = `git root` |> readchomp
@pyinclude "$ROOT/src/util/leiden.py"

"""
B is sparse integer matrix with dim (#nodes, #hyperedges).
- hyperedges: each hyperedge is a set of node indices
- nNodes: optionally specify the total number of nodes
"""
function hyperedges2B(hyperedges::Vector{Set{T}}, nNodes::Union{Int,Nothing}=nothing) where T<:Integer
    Is = [n for     h  in           hyperedges  for n in h]
    Js = [j for (j, h) in enumerate(hyperedges) for n in h]
    if nNodes === nothing
        sparse(Is, Js, 1)
    else
        sparse(Is, Js, 1, nNodes, length(hyperedges))
    end
end
function hyperedges2B(hyperedges::Matrix{T}, nNodes::Union{Int,Nothing}=nothing) where T<:Integer
    hyperedges2B(reps2hyperedges(hyperedges), nNodes)
end
function hyperedges2B(hyperedges::Vector{Vector{Vector{T}}}, nNodes::Union{Int,Nothing}=nothing) where T<:Integer
    hyperedges = [Set(vcat(vs...)) for vs in hyperedges]
    hyperedges2B(hyperedges, nNodes)
end

"""
Convert hyperedge format from matrix to list of sets.
"""
function reps2hyperedges(reps::Matrix{T})::Vector{Set{T}} where T<:Integer
    hyperedges = Set{Int}[]
    indices = reps[:,1]
    _reps = Int.(reps[:,2:end])
    for i in 1:maximum(indices)
        push!(hyperedges, Set(_reps[indices .== i, :]))
    end
    hyperedges
end

function CliqueExpansion(mat)
    N = size(mat, 1)
    ex = zeros(N, N)
    for he in eachcol(mat)
        Is = findall(he .!= 0)
        for i in Is
            for j in Is
                if i != j
                    ex[i,j] = (he[i] + he[j]) / 2
                end
            end
        end
    end
    ex
end

"""
Supports hyperedges in all used formats, i.e. matrix, list of sets, and list of list of list of int.
E.g. reps (cycle representatives) are hyperedges stored in HDF5 files in matrix format,
where the first column is hyperedge identifier, and the following are vertex indices.
"""
function leiden(hyperedges, bars::AbstractMatrix, n::Int)
    B = hyperedges2B(hyperedges, n)
    persistences = bars[:,end]
    H = B .* persistences'
    H |> CliqueExpansion |> py"leiden"
end

function leiden(h5::HDF5.File, dim::Int)
    leiden(h5["reps$dim"][:,:], h5["bars$dim"][:,:], size(h5["Cas"][:,:],1))
end


