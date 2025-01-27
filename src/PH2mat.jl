#!/usr/bin/env julia
"""
USAGE: PH2.jl INFILES.h5... OUTDIR/
USAGE: PH2.jl INDIR/ OUTDIR/
Writes .mat
"""

using HDF5, H5Zzstd
filters = H5Zzstd.ZstdFilter()
using SparseArrays
using DelimitedFiles
using GZip

INFILES..., OUTDIR = ARGS
if length(INFILES) == 1 && isdir(only(INFILES))
    INFILES = readdir(only(INFILES); join=true)
end
@assert isdir(OUTDIR) || !isfile(OUTDIR) "No OUTDIR provided: $OUTDIR"
mkpath(OUTDIR)

"""
Differs from splitext by stripping all extensions.
"""
function basenameroot(path)
    replace(basename(path), r"\..*" => "")
end

"""
Write compressed mat.
Space delimiter by default.
"""
function writegz(path, A; delim=' ')
    GZip.open(path, "w") do io
        writedlm(io, A, delim)
    end
end

for infname in INFILES
    println(infname)
    h5open(infname) do fid
        name = basenameroot(infname)
        outdir = joinpath(OUTDIR, name)
        mkpath(outdir)

        n = attrs(fid)["n"]

        for dim in 1:2
            e = fid["reps$dim"][:,1]
            v = fid["reps$dim"][:,2:end]
            # hypergraph incidence matrix
            Is = vcat((e for _ in 1:size(v,2))...)
            Js = vcat(eachcol(v)...)
            BT = sparse(Is, Js, true, maximum(e), n)

            persistence = fid["bars$dim"][:,2]
            centrality = fid["Cas"][:,4+dim]

            writegz(joinpath(outdir, "BT$dim.ssv.gz"), Int.(BT))
            writegz(joinpath(outdir, "persistence$dim.ssv.gz"), persistence)
            writegz(joinpath(outdir, "centrality$dim.ssv.gz"), centrality)
        end
    end
end
