#!/usr/bin/env julia
using ArgParse
using DataFrames, CSV
using HDF5, H5Zzstd
using PlotlyJS
using Colors, ColorSchemes
using Printf
using Graphs
using SimpleWeightedGraphs
using Chain
using StatsBase: countmap
ROOT = `git root` |> readchomp
include("$ROOT/src/util/leiden.jl")

args_settings = ArgParseSettings(autofix_names=true)
@add_arg_table args_settings begin
    "--no-lines", "-L"
    help = "No lines between points"
    action = :store_true
    "--out", "-o"
    help = "Output directory for HTML files"
    default = "."
    "infiles"
    help = "One or more HDF5 input files"
    nargs = '+'
    arg_type = String
    required = true
end
args = NamedTuple(parse_args(args_settings; as_symbols=true))

function Graphs.SimpleGraph(edges::Matrix{Int32})
    edges |> eachrow .|> Tuple .|> Graphs.SimpleEdge |> SimpleGraph
end

function Graphs.cycle_basis(edges::Matrix{Int32})::Vector{Vector{Int32}}
    cycles = edges |> SimpleGraph |> cycle_basis
    # each cycle doesn't wrap around from Graphs.cycle_basis
    for cycle in cycles
        push!(cycle, cycle[begin])
    end
    cycles
end

"""
Persistence diagram drawn from mat with columns birth, persistence.
Optional dim can be supplied for convenient title.
"""
function persistence_diagram_trace(mat; kwargs...)
    scatter(;
            x=mat[:, 1],
            y=vec(sum(mat[:,:], dims=2)),
            mode="markers",
            kwargs...
            )
end

function persistence_diagram(pds...)
    # make sure HDF5 mats are read
    pds = [pd[:,:] for pd in pds]
    maxaxis = 0
    for pd in pds
        maxaxis = max(maxaxis, maximum(sum(pd, dims=2)))
    end
    ax = [0, maxaxis]
    traces = [persistence_diagram_trace(pd; name="dim=$dim") for (dim, pd) in enumerate(pds)]
    plot(
        [scatter(;
            x=999*ax,
            y=999*ax,
            mode="lines",
            marker_color="lightgray",
            showlegend=false,
            aspect_ratio=:equal,
        ); traces],
        Layout(
            template="simple_white",
            xaxis=attr(
                range=ax,
                scaleratio=1,
                scaleanchor="y",
                constrain="range",
                constraintoward="left",
            ),
            yaxis=attr(
                range=ax,
                scaleratio=1,
                scaleanchor="x",
                constrain="range",
                constraintoward="bottom",
            ),
            legend_x=0.5,
            legend_y=0.1,
            legend_xref="container",
            legend_yref="container",
        );
        config=PlotConfig(
            displaylogo=false,
            # not implemented in the julia version and last change to the github was 5 months ago.
            # showTips=false
        )
    )
end

function persistence_diagram(h5::HDF5.File, dims=1:2)
    persistence_diagram([h5["bars$dim"] for dim in dims]...)
end

# categorical palette with high chroma (minc=20)
# palette = ColorSchemes.glasbey_category10_n256
palette = ColorSchemes.glasbey_bw_minc_20_minl_30_n256
rgbs = [[rgb.r, rgb.g, rgb.b] for rgb in palette]
rgbs = [round.(Int, rgb .* 255) for rgb in rgbs]
rgbs = ["rgb("*join(rgb,',')*')' for rgb in rgbs]

"""
Get as much sequence info as available, i.e. amino acid letter plus number, just one of them.
Worst case returns a string of numbers 1:n.
"""
function get_seq(h5::HDF5.File)::Vector{String}
    attributes = attrs(h5)
    n = get(attributes, "n", length(h5["Cas"][:,1]))
    seq = get(attributes, "AA", ["" for _ in 1:n]) |> collect
    resi = get(attributes, "resi", 1:n) |> collect
    ["$s$i" for (s,i) in zip(seq, resi)]
end

function scatter3d(h5::HDF5.File, color=nothing; text=nothing, kwargs...)
    x, y, z = eachcol(h5["Cas"][:, 1:3])
    seq = get_seq(h5)
    if color === nothing
        color = "gray"
        global name = nothing
        global visible = nothing
    elseif text !== nothing
        global visible = "legendonly"
        if :name in keys(kwargs)
            name = NamedTuple(kwargs).name
        else
        end
    else
        if color == "pLDDT"
            color = h5["Cas"][:, 4]
            global name = "pLDDT"
        elseif color == "cent1"
            color = h5["Cas"][:, 5]
            global name = "cent1"
        elseif color == "cent2"
            color = h5["Cas"][:, 6]
            global name = "cent2"
        end
        global visible="legendonly"
        text = color
    end

    hovertemplate = "%{hovertext}<br>x = %{x: 6.3f}<br>y = %{y: 6.3f}<br>z = %{z: 6.3f}"
    extra = ""
    if name !== nothing
        extra *= "$name<br>"
    end
    if text !== nothing
        extra *= "%{text}"
    end
    if extra != ""
        hovertemplate *= "<extra>$extra</extra>"
    end

    PlotlyJS.scatter3d(
        ;
        x=x,
        y=y,
        z=z,
        hovertext=seq,
        marker_size=5,
        marker_color=color,
        name=name,
        text=text,
        visible=visible,
        mode="markers",
        hovertemplate=hovertemplate,
        kwargs...
    )
end

"""
Categorical data.
"""
function scatter3d(h5::HDF5.File, color::Vector{Int}; kwargs...)
    _color = [c > 0 && c ≤ length(rgbs) ? rgbs[c] : "gray" for c in color]
    scatter3d(h5, _color; text=color, kwargs...)
end


function scatter3dloops(h5::HDF5.File, topn::Int=3)
    Cas = h5["Cas"][:,:]
    bars1 = h5["bars1"][:,:]
    reps1 = h5["reps1"][:,:]
    maxind = size(bars1, 1)

    seq = get_seq(h5)

    hovertemplate = "%{hovertext}<br>x = %{x: 6.3f}<br>y = %{y: 6.3f}<br>z = %{z: 6.3f}"

    traces = GenericTrace[]
    for top in 1:topn
        rep = reps1[reps1[:,1] .== maxind - top + 1, 2:3]
        cycles = cycle_basis(rep)
        persistence = bars1[end-top+1,2]
        opacity = persistence / bars1[end,2] * .9
        extra = @sprintf("<extra>rep1 %d<br>persistence = %.5f</extra>", top, persistence)
        for (i_cycle, cycle) in enumerate(cycles)
            push!(traces,
                  PlotlyJS.scatter3d(
                  ;
                  x=Cas[cycle, 1],
                  y=Cas[cycle, 2],
                  z=Cas[cycle, 3],
                  mode="lines",
                  line_color=rgbs[top],
                  line_width=16,
                  opacity=opacity,
                  name="rep1 $top",
                  hovertext=seq[cycle],
                  hovertemplate = hovertemplate * extra,
                  legendgroup="rep1 $top",
                  visible="legendonly",
                  # combined with legendgroup, this means multiple
                  # cycles for a single homology group will be
                  # toggled together implicity.
                  showlegend=i_cycle==1,
                  ));
        end
    end
    traces
end

function scatter3dvoids(h5::HDF5.File, topn::Int=3)
    Cas = h5["Cas"][:,:]
    bars2 = h5["bars2"][:,:]
    reps2 = h5["reps2"][:,:]
    maxind = size(bars2,1)

    _ijk = reps2[reps2[:,1] .> maxind - topn, :]
    persistences = bars2[end-topn+1:end,2]
    opacities = persistences./maximum(persistences) .* .9
    # rgbs = palette[maxind .- _ijk[:,1] .+ 1]
    # rgbs = hcat([[rgb.r, rgb.g, rgb.b] for rgb in rgbs]...)'
    # rgbs = round.(Int, rgbs .* 255)

    seq = get_seq(h5)

    hovertemplate = "%{hovertext}<br>x = %{x: 6.3f}<br>y = %{y: 6.3f}<br>z = %{z: 6.3f}"

    traces = GenericTrace[]
    for top in 1:topn
        extra = @sprintf("<extra>rep2 %d<br>persistence = %.5f</extra>", top, persistences[end-top+1])

        push!(traces, mesh3d(
            ;
            x=Cas[:, 1],
            y=Cas[:, 2],
            z=Cas[:, 3],
            i=_ijk[_ijk[:,1].==maxind-top+1,2],
            j=_ijk[_ijk[:,1].==maxind-top+1,3],
            k=_ijk[_ijk[:,1].==maxind-top+1,4],
            opacity=opacities[end-top+1],
            hovertext=seq,
            hovertemplate=hovertemplate * extra,
            name="rep2 $top",
            showlegend=true,
            visible="legendonly"
        ))
    end
    return traces
end


mkpath(args.out)

for infile in args.infiles
    title = replace(basename(infile), r"\..*" => "")

    h5open(infile) do fid

        # calculate leiden communities unless already included in the file
        Cas = fid["Cas"][:,:]
        comm1 = if size(Cas, 2) ≥ 7
            Int.(Cas[:, 7])
        else
            leiden(fid, 1)
        end
        comm2 = if size(Cas, 2) ≥ 8
            Int.(Cas[:, 8])
        else
            leiden(fid, 2)
        end

        PD = persistence_diagram(fid)
        savefig(PD, joinpath(args.out, "$title-PD.html"))

        traces = [
            scatter3d(fid; name=title, mode=if args.no_lines "markers" else "markers+lines" end);
            scatter3d(fid, "cent1");
            scatter3d(fid, "cent2");
            scatter3d(fid, comm1; name="Leiden1");
            scatter3d(fid, comm2; name="Leiden2");
            scatter3dloops(fid)...;
            scatter3dvoids(fid);
        ];

        fig = plot(
            traces,
            Layout(
                title_text = title,
                template="simple_white",
            );
            config=PlotConfig(
                displaylogo=false,
                # not implemented in the julia version and last change to the github was 5 months ago.
                # showTips=false
            )
        )
        savefig(fig, joinpath(args.out, "$title-3D.html"))
    end
end

