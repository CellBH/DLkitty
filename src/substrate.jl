function mol_from_smiles(smiles)
    Chem = @pyconst pyimport("rdkit.Chem")
    parser_params = Chem.rdmolfiles.SmilesParserParams()
    parser_params.removeHs = false
    mol = Chem.rdmolfiles.MolFromSmiles(smiles, parser_params)
    # For some reason just tsetting the parser_params doesn't always do it.
    return Chem.rdmolops.AddHs(mol)
end

function get_adjacency_matrix(mol)
    Chem = @pyconst pyimport("rdkit"=>"Chem")
    return pyconvert(Matrix{Bool}, Chem.GetAdjacencyMatrix(mol))
end

gnn_graph(smiles::AbstractString) = gnn_graph(mol_from_smiles(smiles))

function gnn_graph(mol)::GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}
    Chem = @pyconst pyimport("rdkit.Chem")

    sources = Int[]
    targets = Int[]
    edge_data = @NamedTuple{bond_type::Int}[]
    for bond in mol.GetBonds()
        @assert pytruth(bond.GetBondDir() == Chem.rdchem.BondDir.NONE)
        a, b = pyconvert.(Int, (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) .+ 1
        for (source, target) in ((a,b), (b,a))
            # handle both directions as it is symmetric
            push!(sources, source)
            push!(targets, target)
            push!(edge_data, (;
                bond_type=pyconvert(Int, bond.GetBondType()),
            ))
        end
    end

    node_data = @NamedTuple{atomic_num::Int}[]
    for atom in mol.GetAtoms()
        push!(node_data, (;
            atomic_num = bond_type=pyconvert(Int, atom.GetAtomicNum()),
            # TODO: insert fingerprint here: https://github.com/CellBH/DLkitty/issues/9
        ))
    end
    
    graph =  GNNGraph(
        (sources, targets);
        graph_type=:coo, num_nodes=length(node_data),
        ndata=columntable(node_data),
    )
    # add edata after to
    # work-around: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/582
    for (fname, fdata) in pairs(columntable(edge_data))
        setproperty!(graph.edata, fname, fdata)
    end
    return graph
end

function _create_i_j_edge_bond_dict(graph)
    i_j_edge_dict = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (edge, bond) in zip(Graphs.edges(graph), graph.edata.bond_type)
        push!(
            get!(Vector{Tuple{Int, Int}}, i_j_edge_dict, Graphs.src(edge)),
            (Graphs.dst(edge), bond)
        )
    end
    return i_j_edge_dict
end

function extract_fingerprints(graph, radius=3)
    fnodes = graph.ndata.atomic_num
    if length(fnodes) == 1 || radius == 0
        return map(tuple, fnodes)
    end
    
    
    i_j_edge_dict = _create_i_j_edge_bond_dict(graph)
    for cur_rad in 1:radius
        # Update each node id to be the fingerprint id,
        # by considering its neighboring nodes and edges.
        # i.e. r-radius subgraphs or fingerprints
        fnodes = map(eachindex(fnodes)) do i
            #TODO I think this could be written more clearly by using Graphs.neighbors
            # rather than having that encoded in the i_j_edge_dict by j being first element of tuple

            neibs = [(fnodes[j], edge) for (j, edge) in  i_j_edge_dict[i]]
            fingerprint = (fnodes[i], sort(neibs))
            return fingerprint
        end
        cur_rad == radius && break  # exit early

        # prepare for next round by updating i_j_edge_dict
        i_j_edge_dict = Iterators.map(i_j_edge_dict) do (i, j_edges)
            new_j_edges = Tuple{Int, Tuple}[]
            for (j, edge_f) in j_edges
                both_sides = minmax(i, j)
                push!(
                    new_j_edges,
                    (j, (both_sides, edge_f))
                )
            end
            return i => new_j_edges
        end |> Dict
    end
    return fnodes
end

UNKNOWN_FINGERPRINT = missing

function all_substrate_fingerprints(df, radius)
    smiles = skipmissing(reduce(union!, df.SubstrateSMILES; init=Set{Union{Missing,String}}()))
    seen = []
    for s in smiles
        try
            push!(seen, extract_fingerprints(gnn_graph(s), radius))
        catch
            @error "preparing fingerprint" s
            rethrow()
        end
    end
    return push!(collect(seen), UNKNOWN_FINGERPRINT)
end

function save_all_substrate_fingerprints(df, radius)
    fingerprints = all_substrate_fingerprints(df, radius)
    open(joinpath(dirname(@__DIR__), "data", "all_fingerprints_$(radius).txt"), "w") do fh
        for fingerprint in fingerprints
            println(fh, repr(fingerprint))
        end
    end
end

function load_all_substrate_fingerprints(radius)
    open(collect∘eval∘Meta.parse∘eachline, joinpath(dirname(@__DIR__), "data", "all_fingerprints_$(radius).txt"))
end