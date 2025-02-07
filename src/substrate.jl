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

function gnn_graph(mol)
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