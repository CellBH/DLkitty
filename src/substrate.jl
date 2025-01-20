function mol_from_smiles(smiles)
    Chem = @pyconst pyimport("rdkit.Chem")
    mol = Chem.rdmolfiles.MolFromSmiles(smiles)
    # we always want to include hydrogen, because thats what DLKcat does
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
    return GNNGraph(
        (sources, targets);
        graph_type=:coo, edata=columntable(edge_data), ndata=columntable(node_data),
    )
end