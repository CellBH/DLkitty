@testset "rdkit wrappers" begin
    smiles = "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"
    mol = mol_from_smiles(smiles)

    Descriptors = pyimport("rdkit.Chem.Descriptors")
    @test pyconvert(Float64, Descriptors.ExactMolWt(mol)) ≈ 272.17763

    adj = DLkitty.get_adjacency_matrix(mol)
    @test adj isa AbstractMatrix{Bool}
    @test 0 < mean(adj) < 0.1  # sparse-ish
    n_atoms = 18+24+2
    @test size(adj) == (n_atoms, n_atoms)
end

@testset "gnn_graph" begin
    smiles = "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"
    mol = mol_from_smiles(smiles)
    graph = gnn_graph(mol)

    adj = DLkitty.get_adjacency_matrix(mol)
    @test sum(adj) == ne(graph)
    @test size(adj, 1) == nv(graph)
    @test Set(graph.ndata.atomic_num) == Set((1,6,8))
    @test Set(graph.edata.bond_type) == Set((1, 12))  #single, aromatic
end

