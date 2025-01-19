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

@testset "GNN" begin
    smileses = [
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CCC4=C3C=CC(=C4)O",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1C[C@H]([C@@H]2O)O)CCC4=C3C=CC(=C4)O",
        "N",
        "O=C=O",
    ]
    mols = mol_from_smiles.(smileses)
    Descriptors = pyimport("rdkit.Chem.Descriptors")
    weights = pyconvert.(Float64, Descriptors.ExactMolWt.(mols))

    graphs = gnn_graph.(mols)

    atomic_nums = 100  # 94 gives us plutonium, so 6 past that is plenty
    bond_types = 22

    

    @testset "using only atomic weights" begin
        model = GNNChain(
            Embedding(atomic_nums=>16),
            CGConv(16 => 64, relu; residual=false),
            CGConv(64 => 64, relu; residual=true),
            CGConv(64 => 64, relu; residual=true),
            x -> mean(x, dims=2),  # combine all node data
            Dense(64, 1);
        )
        function loss(model, ps, st, (g,x,y))
            ŷ, st = model(g, x, ps, st)  
            return MSELoss()(ŷ, y), (layers = st,), 0
        end

        rng = Xoshiro(0)
        ps, st = LuxCore.setup(rng, model)
    
        train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
        train_losses = Float64[]
        for iter in 1:100
            iter_loss = 0.0
            for (g, y) in zip(graphs, weights)
                x = g.ndata.atomic_num
                _, step_loss, _, train_state = Lux.Training.single_train_step!(
                    AutoZygote(), loss, (g, x, y), train_state
                )
                
                iter_loss += step_loss
            end
            println(iter_loss)
            push!(train_losses, iter_loss)
        end
        
        # should be at least 10x better by the end
        @test train_losses[end] < 0.1 * train_losses[1]

        # no single prediction super wrong
        ŷs = map(graphs) do g
            y, _= model(g, g.ndata.atomic_num, ps, st)
            return y[]
        end
        @test ŷs ≈ weights rtol=0.05 # at most off by 5%
    end
end

