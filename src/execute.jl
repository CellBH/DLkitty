
function prep_input(datum, all_ngrams)
    ngram_len = length(first(all_ngrams))

    substrate_graphs = map(skipmissing(datum.SubstrateSMILES)) do smiles
        mol = mol_from_smiles(smiles)   
        gnn_graph(mol)
    end

    protein_1hots_seqs = map(skipmissing(datum.ProteinSequences)) do seq
        onehotbatch(ngrams(seq, ngram_len), all_ngrams, UNKNOWN_NGRAM)
    end

    # TODO: strongly consider z-normalzing temperature and pH
    temperature = datum.Temperature
    ph = datum.pH
    return (substrate_graphs, protein_1hots_seqs, temperature, ph)
end

function  predict_kcat_dist((;model, ps, st), all_ngrams, datum)
    prepped_datum = prep_input(datum, all_ngrams)
    (y,), _ = model(prepped_datum, ps, st)
    return y
end


function train(
    df,
    all_ngrams,
    opt=Adam(0.0003f0);
    l2_coefficient=1e-5,
    n_samples=1000,
    n_epochs=3
)
    # Increasing n_samples and n_epochs do very similar thing
    # as either way things get duplicated, but n_samples means also correct missing data
    # however n_epochs uses less memory, because we do create the samples eagerly.

    model = DLkittyModel(; num_unique_ngrams=length(all_ngrams))
    tm = TrainedModel(model)

    usable_df = filter(is_usable, df)
    resampled_df = resample(usable_df; n_samples)
    prepped_data = map(Tables.namedtupleiterator(resampled_df)) do datum
        input = prep_input(datum, all_ngrams)
        output = [datum]  # it has the fields we need already
        return input, output
    end

    tstate = Training.TrainState(tm, opt)
    for epoch in 1:n_epochs
        epoch_loss = 0.0
        for (input, output) in prepped_data           
            try
                _, step_loss, _, tstate = Training.single_train_step!(
                    AutoZygote(), L2RegLoss(DistributionLoss(), l2_coefficient),
                    (input, output),
                    tstate
                )
                epoch_loss += step_loss
            catch
                datum = output[]
                @error "issue with processing" datum
                rethrow()
            end
        end
        average_loss = epoch_loss/length(prepped_data)
        @printf "Epoch: %3d \t Loss: %.5g\n" epoch average_loss
    end
    return TrainedModel(tstate)
end