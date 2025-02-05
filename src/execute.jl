
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
    n_samples=1000,
    n_epochs=10
)
    # Increasing n_samples and n_epochs do very similar thing
    # as either way things get duplicated, but n_samples means also correct missing data
    # however n_epochs uses less memory, because we do create the samples eagerly.

    model = DLkittyModel(; num_unique_ngrams=length(all_ngrams))
    tm = TrainedModel(model)

    usable_df = filter(is_usable, df)
    resampled_df = resample(usable_df; n_samples)
    tstate = Training.TrainState(tm, opt)
    for epoch in 1:n_epochs
        epoch_loss = 0.0
        for datum in Tables.namedtupleiterator(resampled_df)
            input = try
                prep_input(datum, all_ngrams)
            catch err
                @warn "failured to preprocess a datum (skipping)" datum exception=err
                continue
            end
            output = [datum]  # it has the fields we need already
            try
                _, step_loss, _, tstate = Training.single_train_step!(
                    AutoZygote(), DistributionLoss(),
                    (input, output),
                    tstate
                )
                epoch_loss += step_loss
            catch
                @error "issue with processing" datum
                rethrow()
            end
        end
        average_loss = epoch_loss/nrow(resampled_df)
        @printf "Epoch: %3d \t Loss: %.5g\n" epoch average_loss
    end
    return TrainedModel(tstate)
end