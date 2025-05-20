struct Preprocessor
    all_ngrams::Vector{String}
    all_fingerprints::Vector{Any}
    fingerprint_radius::Int
    temperature_transformer::ZStandardizer{Float64}
    ph_transformer::ZStandardizer{Float64}
end

function Preprocessor(df=kcat_table_train(); ngram_len::Integer=3, fingerprint_radius::Integer=3)
    return Preprocessor(
        all_sequence_ngrams(df, ngram_len),
        all_substrate_fingerprints(df, fingerprint_radius),
        fingerprint_radius,
        ZStandardizer(df.Temperature),
        ZStandardizer(df.pH),
    )
end

Base.propertynames(::Preprocessor, private=false) = (fieldnames(Preprocessor)..., :ngram_len)
function Base.getproperty(preprocessor::Preprocessor, name::Symbol)
    if name == :ngram_len
        length(first(preprocessor.all_ngrams))
    else
        return getfield(preprocessor, name)
    end
end

function preprocessor_filename(preprocessor::Preprocessor)
    radius = preprocessor.fingerprint_radius
    len = preprocessor.ngram_len
    return preprocessor_filename(radius, len)
end
preprocessor_filename(radius, len) = joinpath(dirname(@__DIR__), "data", "preprocessor__fingerprints$(radius)_ngrams$(len).jsz")

load_preprocessor(;fingerprint_radius=3, ngram_len=3) = deserialize(preprocessor_filename(fingerprint_radius, ngram_len))

"For Internal use ONLY, modified the install folder which probably breaks things if package isn't dev'ed"
save_preprocessor(preprocessor::Preprocessor) = serialize(preprocessor_filename(preprocessor), preprocessor)



function attach_fingerprint_onehots!(graph, preprocessor)
    fingerprint = extract_fingerprints(graph, preprocessor.fingerprint_radius)
    graph.ndata.fingerprint_onehots = onehotbatch(fingerprint, preprocessor.all_fingerprints, UNKNOWN_FINGERPRINT)
    return graph
end

function prep_input(preprocessor, datum)
    (; all_ngrams) = preprocessor
    ngram_len = preprocessor.ngram_len

    substrate_graphs = map(skipmissing(datum.SubstrateSMILES)) do smiles
        attach_fingerprint_onehots!(gnn_graph(smiles), preprocessor)
    end

    protein_1hots_seqs = map(skipmissing(datum.ProteinSequences)) do seq
        onehotbatch(ngrams(seq, ngram_len), all_ngrams, UNKNOWN_NGRAM)
    end

    # TODO: strongly consider z-normalzing temperature and pH
    temperature = preprocessor.temperature_transformer.(datum.Temperature)
    ph = preprocessor.ph_transformer.(datum.pH)
    return (substrate_graphs, protein_1hots_seqs, temperature, ph)
end

function  predict_kcat_dist((;model, ps, st), preprocessor, datum)
    prepped_datum = prep_input(preprocessor, datum)
    (y,), _ = model(prepped_datum, ps, st)
    return y
end


function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end


function train(
    df,
    preprocessor,
    opt=OptimiserChain(ClipGrad(1.0), Adam(0.0003f0));
    l2_coefficient=1e-5,
    n_samples=1000,
    n_epochs=3,
    log_train::Bool=false,
    logger::Union{TBLogger, Nothing}=log_train ? TBLogger("tensorboard_logs/run", min_level=Logging.Info) : nothing
)
    # Increasing n_samples and n_epochs do very similar thing
    # as either way things get duplicated, but n_samples means also correct missing data
    # however n_epochs uses less memory, because we do create the samples eagerly.

    model = DLkittyModel(preprocessor)
    tm = TrainedModel(model)
    tstate = Training.TrainState(tm, opt)
    
    usable_df = filter(is_usable, df)
    resampled_df = resample(usable_df; n_samples)
    prepped_data = map(Tables.namedtupleiterator(resampled_df)) do datum
        input = prep_input(preprocessor, datum)
        output = [datum]  # it has the fields we need already
        return input, output
    end

    _grads, unflatten_grads = ParameterHandling.flatten(tstate.parameters)
    grads = zeros(eltype(_grads), length(_grads))
    print(typeof(grads))

    for epoch in 1:n_epochs
        epoch_loss = 0.0
        grads[:] .= zero(eltype(grads))
        for (input, output) in prepped_data           
            try
                _grads, step_loss, _, tstate = Training.single_train_step!(
                    AutoZygote(), L2RegLoss(DistributionLoss(), l2_coefficient),
                    (input, output),
                    tstate
                )
                grads .+= ParameterHandling.flatten(eltype(grads), _grads)[1]
                epoch_loss += step_loss
            catch
                datum = output[]
                @error "issue with processing" datum
                rethrow()
            end
        end
        avg_loss = epoch_loss/length(prepped_data)
        @printf "Epoch: %3d \t Loss: %.5g\n" epoch avg_loss
        
        if log_train
            avg_grads = grads ./ length(prepped_data)
            _ps = tstate.parameters
            param_dict = Dict{String, Any}()
            grad_dict = Dict{String, Any}()
            fill_param_dict!(param_dict, _ps, "")
            fill_param_dict!(grad_dict, unflatten_grads(avg_grads), "")
            with_logger(logger) do
                @info "train" loss=avg_loss
                @info "model" params=param_dict log_step_increment=0
                @info "gradients" params=grad_dict log_step_increment=0
            end
        end

    end
    return TrainedModel(tstate)
end