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

function prep_input(preprocessor, datum, graphs)
    (; all_ngrams) = preprocessor
    ngram_len = preprocessor.ngram_len

    substrate_graphs = attach_fingerprint_onehots!.(graphs, (preprocessor,))

    protein_1hots_seqs = map(skipmissing(datum.ProteinSequences)) do seq
        onehotbatch(ngrams(seq, ngram_len), all_ngrams, UNKNOWN_NGRAM)
    end

    temperature = preprocessor.temperature_transformer.(datum.Temperature)
    ph = preprocessor.ph_transformer.(datum.pH)
    temperature = Float32.(temperature)
    ph = Float32.(ph)
    return (substrate_graphs, protein_1hots_seqs, temperature, ph)
end

get_gnn_graphs(datum::DataFrameRow) = map(gnn_graph, skipmissing(datum.SubstrateSMILES))

function get_gnn_graphs(df::DataFrame)
    # no multithreading as performing parallel PythonCalls is messy
    all_graphs = map(eachrow(df)) do datum
        get_gnn_graphs(datum)
    end
    return all_graphs
end

function prep_data(preprocessor, df)
    # multi-threaded data prep
    all_graphs = get_gnn_graphs(df)
    # NOTE: Single-point estimate
    data = DLkitty.OhMyThreads.tmap(1:DLkitty.nrow(df)) do i
        datum = df[i, :]
        graphs = all_graphs[i]
        DLkitty.prep_input(preprocessor, datum, graphs), Float32.(log(datum.Value))
    end
    return data
end

function predict_kcat_dist((;model, ps, st), preprocessor, datum)
    prepped_datum = prep_input(preprocessor, datum, get_gnn_graphs(datum))
    (y,), _ = model(prepped_datum, ps, st)
    return y
end

function compute_loss(lossf, model, ps, st, data::AbstractArray)
    loss = OhMyThreads.tmapreduce(+, 1:length(data)) do i
        lossf(model, ps, st, data[i])[1]
    end
    return loss/length(data)
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

function train(preprocessor::Preprocessor, 
               args...; 
               opt=OptimiserChain(ClipGrad(1.0), Adam(0.0005f0)), 
               kwargs...)
    model = DLkittyModel(preprocessor)
    tm = TrainedModel(model)
    tstate = Training.TrainState(tm, opt)
    train(tstate, args...; kwargs...)
end

function train(
    tstate::Training.TrainState,
    train_data,
    valid_data;
    l2_coefficient=1f-5,
    n_epochs=3,
    ad::Lux.AbstractADType=AutoZygote(),
    show_progress::Bool=false,
    logger::TBLogger=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
)
    #lossf = L2RegLoss(DistributionLoss(), l2_coefficient)
    # NOTE: Single-point estimate
    lossf = L2RegLoss(MSELoss(), l2_coefficient)
    _grads, unflatten_grads = ParameterHandling.flatten(tstate.parameters)
    grads = zeros(length(_grads))
    for epoch in 1:n_epochs
        #BLAS.set_num_threads(6)
        #println("BLAS threads: ", BLAS.get_num_threads())
        epoch_loss = 0f0
        grads[:] .= 0f0
        p = Progress(length(train_data); enabled=show_progress, showspeed=true)
        for (input, output) in train_data           
            try
                _grads, step_loss, _, tstate = Training.single_train_step!(
                    ad, lossf, (input, output), tstate
                )
                grads .+= ParameterHandling.flatten(_grads)[1]
                epoch_loss += step_loss
                next!(p)
            catch
                datum = output[]
                @error "issue with processing" datum
                rethrow()
            end
        end
        average_loss = epoch_loss/length(train_data)
        avg_grads = grads ./ length(train_data)
        _ps = tstate.parameters
        _st = Lux.testmode(tstate.states)
        valid_loss = compute_loss(lossf, tstate.model, _ps, _st, valid_data)
        show_progress && @printf "Epoch: %3d \t Training Loss: %.5g \t Validation Loss: %.5g\n" epoch average_loss valid_loss
        param_dict = Dict{String, Any}()
        grad_dict = Dict{String, Any}()
        fill_param_dict!(param_dict, _ps, "")
        fill_param_dict!(grad_dict, unflatten_grads(avg_grads), "")
        with_logger(logger) do
            @info "train" loss=average_loss
            @info "valid" loss=valid_loss log_step_increment=0
            @info "model" params=param_dict log_step_increment=0
            @info "gradients" params=grad_dict log_step_increment=0
        end
    end
    return tstate
end
    return TrainedModel(tstate)
end