using Pkg; Pkg.activate()
using Revise
using DLkitty
using Random, DataFrames
using Statistics
using Zygote
using LinearAlgebra
using DLkitty: train_batch
versioninfo()
BLAS.get_num_threads()
BLAS.set_num_threads(1)

df = DLkitty.full_kcat_table()
nrow(df)
nrow(filter(is_usable, df))
nrow(kcat_table_train())
nrow(kcat_table_valid())
df_train = kcat_table_train()
df_valid = filter(is_complete, kcat_table_valid())
preprocessor = DLkitty.Preprocessor(df_train);

rows_to_use = 50
df_train_small = df_train[shuffle(1:nrow(df_train))[1:rows_to_use], :]
df_valid_small = df_valid[shuffle(1:nrow(df_valid))[1:rows_to_use], :]

resampled_df = resample(df_train_small; n_samples=1)
@time train_data = prep_data(preprocessor, resampled_df)
@time valid_data = prep_data(preprocessor, df_valid_small)

# single-sample SGD
@time tstate = train(preprocessor, train_data, valid_data; n_epochs=10, show_progress=true)
@time tstate = train(tstate, train_data, valid_data; n_epochs=5, show_progress=true)
#trained_model = TrainedModel(tstate)

# minibatch SGD
@time tstate = train_batch(preprocessor, train_data, valid_data; n_epochs=5, batchsize=10, show_progress=true)
@time tstate = train_batch(tstate, train_data, valid_data; n_epochs=5, batchsize=10, show_progress=true)
trained_model = TrainedModel(tstate)

# save model parameters
#using JLD2
#(;model, ps, st) = trained_model
#@save "trained_model_TEST.jld2" {compress = true} ps st

# check for spurious type promotion
(;model, ps, st) = trained_model
#lossf = DLkitty.L2RegLoss(DLkitty.PointEstimateLoss(), 1f-5)
#eltype(first(lossf(model, ps, st, train_data[1:5])))
lossf = DLkitty.L2RegLoss(DLkitty.MSELoss(), 1f-5)
eltype(first(lossf(model, ps, st, train_data[1])))

model_debug = DLkitty.Lux.Experimental.@debug_mode model
try
    model_debug(train_data[1][1], ps, st)
catch e
    println(e)
end

# simple profiling
using Profile
test() = lossf(model, ps, st, valid_data[1])
@profview test()
@profview_allocs test() sample_rate=1.0 C=true