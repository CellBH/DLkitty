using DLkitty
using DLkitty: reaction_rates_table
using DataFrames
using Statistics
using Plots
tbl = reaction_rates_table()
kcats = filter("parameter.type" => ==("kcat"), tbl)
filter!("parameter.unit" => x->!ismissing(x) && x==("s^(-1)"), kcats)

println("total kcats samples: ", nrow(kcats))

reaction_kcats = groupby(kcats, "Reaction")
reaction_kcat_counts = combine(nrow, reaction_kcats).nrow
println("total reactions: ", length(reaction_kcat_counts))
histogram(reaction_kcat_counts, xlabel="number of kcat samples")


_, most_common_reaction = findmax(nrow, reaction_kcats)
most_common_reaction_tbl = reaction_kcats[most_common_reaction]
vscodedisplay(most_common_reaction_tbl)

histogram(most_common_reaction_tbl.var"parameter.startValue")
