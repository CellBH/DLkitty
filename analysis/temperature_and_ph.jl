using DLkitty
using Plots
using DataFrames

const full_df = kcat_table_train_and_valid()

plots_dir(fn) = joinpath(@__DIR__, "plots", fn)
for field in ("Temperature", "pH")
    df = dropmissing(full_df, field)

    histogram(
        df[:,field],
        normalize=:probability,
        ylabel="prob",
        xlabel=field,
        title="unweighted",
        legend=false,
        ylims=(0, 0.5),
    )
    savefig(plots_dir("hist_$(field)_unweighted.png"))
    for grouping in ("ECNumber", "Organism")
        weights = select(groupby(df, grouping), grp->1/nrow(grp)).x1
        histogram(
            df[:, field],
            normalize=:probability,
            ylabel="prob",
            xlabel=field,
            title="$grouping weighted",
            weights=weights,
            legend=false,
            ylims=(0, 0.5),
        )
        savefig(plots_dir("hist_$(field)_$(grouping)_weighted.png"))
    end
end