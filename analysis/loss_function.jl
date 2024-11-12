using DLkitty
using Distributions
using Random
using Zygote
using Plots
import DLkitty: DistributionLoss, loss1
rng = Random.default_rng()
Random.seed!(rng, 0)

n_samples = 10_000
target_dist = LogNormal(0.1, 0.5)
point_samples = rand(rng, target_dist, n_samples)

stat_samples = map(1:n_samples) do _
    data = rand(rng, target_dist, 5)
    return (; mean=mean(data), std=std(data))
end

loss_fun = DistributionLoss(mean_std_weight=3.0)
point_losses = loss1.(loss_fun, target_dist, point_samples)

stat_losses = loss1.(loss_fun, target_dist, stat_samples)


histogram(
    [point_losses stat_losses],
    bins=0:0.1:5, alpha=0.5,
    label=["Point Losses" "Stat Losses"]
)

