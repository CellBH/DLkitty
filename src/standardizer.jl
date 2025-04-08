# StatsBase's ZScoreTransform is too old and borked
# MLJ's is too complicated
# Just write out own little one.

struct ZStandardizer{T}
    mean::T
    std::T
end

ZStandardizer(data) = ZStandardizer(mean(skipmissing(data)), std(skipmissing(data)))
(trans::ZStandardizer)(x) = (x-trans.mean)/trans.std
