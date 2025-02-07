function resample(df; n_samples=100, fields_to_impute=[:Temperature, :pH], shuffle=true)
    rows = Tables.namedtupleiterator(df)
    
    fill_sources = NamedTuple([
        field => collect(skipmissing(df[:, field]))
        for field in fields_to_impute]
    )
    # use a function barrier:
    return _resample(rows, fill_sources, n_samples, shuffle)
end

function _resample(rows, fill_sources, n_samples, shuffle)
    new_rows = eltype(rows)[]
    sizehint!(new_rows, n_samples*length(rows))
    for row in rows
        for _ in 1:n_samples
            new_row = row
            for field in keys(fill_sources)
                if ismissing(row[field])
                    val = rand(fill_sources[field])
                    new_row = merge(new_row, (field=>val,))
                end
            end
            push!(new_rows, new_row)
        end
    end

    shuffle && shuffle!(new_rows)
    return DataFrame(new_rows)
end

