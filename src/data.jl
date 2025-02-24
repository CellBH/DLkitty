

function init_data()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true  # bypass prompts

    register(DataDep(
        "dlkitty_reaction_rates",
        """
        First basic set of data, Data originally from SABIO
        prepared by Michael and Augustinas.
        """,
        "https://dlkitty.file.core.windows.net/dlkitty/initial_dataset.tsv?sp=r&st=2024-10-22T10:35:35Z&se=2052-10-22T10:27:00Z&sv=2022-11-02&sig=o8AMnCfjwxRtGEGaditHwGj25Qg%2FBKZx3N5eYyxcu%2Bo%3D&sr=f"
    )),
    "14f0c966b51603e15c9e0fce5e378ae0cbb8595c6e4ebd5caf9221d01e467f06"


    register(DataDep(
        "Sabio-RK_kcats_full_nov_2024",
        """
        Kcat data from Sabio
        crossreferenced with protien sequences from Uniprot
        crossreferenced with SMILES from pubchem.
        Some data missing
        """,
        "https://dlkitty.file.core.windows.net/dlkitty/Sabio-RK_kcats_full_nov_2024.json?sp=r&st=2024-12-03T07:17:45Z&se=2165-12-04T07:09:00Z&spr=https&sv=2022-11-02&sig=R2BV4ogxEZE6I3uDkRwlIVKZt%2BtdPnIdZaHnG7M%2FCAc%3D&sr=f",
        "d513e29d9c21ce636b45a17860817a0575e9a18c50cee29903b63dec67c46c47"
    ))
end
###########

# early data

reaction_rates_file() = datadep"dlkitty_reaction_rates/initial_dataset.tsv"
reaction_rates_table() = DataFrame(CSV.File(reaction_rates_file()))

###############

full_kcat_file() = datadep"Sabio-RK_kcats_full_nov_2024/Sabio-RK_kcats_full_nov_2024.json"
function full_kcat_table()
    df = DataFrames.DataFrame(jsontable(full_kcat_file()))
    df.ProteinSequences = map(df.ProteinSequences) do protseqs
        if ismissing(protseqs)
            missing
        else
            [isnothing(seq) ? missing : seq for seq in protseqs]
        end
    end
    df.SubstrateSMILES = map(df.SubstrateSMILES) do smilesseqs
        [isnothing(seq) ? missing : seq for seq in smilesseqs]
    end
    return df
end


function kcat_table_train_and_valid(full_df=full_kcat_table())
    return filter(!is_holdout, full_df)
end


function kcat_table_train(full_df=full_kcat_table())
    return filter(is_train, full_df)
end


function kcat_table_valid(full_df=full_kcat_table())
    return filter(is_validation, full_df)
end


##################################################

"""
This function will consistently assign a given row to the holdout or not.
Even if we have duplicates (both will be assigned to holdout or not).
Or even if it is shuffled or some are deleted or added.
"""
function is_holdout(row)
    fields = (row.Value, row.StandardDeviation, row.Temperature, row.pH, row.UniProtID, row.Substrate)
    id_hash = foldr(hash, fields, init=0x00FAEBABE)
    # select 3/16ths (18.75%) of all items.
    # Can drop 1 of these later to shrink our holdout set.
    return (id_hash & 0x0F) ∈ (0x00, 0x01, 0x02)
end

"""
This function will consistently assign a given row to the static validation or not.
No need to use this if doing kfolds cross validation.
"""
function is_validation(row)
    fields = (row.Value, row.StandardDeviation, row.Temperature, row.pH, row.UniProtID, row.Substrate)
    id_hash = foldr(hash, fields, init=0x00FAEBABE)
    # select 3/16ths (18.75%) of all items.
    # Can drop 1 of these later to shrink our holdout set.
    return (id_hash & 0x0F) ∈ (0x03, 0x04, 0x05)
end

"""
This function will consistently assign a given row to the validation or not.
No need to use this if doing kfolds cross validation.
"""
is_train(row) = !is_validation(row) && !is_holdout(row)


function is_usable(row)
    has_smiles = all(!ismissing, row.SubstrateSMILES) && length(row.SubstrateSMILES) > 0
    has_seq = !ismissing(row.ProteinSequences) && all(!ismissing, row.ProteinSequences) && length(row.ProteinSequences) > 0
    return has_smiles && has_seq
end

is_complete(row) = is_usable(row) && !ismissing(row.Temperature) && !ismissing(row.pH)

