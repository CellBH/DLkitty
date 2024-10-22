

function init_data()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true  # bypass prompts

    register(DataDep(
        "dlkitty_reaction_rates",
        """
        Data originally from SABIO???
        prepared by Michael and Augustinas.
        """,
        "https://dlkitty.file.core.windows.net/dlkitty/initial_dataset.tsv?sp=r&st=2024-10-22T10:35:35Z&se=2052-10-22T10:27:00Z&sv=2022-11-02&sig=o8AMnCfjwxRtGEGaditHwGj25Qg%2FBKZx3N5eYyxcu%2Bo%3D&sr=f"
    )),
    "14f0c966b51603e15c9e0fce5e378ae0cbb8595c6e4ebd5caf9221d01e467f06"
end

reaction_rates_file() = datadep"dlkitty_reaction_rates/initial_dataset.tsv"
reaction_rates_table() = DataFrame(CSV.File(reaction_rates_file()))