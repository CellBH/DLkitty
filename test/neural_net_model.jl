@testset "Protein Sequence Input" begin
    df = kcat_table_train_and_valid()
    ngram_len = 3
    all_ngrams = DLkitty.all_sequence_ngrams(df, ngram_len)

    # These are all single protien cases:
    seqs_batch1 = only.(df.ProteinSequences[1:5])
    ngrams_seqs_batch1 = DLkitty.ngrams.(seqs_batch1, ngram_len)
    onehots_seqs_batch1 = onehotbatch.(ngrams_seqs_batch1, (all_ngrams,), DLkitty.UNKNOWN_NGRAM)
    @test onehots_seqs_batch1 isa Vector{<:OneHotMatrix}
    @test onecold.(onehots_seqs_batch1, (all_ngrams,)) == ngrams_seqs_batch1


    num_unique_ngrams = length(all_ngrams)
    attention_cnn = DLkitty.AttentionCNN(num_unique_ngrams)

    substrate_vector = randn(20)
    ps, st = Lux.setup(Xoshiro(0), attention_cnn)
    for onehots_seq in onehots_seqs_batch1
        z, _= attention_cnn((substrate_vector, onehots_seq), ps, st)
        @test size(z)== (20,)
    end
end

@testset "two substrates train" begin
    datum1 = @NamedTuple{PubMedID::Union{Missing, Int64}, Organism::String, Substrate::Vector{String}, ECNumber::Union{Missing, Vector{String}}, EnzymeName::Union{Missing, String}, EnzymeType::String, UniProtID::Union{Missing, Vector{String}}, pH::Union{Missing, Float64}, Temperature::Union{Missing, Real}, Value::Real, StandardDeviation::Union{Missing, Float64}, ProteinSequences::Union{Missing, Vector}, SubstrateSMILES::Vector}(
        (8369299, "Escherichia coli", ["Isocitrate", "NAD+"], ["1.1.1.42"], "isocitrate dehydrogenase (NADP+)", "wildtype", ["P08200"], 7.3, 21, 3.22, missing, ["MESKVVVPAQGKKITLQNGKLNVPENPIIPYIEGDGIGVDVTPAMLKVVDAAVEKAYKGERKISWMEIYTGEKSTQVYGQDVWLPAETLDLIREYRVAIKGPLTTPVGGGIRSLNVALRQELDLYICLRPVRYYQGTPSPVKHPELTDMVIFRENSEDIYAGIEWKADSADAEKVIKFLREEMGVKKIRFPEHCGIGIKPCSEEGTKRLVRAAIEYAIANDRDSVTLVHKGNIMKFTEGAFKDWGYQLAREEFGGELIDGGPWLKVKNPNTGKEIVIKDVIADAFLQQILLRPAEYDVIACMNLNGDYISDALAAQVGGIGIAPGANIGDECALFEATHGTAPKYAGQDKVNPGSIILSAEMMLRHMGWTEAADLIVKGMEGAINAKTVTYDFERLMDGAKLLKCSEFGDAIIENM"], ["C(C(C(C(=O)O)O)C(=O)O)C(=O)O", "C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)([O-])OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O)C(=O)N"])
    )
    datum2 = @NamedTuple{PubMedID::Union{Missing, Int64}, Organism::String, Substrate::Vector{String}, ECNumber::Union{Missing, Vector{String}}, EnzymeName::Union{Missing, String}, EnzymeType::String, UniProtID::Union{Missing, Vector{String}}, pH::Union{Missing, Float64}, Temperature::Union{Missing, Real}, Value::Real, StandardDeviation::Union{Missing, Float64}, ProteinSequences::Union{Missing, Vector}, SubstrateSMILES::Vector}(
        (8369299, "Escherichia coli", ["Isocitrate", "NAD+"], ["1.1.1.42"], "isocitrate dehydrogenase (NADP+)", "wildtype", ["P08200"], 8.3, 31, 4.22, missing, ["MESKVVVPAQGKKITLQNGKLNVPENPIIPYIEGDGIGVDVTPAMLKVVDAAVEKAYKGERKISWMEIYTGEKSTQVYGQDVWLPAETLDLIREYRVAIKGPLTTPVGGGIRSLNVALRQELDLYICLRPVRYYQGTPSPVKHPELTDMVIFRENSEDIYAGIEWKADSADAEKVIKFLREEMGVKKIRFPEHCGIGIKPCSEEGTKRLVRAAIEYAIANDRDSVTLVHKGNIMKFTEGAFKDWGYQLAREEFGGELIDGGPWLKVKNPNTGKEIVIKDVIADAFLQQILLRPAEYDVIACMNLNGDYISDALAAQVGGIGIAPGANIGDECALFEATHGTAPKYAGQDKVNPGSIILSAEMMLRHMGWTEAADLIVKGMEGAINAKTVTYDFERLMDGAKLLKCSEFGDAIIENM"], ["C(C(C(C(=O)O)O)C(=O)O)C(=O)O", "C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)([O-])OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O)C(=O)N"])
    )
    seq = datum.ProteinSequences[1]
    fingerprint_radius = 3
    preprocessor = Preprocessor(DataFrame([datum1, datum2]))
    tm = TrainedModel(DLkittyModel(preprocessor))
    tstate = Training.TrainState(tm, Adam(0.003f0))

    # Test forward
    input = DLkitty.prep_input(preprocessor, datum)
    (y,), _ = tm.model(input, tm.ps, tm.st)
    @test y isa LogNormal

    # Test reverse
    output = [datum1]  # it has the fields we need already
    _, step_loss, _, _ = Training.single_train_step!(
        AutoZygote(), DLkitty.DistributionLoss(),
        (input, output),
        tstate
    )
    @test isfinite(step_loss)
    @test step_loss >= 0
end