@testset "Dataset holdout" begin
    # check holding out enough
    full = DLkitty.full_kcat_table()
    allowed = kcat_table_train_and_valid()
    @test (nrow(full) - nrow(allowed))/nrow(full) ≈ 0.20 atol=0.1

    full_usable = filter(is_usable, full)
    allowed_usable =filter(is_usable, allowed)
    @test (nrow(full_usable) - nrow(allowed_usable))/nrow(full_usable) ≈ 0.20 atol=0.1
    
    # check holding out consistently even if shuffled
    s_allowed = kcat_table_train_and_valid(shuffle(full))
    @test nrow(s_allowed) == nrow(allowed)
end