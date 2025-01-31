@testset "no missings" begin
    df = DataFrame([
        (; id=1, Temperature=1, pH=1, Value=1.0),
        (; id=2, Temperature=2, pH=2, Value=2.0)
    ])

    new_df = resample(df; n_samples=100)

    @test df.id == [1,2]  # unchanged
    @test nrow(new_df) == 200
    @test count(==(1), new_df.id) == 100
    @test count(==(2), new_df.id) == 100
end


@testset "some missings" begin
    df = DataFrame([
        (; id=1, Temperature=10, pH=100, Value=1.0),
        (; id=2, Temperature=20, pH=200, Value=2.0),
        (; id=3, Temperature=30, pH=missing, Value=2.0),
        (; id=4, Temperature=missing, pH=missing, Value=2.0),
    ])

    new_df = resample(df; n_samples=100)
    @test nrow(new_df) == 400
    @test count(==(1), new_df.id) == 100
    @test count(==(2), new_df.id) == 100
    @test count(==(3), new_df.id) == 100
    @test count(==(4), new_df.id) == 100

    @test new_df.Temperature ⊆ [10,20,30]
    @test new_df.pH ⊆ [100, 200]
end