@testset "ZStandardizer" begin
    ZStandardizer = DLkitty.ZStandardizer

    x = [10, 20, 30, missing]
    trans = ZStandardizer(x)
    x′ = trans.(x)
    @test x′ isa Vector{Union{Float64, Missing}} 
    @test mean(skipmissing(x′)) == 0
    @test std(skipmissing(x′)) == 1
    @test trans(20) == 0
    @test ismissing(trans(missing))
    @test trans(40) == 2
end