using DLkitty
using Test

@testset "DLkitty.jl" begin
    @test DLkitty.hello_world() == "Hello, World!"
end
