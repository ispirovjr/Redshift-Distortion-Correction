using RedshiftDistortionCorrection
using Test

@testset "RedshiftDistortionCorrection.jl" begin

    @testset "Data" begin
        obs = [50.0, 50.0, -1500.0]
        data = generateSyntheticData(100, obs; seed=99)
        @test size(data.features) == (3, 100)
        @test size(data.targets, 2) == 100
        @test size(data.neighborIdx) == (4, 100)

        splits = splitData(100)
        total = length(splits.trainIdx) + length(splits.valIdx) + length(splits.testIdx)
        @test total == 100
    end

    @testset "Model" begin
        net = buildCorrectionNetwork(; hiddenDim=16)

        # Dummy forward pass
        x = randn(Float32, 3, 50)
        idx = rand(1:50, 4, 50)
        y = net(x, idx)
        @test size(y) == (1, 50)
        @test all(y .>= 0)   # ReLU guarantees non-negative
    end

    @testset "Utils" begin
        ŷ = rand(Float32, 1, 50) .+ 100f0
        y = ŷ .+ 0.01f0 .* randn(Float32, 1, 50)
        m = computeMetrics(ŷ, y)
        @test haskey(pairs(m), :mse)
        @test haskey(pairs(m), :mae)
        @test haskey(pairs(m), :r2)
    end
end
