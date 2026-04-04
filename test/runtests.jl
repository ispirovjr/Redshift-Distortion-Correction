using RedshiftDistortionCorrection
using Test
using Flux
using LinearAlgebra
using Statistics

@testset "RedshiftDistortionCorrection.jl" begin

    @testset "Graph" begin
        @testset "toSpherical" begin
            # Single point along +z axis
            pos = [0.0 0.0; 0.0 0.0; 0.0 5.0]
            obs = [0.0, 0.0, 0.0]
            r, θ, φ = toSpherical(pos, obs)
            @test r[1] == 0.0 || isnan(r[1])  # origin is degenerate
            @test r[2] ≈ 5.0
            @test θ[2] ≈ 0.0   # along +z → θ = 0
        end

        @testset "buildTessellation" begin
            # Simple 5 points (tetrahedron + center)
            pts = Float64[0 1 0 0 0.25;
                          0 0 1 0 0.25;
                          0 0 0 1 0.25]
            tets = buildTessellation(pts)
            @test size(tets, 2) == 4  # 4 vertices per tet
            @test size(tets, 1) >= 1  # at least one tet
        end

        @testset "extractNeighbors" begin
            pts = Float64[0 1 0 0 0.25;
                          0 0 1 0 0.25;
                          0 0 0 1 0.25]
            tets = buildTessellation(pts)
            neighbors = extractNeighbors(tets, 5)
            @test length(neighbors) == 5
            # Every point should have at least one neighbor
            @test all(length(n) > 0 for n in neighbors)
        end

        @testset "buildGraph" begin
            pts = randn(Float64, 3, 100)
            obs = [10.0, 0.0, 0.0]
            features, neighborIdx = buildGraph(pts, obs)
            @test size(features) == (3, 100)
            @test size(neighborIdx) == (4, 100)
            # Features should be Float32
            @test eltype(features) == Float32
            # All neighbor indices should be valid
            @test all(1 .<= neighborIdx .<= 100)
        end
    end

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
        result = net(x, idx)

        @test result isa Tuple
        pred, logσ = result
        @test size(pred) == (1, 50)
        @test size(logσ) == (1, 50)

        # Verify skip connection: pred should be x[1,:] + δ
        # δ comes from dense output row 1, so pred ≠ x[1,:] in general
        # but pred should have reasonable magnitude
        @test all(isfinite.(pred))
        @test all(isfinite.(logσ))
    end

    @testset "Per-File Normalization" begin
        features = Float32[10 20 30 40; 0.5 1.0 1.5 2.0; -1.0 0.0 1.0 2.0]
        targets = Float32[12 22 32 42]'

        featNorm, targNorm, rMax = normalizePerFile(features, targets)

        @test rMax ≈ 40.0f0
        # Radii should be normalized to [0, 1]
        @test maximum(featNorm[1, :]) ≈ 1.0f0
        @test minimum(featNorm[1, :]) ≈ 0.25f0
        # Angles should be unchanged
        @test featNorm[2, :] == features[2, :]
        @test featNorm[3, :] == features[3, :]
        # Targets should be scaled by same factor
        @test targNorm[1, 1] ≈ 12.0f0 / 40.0f0
        # Original arrays should not be mutated
        @test features[1, 1] ≈ 10.0f0
    end

    @testset "Uncertainty Loss" begin
        pred = Float32[1.0 2.0 3.0]
        target = Float32[1.1 1.9 3.2]
        logσ = Float32[0.0 0.0 0.0]  # σ = 1

        loss = uncertaintyLoss(pred, logσ, target)
        @test loss isa Real
        @test loss > 0  # loss should be positive

        # With σ = 1, loss ≈ mean of squared residuals (+ 0 for log terms)
        residuals = target .- pred
        expectedMSE = mean(residuals .^ 2)
        @test loss ≈ Float32(expectedMSE)

        # Larger uncertainty should reduce the loss when there are residuals
        logσ_large = Float32[1.0 1.0 1.0]
        loss_large = uncertaintyLoss(pred, logσ_large, target)
        # The residual term shrinks but log term grows — test just that it's finite
        @test isfinite(loss_large)

        # Zero residual with small σ should give lower loss than large σ
        pred_perfect = copy(target)
        loss_perfect_small_σ = uncertaintyLoss(pred_perfect, Float32[-2.0 -2.0 -2.0], target)
        loss_perfect_large_σ = uncertaintyLoss(pred_perfect, Float32[2.0 2.0 2.0], target)
        @test loss_perfect_small_σ < loss_perfect_large_σ
    end

    @testset "Utils" begin
        ŷ = rand(Float32, 1, 50) .+ 100f0
        y = ŷ .+ 0.01f0 .* randn(Float32, 1, 50)
        m = computeMetrics(ŷ, y)
        @test haskey(pairs(m), :mse)
        @test haskey(pairs(m), :mae)
        @test haskey(pairs(m), :r2)
    end

    @testset "Gradient Flow" begin
        # Verify gradients flow through the full model + loss
        net = buildCorrectionNetwork(; hiddenDim=8)
        x = randn(Float32, 3, 20)
        idx = rand(1:20, 4, 20)
        target = randn(Float32, 1, 20)

        # Get initial parameters
        paramsBefore, _ = Flux.destructure(net)

        loss, grads = Flux.withgradient(net) do m
            pred, logσ = m(x, idx)
            uncertaintyLoss(pred, logσ, target)
        end

        @test isfinite(loss)
        @test grads[1] !== nothing

        # Apply one gradient step and verify parameters changed
        opt = Flux.setup(Flux.Adam(1e-2), net)
        Flux.update!(opt, net, grads[1])
        paramsAfter, _ = Flux.destructure(net)
        @test paramsBefore != paramsAfter
    end
end
