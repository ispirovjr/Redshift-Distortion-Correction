# ══════════════════════════════════════════════════════════════════
#  data.jl — Dataset loading, synthetic generation, splitting
#
#  For the correction network, features are spherical coordinates
#  (r_obs, θ, φ) and the target is R_true (scalar per node).
# ══════════════════════════════════════════════════════════════════

"""
Load a dataset from a JLD2 file.

Expected keys:
- `features`    : 3 × N Float32  (r_obs, θ, φ)
- `targets`     : 1 × N Float32  (R_true)
- `neighborIdx` : 4 × N Int      (sorted neighbor indices)
"""
function loadDataset(path::String)
    data = JLD2.load(path)
    return (
        features=data["features"],
        targets=data["targets"],
        neighborIdx=data["neighborIdx"],
    )
end

"""
Generate a synthetic infall dataset in spherical coordinates.
Uniform velocity toward the center of mass is assumed.

Returns `(features, targets, neighborIdx)`:
- `features`     : 3 × N Float32  (r_distorted, θ, φ)
- `targets`      : 1 × N Float32  (r_true)
- `neighborIdx`  : 4 × N Int      (from tessellating distorted positions)

"""
function generateSyntheticData(nPoints::Int, observer::Vector{Float64};
    vPecMag::Float64=500.0,
    H0::Float64=70.0,
    boxSize::Float64=100.0,
    seed::Int=13)
    rng = MersenneTwister(seed)

    truePos = rand(rng, Float64, 3, nPoints) .* boxSize

    com = vec(mean(truePos; dims=2)) #Center of mass
    vPec = zeros(3, nPoints)
    for i in 1:nPoints
        dir = com .- @view(truePos[:, i])
        vPec[:, i] .= vPecMag .* dir ./ norm(dir)
    end

    distortedPos = similar(truePos)
    for i in 1:nPoints
        losVec = @view(truePos[:, i]) .- observer #line of sight vector
        dTrue = norm(losVec)
        losHat = losVec ./ dTrue

        vObs = H0 * dTrue + dot(@view(vPec[:, i]), losHat) #distortions only radially
        dDistort = vObs / H0
        distortedPos[:, i] .= observer .+ dDistort .* losHat
    end


    # build graph
    features, neighborIdx = buildGraph(distortedPos, observer)
    rTrue, _, _ = toSpherical(truePos, observer)
    targets = Float32.(rTrue')

    return (
        features=features,
        targets=targets,
        neighborIdx=neighborIdx,
    )
end

"""
Return disjoint index vectors for train / validation / test splits.
"""
function splitData(n::Int; trainFrac::Float64=0.7,
    valFrac::Float64=0.15, seed::Int=42)
    rng = MersenneTwister(seed) # we shouldn't seed for test splits
    perm = randperm(rng, n)

    nTrain = round(Int, trainFrac * n)
    nVal = round(Int, valFrac * n)

    trainIdx = perm[1:nTrain]
    valIdx = perm[nTrain+1:nTrain+nVal]
    testIdx = perm[nTrain+nVal+1:end]

    return (trainIdx=trainIdx, valIdx=valIdx, testIdx=testIdx)
end
