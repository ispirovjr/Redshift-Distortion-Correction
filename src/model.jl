# ══════════════════════════════════════════════════════════════════
#  model.jl — Per-neighbor graph convolution & correction network
#
#  Core layer: NaturalNeighborConv
#    5 independent weight matrices (W_self, W₁–W₄), one per
#    neighbor slot.  Neighbors are sorted by distance so each
#    W_k always applies to the k-th nearest neighbor.
#
#  Network:  2× NNConv  →  Dense  →  skip(r_obs)  →  R_corr
# ══════════════════════════════════════════════════════════════════

# ── NaturalNeighborConv ──────────────────────────────────────────

"""
Graph convolution for fixed-degree Delaunay graphs

Has 5 independent weight matrices:
- `wSelf`      : applied to each node's own features
- `w1` … `w4`  : applied to each of its natural neighbors (sorted by distance)

Forward pass for node `i`:

    hᵢ' = σ( wSelf(hᵢ) + w1(hⱼ₁) + w2(hⱼ₂) + w3(hⱼ₃) + w4(hⱼ₄) + b )

where `j₁ … j₄` are the neighbors of `i` sorted by ascending distance.

# Arguments
- `x`           : F × N feature matrix
- `neighborIdx` : 4 × N matrix of neighbor indices (sorted by distance)
"""
struct NaturalNeighborConv{F,D} #F- Function, D- Dense layer
    wSelf::D
    w1::D
    w2::D
    w3::D
    w4::D
    bias::Vector{Float32}
    σ::F
end

Flux.@layer NaturalNeighborConv

function NaturalNeighborConv(ch::Pair{Int,Int}; σ=Flux.relu) # ch is the input and output channels
    inCh, outCh = ch
    NaturalNeighborConv(
        Flux.Dense(inCh, outCh; bias=false),  # wSelf
        Flux.Dense(inCh, outCh; bias=false),  # w1
        Flux.Dense(inCh, outCh; bias=false),  # w2
        Flux.Dense(inCh, outCh; bias=false),  # w3
        Flux.Dense(inCh, outCh; bias=false),  # w4
        zeros(Float32, outCh),                # bias
        σ,                                    # activation function
    )
end

function (l::NaturalNeighborConv)(x::AbstractMatrix, neighborIdx::AbstractMatrix{<:Integer})
    hSelf = l.wSelf(x)
    h1 = l.w1(x[:, @view(neighborIdx[1, :])])
    h2 = l.w2(x[:, @view(neighborIdx[2, :])])
    h3 = l.w3(x[:, @view(neighborIdx[3, :])])
    h4 = l.w4(x[:, @view(neighborIdx[4, :])])

    return l.σ.(hSelf .+ h1 .+ h2 .+ h3 .+ h4 .+ l.bias)
end

# ── RedshiftCorrectionNet ────────────────────────────────────────

"""
    RedshiftCorrectionNet

End-to-end network for redshift distortion correction.

Architecture:
    (r_obs, θ, φ)  →  NNConv₁(3→H)  →  NNConv₂(H→H)  →  Dense(H→1)
                                                               ↓
                         r_obs ──── skip connection ──────→  (+)  →  ReLU  →  R_corr

The skip connection adds the original `r_obs` (1st input feature) to the
dense output, so the network only needs to learn the correction term.
"""
struct RedshiftCorrectionNet{C1,C2,D}
    conv1::C1
    conv2::C2
    dense::D
end

Flux.@layer RedshiftCorrectionNet

function (net::RedshiftCorrectionNet)(x::AbstractMatrix, neighborIdx::AbstractMatrix{<:Integer})
    h = net.conv1(x, neighborIdx)          # 3  → H,  ReLU
    h = net.conv2(h, neighborIdx)          # H  → H,  ReLU
    δ = net.dense(h)                       # H  → 1,  linear
    rObs = x[1:1, :]                          # extract r_obs (1st feature row)
    rCorr = Flux.relu.(rObs .+ δ)              # skip + positivity
    return rCorr                               # 1 × N
end

"""
Build the default correction network:
  2× NaturalNeighborConv (3→H→H, ReLU)  +  Dense (H→1)  +  skip.
"""
function buildCorrectionNetwork(; hiddenDim::Int=32)
    RedshiftCorrectionNet(
        NaturalNeighborConv(3 => hiddenDim; σ=Flux.relu),
        NaturalNeighborConv(hiddenDim => hiddenDim; σ=Flux.relu),
        Flux.Dense(hiddenDim, 1),   # linear output (correction δ)
    )
end
