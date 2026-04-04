# ══════════════════════════════════════════════════════════════════
#  model.jl — Per-neighbor graph convolution & correction network
#
#  Core layer: NaturalNeighborConv
#    5 independent weight matrices (W_self, W₁–W₄), one per
#    neighbor slot.  Neighbors are sorted by distance so each
#    W_k always applies to the k-th nearest neighbor.
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
    bias::AbstractVector{Float32}
    σ::F
end

Flux.@layer NaturalNeighborConv

function NaturalNeighborConv(ch::Pair{Int,Int}; σ=Flux.leakyrelu) # ch is the input and output channels
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

End-to-end network for redshift distortion correction with uncertainty.

The second output row is log(σ), the log-uncertainty.
Returns a tuple `(rCorr, logσ)`, each `1 × N`.
"""
struct RedshiftCorrectionNet{C1,C2,C3,N1,N2,D}
    conv1::C1
    norm1::N1
    conv2::C2
    norm2::N2
    conv3::C3
    dense::D
end

Flux.@layer RedshiftCorrectionNet

function (net::RedshiftCorrectionNet)(x::AbstractMatrix, neighborIdx::AbstractMatrix{<:Integer})
    h = net.conv1(x, neighborIdx)          # 3  → H,  LeakyReLU
    h = net.norm1(h)                       # LayerNorm
    h = net.conv2(h, neighborIdx)          # H  → H,  LeakyReLU
    h = net.norm2(h)                       # LayerNorm
    h = net.conv3(h, neighborIdx)           # H  → H,  LeakyReLU
    out = net.dense(h)                     # H  → 2,  linear
    δ = out[1:1, :]                     # correction term
    logσ = out[2:2, :]                     # log-uncertainty
    rObs = x[1:1, :]                       # extract r_obs (1st feature row)
    rCorr = rObs .+ δ                      # skip connection
    return (rCorr, logσ)                   # tuple of 1 × N each
end

"""
Build the default correction network with uncertainty output:
  3× NaturalNeighborConv (3→H→H→H, leakyrelu) + LayerNorm + Dense (H→2) + skip.
"""
function buildCorrectionNetwork(; firstHiddenDim::Int=32, secondHiddenDim::Int=64)
    RedshiftCorrectionNet(
        NaturalNeighborConv(3 => firstHiddenDim; σ=Flux.leakyrelu),
        Flux.LayerNorm(firstHiddenDim),
        NaturalNeighborConv(firstHiddenDim => secondHiddenDim; σ=Flux.leakyrelu),
        Flux.LayerNorm(secondHiddenDim),
        NaturalNeighborConv(secondHiddenDim => firstHiddenDim; σ=Flux.leakyrelu),
        Flux.Dense(firstHiddenDim, 2),   # 2 outputs: correction δ and logσ #check activtation = linear
    )
end
