module RedshiftDistortionCorrection

# ── Standard library ─────────────────────────────────────────────
using LinearAlgebra
using SparseArrays
using Statistics
using Random

# ── External dependencies ────────────────────────────────────────
using Flux
using TetGen
using JLD2
using ProgressMeter

# ── Submodules ───────────────────────────────────────────────────
include("graph.jl")
include("model.jl")
include("data.jl")
include("training.jl")
include("utils.jl")

# ── Public API ───────────────────────────────────────────────────
# Graph construction
export buildTessellation, extractNeighbors, buildNeighborIdx, buildGraph
export toSpherical

# Model
export NaturalNeighborConv, RedshiftCorrectionNet, buildCorrectionNetwork

# Data
export loadDataset, generateSyntheticData, splitData

# Training
export train!, saveCheckpoint, loadCheckpoint

# Utils
export computeMetrics, makeSectorData

end # module RedshiftDistortionCorrection