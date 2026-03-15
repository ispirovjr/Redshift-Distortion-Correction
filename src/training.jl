# ══════════════════════════════════════════════════════════════════
#  training.jl — Training loop, checkpointing
# ══════════════════════════════════════════════════════════════════

"""
Train the correction network on a single graph.

# Arguments
- model       : `RedshiftCorrectionNet` 
- features    : 3 × N Float32 input   (r_obs, θ, φ)
- neighborIdx : 4 × N Int             (sorted neighbor indices)
- targets     : 1 × N Float32         (R_true)
- epochs      : number of training epochs
- lr          : learning rate for Adam
- lossFn      : loss `(ŷ, y) -> scalar`  (default: MSE)

# Returns
A vector of per-epoch loss values.
"""
function train!(model, features::AbstractMatrix, neighborIdx::AbstractMatrix{<:Integer},
    targets::AbstractMatrix;
    epochs::Int=100, lr::Float64=1e-3,
    lossFn=Flux.mse, optimizer=Flux.Adam)

    opt = Flux.setup(optimizer(lr), model)
    lossHistory = Float64[]

    @showprogress "Training… " for epoch in 1:epochs
        lossVal, grads = Flux.withgradient(model) do m
            ŷ = m(features, neighborIdx)
            lossFn(ŷ, targets)
        end

        Flux.update!(opt, model, grads[1])
        push!(lossHistory, lossVal)
    end

    return lossHistory
end

# ── Checkpointing ────────────────────────────────────────────────

"""
Save model parameters to a JLD2 file.
"""
function saveCheckpoint(path::String, model; metadata::Dict=Dict())
    JLD2.jldsave(path;
        params=Flux.state(model),
        metadata=metadata,
    )
    @info "Checkpoint saved" path
end

"""
Load parameters from a JLD2 checkpoint into an existing model.
"""
function loadCheckpoint(path::String, model)
    checkpoint = JLD2.load(path)
    Flux.loadmodel!(model, checkpoint["params"])
    @info "Checkpoint loaded" path
    return model
end
