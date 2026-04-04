"""
Heteroscedastic Gaussian NLL loss with logσ floor.

    loss = mean( ((target - pred) / σ)^2 + 2 * log(σ) )

where σ = exp(logσ) guarantees positivity.
logσ is clamped to a minimum of `LOG_SIGMA_FLOOR` to prevent
numerical instability from extremely small σ values.
"""
const LOG_SIGMA_FLOOR = -4.0f0  # σ ≥ exp(-4) ≈ 0.018

function uncertaintyLoss(pred, logσ, target)
    logσClamped = max.(logσ, LOG_SIGMA_FLOOR)
    return mean(((target .- pred) ./ exp.(logσClamped)) .^ 2 .+ 2 .* logσClamped)
end

"""
Normalize features and targets for a single observation file.

Divides all radii (r_obs in features[1,:] and r_true in targets[1,:])
by the maximum observed radius in that file.  Angles are left unchanged.

Returns `(featNorm, targNorm, rMax)`.
"""
function normalizePerFile(features, targets)
    rMax = maximum(features[1, :])
    featNorm = copy(features)
    featNorm[1, :] ./= rMax
    targNorm = copy(targets)
    targNorm[1, :] ./= rMax
    return featNorm, targNorm, rMax
end

"""
Cosine annealing learning rate schedule.

Returns the learning rate for a given epoch:
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * max(0, t - warmup) / (T - warmup)))

The schedule holds lr_max constant for the first `warmupFrac` of epochs,
then cosine-decays to `lr_min`. This prevents premature decay.

# Arguments
- `epoch`      : current epoch (1-indexed)
- `totalEpochs`: total number of epochs
- `lrMax`      : initial (maximum) learning rate
- `lrMin`      : final (minimum) learning rate (default: lrMax/50)
- `warmupFrac` : fraction of epochs at constant lrMax before decay begins (default: 0.3)
"""
function cosineAnnealLR(epoch::Int, totalEpochs::Int, lrMax::Float64;
    lrMin::Float64=lrMax / 50.0, warmupFrac::Float64=0.3)
    warmupEpochs = round(Int, warmupFrac * totalEpochs)
    if epoch <= warmupEpochs
        return lrMax
    end
    t = (epoch - warmupEpochs) / (totalEpochs - warmupEpochs)
    return lrMin + 0.5 * (lrMax - lrMin) * (1.0 + cos(π * t))
end

"""
Clip gradient norms in-place to a maximum value.

Returns the original (unclipped) gradient norm for logging.
"""
function clipGradNorm!(grads, maxNorm::Float64=1.0)
    totalNorm = 0.0
    for x in Flux.trainables(grads)
        if x !== nothing
            totalNorm += sum(abs2, x)
        end
    end
    totalNorm = sqrt(totalNorm)
    if totalNorm > maxNorm
        scale = Float32(maxNorm / totalNorm)
        for x in Flux.trainables(grads)
            if x !== nothing
                x .*= scale
            end
        end
    end
    return totalNorm
end

"""
Train the correction network on a dataset directory with per-file normalization.

# Arguments
- model       : `RedshiftCorrectionNet`
- datasetDir  : directory containing `.jld2` files
- epochs      : number of training epochs
- lr          : initial learning rate for Adam (peak of cosine schedule)
- trainFrac   : fraction of data to use for training
- maxGradNorm : maximum gradient norm for clipping (default 1.0)

# Returns
A NamedTuple with `trainLoss` and `testLoss` epoch histories.
"""
function train!(model, datasetDir::String;
    epochs::Int=100, lr::Float64=1e-3,
    optimizer=Flux.Adam,
    trainFrac::Float64=0.8,
    maxGradNorm::Float64=1.0)

    fileList = filter(f -> endswith(f, ".jld2"), readdir(datasetDir; join=true))
    nFiles = length(fileList)

    if nFiles == 0
        error("No .jld2 files found in $datasetDir")
    end

    # Split files into train and test
    splits = splitData(nFiles; trainFrac=trainFrac, valFrac=0.0)
    trainFiles = fileList[splits.trainIdx]
    testFiles = fileList[splits.testIdx]

    opt = Flux.setup(optimizer(lr), model)

    trainLossHistory = Float64[]
    testLossHistory = Float64[]

    @info "Starting training" nFiles nTrain = length(trainFiles) nTest = length(testFiles) hiddenDim = length(model.norm1.diag.scale)

    p = Progress(epochs; desc="Overall Training… ", offset=0)

    for epoch in 1:epochs
        epochTrainLoss = 0.0
        epochMaxGradNorm = 0.0

        # Cosine annealing LR — update optimizer
        currentLR = cosineAnnealLR(epoch, epochs, lr)
        Flux.adjust!(opt, currentLR)

        # Shuffle training files each epoch
        shuffle!(trainFiles)

        pTrain = Progress(length(trainFiles); desc="  Epoch $epoch (Train): ", offset=1)

        for file in trainFiles
            data = loadDataset(file)

            # Per-file normalization
            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            lossVal, grads = Flux.withgradient(model) do m
                pred, logσ = m(featNorm, data.neighborIdx)
                uncertaintyLoss(pred, logσ, targNorm)
            end

            # Gradient clipping
            gradNorm = clipGradNorm!(grads[1], maxGradNorm)
            epochMaxGradNorm = max(epochMaxGradNorm, gradNorm)

            Flux.update!(opt, model, grads[1])
            epochTrainLoss += lossVal

            next!(pTrain; showvalues=[(:currentFileLoss, lossVal)])
        end

        avgTrainLoss = epochTrainLoss / length(trainFiles)
        push!(trainLossHistory, avgTrainLoss)

        # Calculate test loss
        epochTestLoss = 0.0
        for file in testFiles
            data = loadDataset(file)

            # Per-file normalization
            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            pred, logσ = model(featNorm, data.neighborIdx)
            epochTestLoss += uncertaintyLoss(pred, logσ, targNorm)
        end

        avgTestLoss = epochTestLoss / length(testFiles)
        push!(testLossHistory, avgTestLoss)


        # Update progress bar with metrics
        next!(p; showvalues=[
            (:epoch, epoch),
            (:trainLoss, avgTrainLoss),
            (:testLoss, avgTestLoss),
            (:lr, currentLR),
            (:maxGradNorm, round(epochMaxGradNorm, digits=2))
        ])
    end

    return (trainLoss=trainLossHistory, testLoss=testLossHistory)
end

function trainSilent!(model, datasetDir::String;
    epochs::Int=100, lr::Float64=1e-3,
    optimizer=Flux.Adam,
    trainFrac::Float64=0.8)

    fileList = filter(f -> endswith(f, ".jld2"), readdir(datasetDir; join=true))
    nFiles = length(fileList)

    if nFiles == 0
        error("No .jld2 files found in $datasetDir")
    end

    # Split files into train and test
    splits = splitData(nFiles; trainFrac=trainFrac, valFrac=0.0)
    trainFiles = fileList[splits.trainIdx]
    testFiles = fileList[splits.testIdx]

    opt = Flux.setup(optimizer(lr), model)

    trainLossHistory = Float64[]
    testLossHistory = Float64[]

    for epoch in 1:epochs
        epochTrainLoss = 0.0

        currentLR = cosineAnnealLR(epoch, epochs, lr)
        Flux.adjust!(opt, currentLR)

        shuffle!(trainFiles)


        for file in trainFiles
            data = loadDataset(file)

            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            lossVal, grads = Flux.withgradient(model) do m
                pred, logσ = m(featNorm, data.neighborIdx)
                uncertaintyLoss(pred, logσ, targNorm)
            end


            Flux.update!(opt, model, grads[1])
            epochTrainLoss += lossVal

        end

        avgTrainLoss = epochTrainLoss / length(trainFiles)
        push!(trainLossHistory, avgTrainLoss)


        epochTestLoss = 0.0
        for file in testFiles
            data = loadDataset(file)


            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            pred, logσ = model(featNorm, data.neighborIdx)
            epochTestLoss += uncertaintyLoss(pred, logσ, targNorm)
        end

        avgTestLoss = epochTestLoss / length(testFiles)
        push!(testLossHistory, avgTestLoss)

    end

    return (trainLoss=trainLossHistory, testLoss=testLossHistory)
end


function trainGPU!(model, datasetDir::String;
    epochs::Int=100, lr::Float64=1e-3,
    optimizer=Flux.Adam,
    trainFrac::Float64=0.8) # Added a flag to toggle GPU

    fileList = filter(f -> endswith(f, ".jld2"), readdir(datasetDir; join=true))
    nFiles = length(fileList)

    if nFiles == 0
        error("No .jld2 files found in $datasetDir")
    end

    # Determine device and cast model to Float32 (GPUs are much faster with 32-bit floats)
    device = CUDA.functional() ? gpu : cpu
    model = model |> f32 |> device

    # Split files into train and test
    splits = splitData(nFiles; trainFrac=trainFrac, valFrac=0.0)
    trainFiles = fileList[splits.trainIdx]
    testFiles = fileList[splits.testIdx]

    # Optimizer state will automatically sit on the GPU if the model is on the GPU
    opt = Flux.setup(optimizer(lr), model)

    trainLossHistory = Float64[]
    testLossHistory = Float64[]

    for epoch in 1:epochs
        epochTrainLoss = 0.0

        currentLR = cosineAnnealLR(epoch, epochs, lr)
        Flux.adjust!(opt, currentLR)

        shuffle!(trainFiles)

        for file in trainFiles
            data = loadDataset(file)

            # Normalization remains on CPU (usually fine unless matrices are massive)
            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            # Move data to GPU and ensure it is Float32
            featD = featNorm |> f32 |> device
            targD = targNorm |> f32 |> device
            neighD = data.neighborIdx |> device # Assuming this is an integer array

            lossVal, grads = Flux.withgradient(model) do m
                pred, logσ = m(featD, neighD)
                uncertaintyLoss(pred, logσ, targD)
            end

            Flux.update!(opt, model, grads[1])

            # CRITICAL: Convert loss back to a standard CPU Float64.
            # If you leave it as a CuArray scalar, it will slow down your loop.
            epochTrainLoss += Float64(lossVal)
        end

        avgTrainLoss = epochTrainLoss / length(trainFiles)
        push!(trainLossHistory, avgTrainLoss)

        epochTestLoss = 0.0
        for file in testFiles
            data = loadDataset(file)

            featNorm, targNorm, _ = normalizePerFile(data.features, data.targets)

            # Move test data to GPU
            featD = featNorm |> f32 |> device
            targD = targNorm |> f32 |> device
            neighD = data.neighborIdx |> device

            pred, logσ = model(featD, neighD)

            # Evaluate loss and bring it back to CPU
            batchLoss = uncertaintyLoss(pred, logσ, targD)
            epochTestLoss += Float64(batchLoss)
        end

        avgTestLoss = epochTestLoss / length(testFiles)
        push!(testLossHistory, avgTestLoss)
    end

    # Return model to CPU if desired at the end (optional)
    # model = model |> cpu

    return (trainLoss=trainLossHistory, testLoss=testLossHistory)
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
    return (model=model, metadata=get(checkpoint, "metadata", Dict()))
end
