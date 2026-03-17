"""
Calculate mean and standard deviation for features and targets across the entire dataset.
"""
function calculateDatasetStats(fileList::Vector{String})
    allFeatures = []
    allTargets = []
    
    @showprogress "Calculating dataset stats… " for file in fileList
        data = loadDataset(file)
        push!(allFeatures, data.features)
        push!(allTargets, data.targets)
    end
    
    catFeatures = hcat(allFeatures...)
    catTargets = hcat(allTargets...)
    
    stats = (
        featMean = mean(catFeatures, dims=2),
        featStd  = std(catFeatures, dims=2) .+ 1f-8, # avoid div by zero
        targMean = mean(catTargets),
        targStd  = std(catTargets) + 1f-8
    )
    
    return stats
end

"""
Train the correction network on a dataset directory with data standardization.

# Arguments
- model       : `RedshiftCorrectionNet` 
- datasetDir  : directory containing `.jld2` files
- epochs      : number of training epochs
- lr          : learning rate for Adam
- lossFn      : loss `(ŷ, y) -> scalar`  (default: MSE)
- trainFrac   : fraction of data to use for training

# Returns
A dictionary with `trainLoss`, `testLoss` histories, and `stats`.
"""
function train!(model, datasetDir::String;
    epochs::Int=100, lr::Float64=1e-3,
    lossFn=Flux.mse, optimizer=Flux.Adam,
    trainFrac::Float64=0.8)

    fileList = filter(f -> endswith(f, ".jld2"), readdir(datasetDir; join=true))
    nFiles = length(fileList)
    
    if nFiles == 0
        error("No .jld2 files found in $datasetDir")
    end

    # Calculate dataset statistics for normalization
    stats = calculateDatasetStats(fileList)
    @info "Dataset Stats" stats.featMean stats.featStd stats.targMean stats.targStd

    # Split files into train and test
    splits = splitData(nFiles; trainFrac=trainFrac, valFrac=0.0)
    trainFiles = fileList[splits.trainIdx]
    testFiles = fileList[splits.testIdx]

    opt = Flux.setup(optimizer(lr), model)
    
    trainLossHistory = Float64[]
    testLossHistory = Float64[]

    @info "Starting training" nFiles nTrain=length(trainFiles) nTest=length(testFiles)

    p = Progress(epochs; desc="Overall Training… ", offset=0)

    for epoch in 1:epochs
        epochTrainLoss = 0.0
        
        # Shuffle training files each epoch
        shuffle!(trainFiles)

        pTrain = Progress(length(trainFiles); desc="  Epoch $epoch (Train): ", offset=1)

        for file in trainFiles
            data = loadDataset(file)
            
            # Normalize
            featNorm = (data.features .- stats.featMean) ./ stats.featStd
            targNorm = (data.targets .- stats.targMean) ./ stats.targStd
            
            lossVal, grads = Flux.withgradient(model) do m
                ŷ = m(featNorm, data.neighborIdx)
                lossFn(ŷ, targNorm)
            end

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
            
            # Normalize
            featNorm = (data.features .- stats.featMean) ./ stats.featStd
            targNorm = (data.targets .- stats.targMean) ./ stats.targStd
            
            ŷ = model(featNorm, data.neighborIdx)
            epochTestLoss += lossFn(ŷ, targNorm)
        end
        
        avgTestLoss = epochTestLoss / length(testFiles)
        push!(testLossHistory, avgTestLoss)
        
        # Update progress bar with metrics
        next!(p; showvalues = [
            (:epoch, epoch),
            (:trainLoss, avgTrainLoss),
            (:testLoss, avgTestLoss)
        ])
    end

    return (trainLoss=trainLossHistory, testLoss=testLossHistory, stats=stats)
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
