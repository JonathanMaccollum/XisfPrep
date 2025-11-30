module Algorithms.Self2SelfDenoise

open System
open TorchSharp
open Serilog

// ============================================================================
// U-NET DENOISER (DIRECT PREDICTION)
// ============================================================================

/// Mini U-Net for Self2Self denoising. Predicts denoised IMAGE directly (not noise residual).
/// Input: Z-score normalized tensor with masked pixels (Self2Self dropout)
/// Output: Predicted denoised image in Z-score space (linear, unbounded)
/// Architecture: Encoder → Bottleneck → Decoder with skip connections
type SimpleDenoiser(numLayers: int, numFilters: int) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("SimpleUNet")

    // Encoder: Extract features and downsample
    let enc1: TorchSharp.Modules.Sequential =
        torch.nn.Sequential(
            torch.nn.Conv2d(1L, int64 numFilters, 3L, padding=1L),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int64 numFilters, int64 numFilters, 3L, padding=1L),
            torch.nn.ReLU()
        )

    let pool: TorchSharp.Modules.MaxPool2d = torch.nn.MaxPool2d(2L, 2L)

    // Bottleneck: Process at lower resolution with more channels
    let bottleneck: TorchSharp.Modules.Sequential =
        torch.nn.Sequential(
            torch.nn.Conv2d(int64 numFilters, int64 (numFilters * 2), 3L, padding=1L),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int64 (numFilters * 2), int64 (numFilters * 2), 3L, padding=1L),
            torch.nn.ReLU()
        )

    // Decoder: Upsample and combine with skip connection
    let upsample: TorchSharp.Modules.Upsample = torch.nn.Upsample(scale_factor=[|2.0; 2.0|], mode=torch.UpsampleMode.Bilinear, align_corners=false)

    let dec1: TorchSharp.Modules.Sequential =
        torch.nn.Sequential(
            // Input: (numFilters*2 from bottleneck) + (numFilters from skip) = numFilters*3
            torch.nn.Conv2d(int64 (numFilters * 3), int64 numFilters, 3L, padding=1L),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int64 numFilters, 1L, 3L, padding=1L)
            // NO Sigmoid - output is in Z-score space, can be any value
        )

    do
        this.RegisterComponents()

    override _.forward(input) =
        // Encoder
        use e1 = enc1.forward(input)
        use p1 = pool.forward(e1)

        // Bottleneck
        use b = bottleneck.forward(p1)

        // Decoder
        use u1 = upsample.forward(b)

        // Skip connection: concatenate along channel dimension
        use cat = torch.cat([| u1; e1 |], 1L)

        // Final prediction: denoised image in Z-score space
        dec1.forward(cat)

// ============================================================================
// TRAINING CONFIG
// ============================================================================

type TrainingConfig = {
    NumEpochs: int
    NumLayers: int
    NumFilters: int
    LearningRate: float
    DropoutRate: float
    UseGpu: bool
    PatchesPerEpoch: int      // Number of patches to extract per epoch
    BatchSize: int             // Number of patches per training batch
    ValidationSplit: float     // Fraction of patches for validation (e.g., 0.1)
    CheckpointEvery: int       // Save checkpoint every N epochs
    EarlyStoppingPatience: int // Stop if no improvement for N epochs
}

let defaultConfig = {
    NumEpochs = 100
    NumLayers = 5
    NumFilters = 48
    LearningRate = 0.001
    DropoutRate = 0.5
    UseGpu = true
    PatchesPerEpoch = 200      // 200 patches per epoch (was 1!)
    BatchSize = 16             // Process 16 patches at once
    ValidationSplit = 0.1      // 10% for validation
    CheckpointEvery = 50       // Save every 50 epochs
    EarlyStoppingPatience = 100 // Stop if no improvement for 100 epochs
}

// ============================================================================
// SELF2SELF TRAINING
// ============================================================================

/// Create random binary dropout mask
let private createDropoutMask (height: int64) (width: int64) (dropoutRate: float) (device: torch.Device) =
    use randTensor = torch.rand([|1L; 1L; height; width|], device=device)
    let threshold = TorchSharp.Scalar.op_Implicit(dropoutRate)
    randTensor.gt(threshold) // Returns 1 where pixel is kept, 0 where dropped

/// Train on a batch of patches with random masking (Self2Self)
let private trainBatch
    (model: SimpleDenoiser)
    (optimizer: torch.optim.Optimizer)
    (batchTensor: torch.Tensor)
    (dropoutRate: float)
    (debugFirstBatch: bool) =

    optimizer.zero_grad()

    let batchSize = batchTensor.shape.[0]
    let height = batchTensor.shape.[2]
    let width = batchTensor.shape.[3]
    let device = batchTensor.device

    // Z-score normalization based on batch statistics (robust to astrophotography dynamic range)
    use batchMean = batchTensor.mean()
    use batchStd = batchTensor.std()
    let epsilon = TorchSharp.Scalar.op_Implicit(1e-6f)
    use stdWithEps = batchStd + epsilon
    use normalized = (batchTensor - batchMean) / stdWithEps

    // Create dropout masks for entire batch
    use keepMask = createDropoutMask height width dropoutRate device
    use keepMaskBatch = keepMask.expand([|batchSize; 1L; height; width|])

    // Mask normalized input: dropped pixels set to 0 (Self2Self corruption)
    use maskedInput = normalized * keepMaskBatch

    // Forward pass: DIRECT PREDICTION of denoised image
    // Model sees masked input and predicts what the full image should be
    use predictedImage = model.forward(maskedInput)

    // Compute MSE loss ONLY on dropped pixels
    // Compare prediction to original (noisy) values at dropped locations
    // (Self2Self: expectation of noise is the clean signal)
    use dropMask = keepMaskBatch.logical_not().to_type(torch.float32)
    use diff = (predictedImage - normalized) * dropMask
    let exponent = TorchSharp.Scalar.op_Implicit(2.0)

    // Normalize loss by number of dropped pixels (not all pixels)
    use squaredDiff = diff.pow(exponent)
    use pixelLoss = squaredDiff.sum() / dropMask.sum()

    if debugFirstBatch then
        Log.Information("  [DEBUG] Input range: [{Min:F2}, {Max:F2}]", batchTensor.min().ToSingle(), batchTensor.max().ToSingle())
        Log.Information("  [DEBUG] Z-score stats: mean={Mean:F2}, std={Std:F2}", batchMean.ToSingle(), batchStd.ToSingle())
        Log.Information("  [DEBUG] Normalized range: [{Min:F4}, {Max:F4}]", normalized.min().ToSingle(), normalized.max().ToSingle())
        Log.Information("  [DEBUG] Predicted image range: [{Min:F4}, {Max:F4}]", predictedImage.min().ToSingle(), predictedImage.max().ToSingle())
        Log.Information("  [DEBUG] Dropout mask sum: {Sum} / {Total} pixels", dropMask.sum().ToInt32(), dropMask.numel())
        Log.Information("  [DEBUG] Loss value: {Loss:F8}", pixelLoss.ToSingle())

    // Backward pass
    pixelLoss.backward()
    optimizer.step() |> ignore

    pixelLoss.ToSingle()

/// Validate on a batch of patches (no gradient update)
let private validateBatch
    (model: SimpleDenoiser)
    (batchTensor: torch.Tensor)
    (dropoutRate: float) =

    let batchSize = batchTensor.shape.[0]
    let height = batchTensor.shape.[2]
    let width = batchTensor.shape.[3]
    let device = batchTensor.device

    use _ = torch.no_grad()

    // Z-score normalization based on batch statistics
    use batchMean = batchTensor.mean()
    use batchStd = batchTensor.std()
    let epsilon = TorchSharp.Scalar.op_Implicit(1e-6f)
    use stdWithEps = batchStd + epsilon
    use normalized = (batchTensor - batchMean) / stdWithEps

    use keepMask = createDropoutMask height width dropoutRate device
    use keepMaskBatch = keepMask.expand([|batchSize; 1L; height; width|])
    use maskedInput = normalized * keepMaskBatch

    // Forward pass: DIRECT PREDICTION
    use predictedImage = model.forward(maskedInput)

    // Compute MSE loss on dropped pixels only
    use dropMask = keepMaskBatch.logical_not().to_type(torch.float32)
    use diff = (predictedImage - normalized) * dropMask
    let exponent = TorchSharp.Scalar.op_Implicit(2.0)

    // Normalize by number of dropped pixels
    use squaredDiff = diff.pow(exponent)
    use pixelLoss = squaredDiff.sum() / dropMask.sum()

    pixelLoss.ToSingle()

/// Convert 1D float array (channel-planar) to 2D float32 array
let private array1DTo2D (data: float[]) (width: int) (height: int) (channelIndex: int) : float32[,] =
    let channelSize = width * height
    let offset = channelIndex * channelSize
    let result = Array2D.zeroCreate<float32> height width

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let idx = offset + y * width + x
            result.[y, x] <- float32 data.[idx]

    result

/// Convert 2D float32 array back to 1D float array (single channel)
let private array2DTo1D (data: float32[,]) : float[] =
    let height = Array2D.length1 data
    let width = Array2D.length2 data
    let result = Array.zeroCreate (width * height)

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let idx = y * width + x
            result.[idx] <- float data.[y, x]

    result

/// Extract random patch from 2D image
let private extractRandomPatch (imageData2D: float32[,]) (patchSize: int) (rng: Random) =
    let height = Array2D.length1 imageData2D
    let width = Array2D.length2 imageData2D

    let maxY = height - patchSize
    let maxX = width - patchSize

    let startY = rng.Next(0, maxY)
    let startX = rng.Next(0, maxX)

    Array2D.init patchSize patchSize (fun y x -> imageData2D.[startY + y, startX + x])

/// Convert 2D patch to tensor [1, 1, H, W] on device
let private patchToTensor (patch: float32[,]) (device: torch.Device) =
    torch.tensor(patch, dtype=torch.ScalarType.Float32, device=device)
        .unsqueeze(0L)
        .unsqueeze(1L)

/// Extract multiple random patches from image
let private extractRandomPatches (imageData2D: float32[,]) (patchSize: int) (numPatches: int) (rng: Random) =
    [| for _ in 1 .. numPatches do
        yield extractRandomPatch imageData2D patchSize rng |]

/// Create batch tensor from array of patches
let private patchesToBatchTensor (patches: float32[,][]) (device: torch.Device) =
    let batchSize = patches.Length
    let patchSize = Array2D.length1 patches.[0]

    use batchTensor = torch.zeros([| int64 batchSize; 1L; int64 patchSize; int64 patchSize |],
                                   dtype=torch.ScalarType.Float32, device=device)

    for i in 0 .. batchSize - 1 do
        use patchTensor = patchToTensor patches.[i] device
        batchTensor.[int64 i] <- patchTensor.squeeze(0L)

    batchTensor.clone()

/// Core training loop with batching, validation, and checkpointing
let private runTraining (model: SimpleDenoiser) (imageData2D: float32[,]) (config: TrainingConfig) (device: torch.Device) (modelPath: string) =
    let patchSize = 512
    use optimizer = torch.optim.AdamW(model.parameters(), config.LearningRate)
    let rng = Random()

    Log.Information("Production Self2Self Training:")
    Log.Information("  Epochs: {Epochs}", config.NumEpochs)
    Log.Information("  Patches per epoch: {Patches}", config.PatchesPerEpoch)
    Log.Information("  Batch size: {BatchSize}", config.BatchSize)
    Log.Information("  Patch size: {Size}×{Size}", patchSize, patchSize)
    Log.Information("  Validation split: {Split:P0}", config.ValidationSplit)
    Log.Information("  Total training samples per epoch: {Total}", config.PatchesPerEpoch * patchSize * patchSize)

    // Calculate train/val split
    let numValPatches = int (float config.PatchesPerEpoch * config.ValidationSplit)
    let numTrainPatches = config.PatchesPerEpoch - numValPatches

    Log.Information("  Training patches: {Train}, Validation patches: {Val}", numTrainPatches, numValPatches)

    let mutable bestValLoss = System.Single.MaxValue
    let mutable epochsSinceImprovement = 0
    let mutable bestEpoch = 0
    let mutable shouldStop = false
    let mutable epoch = 1

    while epoch <= config.NumEpochs && not shouldStop do
        // Extract patches for this epoch
        let allPatches = extractRandomPatches imageData2D patchSize config.PatchesPerEpoch rng
        let trainPatches = allPatches.[0 .. numTrainPatches - 1]
        let valPatches = allPatches.[numTrainPatches ..]

        // Training phase
        model.train()
        let mutable trainLossSum = 0.0f
        let mutable trainBatches = 0

        for batchStart in 0 .. config.BatchSize .. numTrainPatches - 1 do
            let batchEnd = min (batchStart + config.BatchSize - 1) (numTrainPatches - 1)
            let batchPatches = trainPatches.[batchStart .. batchEnd]

            use batchTensor = patchesToBatchTensor batchPatches device
            let debugThisBatch = (epoch = 1 && trainBatches = 0)
            let loss = trainBatch model optimizer batchTensor config.DropoutRate debugThisBatch

            trainLossSum <- trainLossSum + loss
            trainBatches <- trainBatches + 1

        let avgTrainLoss = trainLossSum / float32 trainBatches

        // Validation phase
        model.eval()
        let mutable valLossSum = 0.0f
        let mutable valBatches = 0

        for batchStart in 0 .. config.BatchSize .. numValPatches - 1 do
            let batchEnd = min (batchStart + config.BatchSize - 1) (numValPatches - 1)
            let batchPatches = valPatches.[batchStart .. batchEnd]

            use batchTensor = patchesToBatchTensor batchPatches device
            let loss = validateBatch model batchTensor config.DropoutRate

            valLossSum <- valLossSum + loss
            valBatches <- valBatches + 1

        let avgValLoss = valLossSum / float32 valBatches

        // Logging every epoch
        Log.Information("Epoch {Epoch}/{Total}: TrainLoss={Train:F6}, ValLoss={Val:F6}",
                       epoch, config.NumEpochs, avgTrainLoss, avgValLoss)

        // Track best model
        if avgValLoss < bestValLoss then
            bestValLoss <- avgValLoss
            bestEpoch <- epoch
            epochsSinceImprovement <- 0

            // Save best model
            let bestModelPath = modelPath.Replace(".pt", "_best.pt")
            model.save(bestModelPath) |> ignore
            Log.Information("  → New best model! ValLoss={Loss:F6} (saved to {Path})", avgValLoss, System.IO.Path.GetFileName bestModelPath)
        else
            epochsSinceImprovement <- epochsSinceImprovement + 1

        // Checkpoint saving
        if epoch % config.CheckpointEvery = 0 then
            let checkpointPath = modelPath.Replace(".pt", $"_epoch{epoch}.pt")
            model.save(checkpointPath) |> ignore
            Log.Information("  Checkpoint saved: {Path}", System.IO.Path.GetFileName checkpointPath)

        // Early stopping
        if epochsSinceImprovement >= config.EarlyStoppingPatience then
            Log.Information("Early stopping: No improvement for {Patience} epochs", config.EarlyStoppingPatience)
            Log.Information("Best validation loss: {Loss:F6} at epoch {Epoch}", bestValLoss, bestEpoch)
            shouldStop <- true

        epoch <- epoch + 1

    Log.Information("Training complete!")
    Log.Information("Best model: Epoch {Epoch}, ValLoss={Loss:F6}", bestEpoch, bestValLoss)

/// Train new Self2Self model from scratch
let trainModel (imageData: float[]) (width: int) (height: int) (config: TrainingConfig) (modelPath: string) : SimpleDenoiser =

    let imageData2D = array1DTo2D imageData width height 0

    let device =
        if config.UseGpu && torch.cuda.is_available() then
            Log.Information("Using GPU (CUDA) for training")
            torch.CUDA
        else
            Log.Information("Using CPU for training")
            torch.CPU

    let model = new SimpleDenoiser(config.NumLayers, config.NumFilters)
    model.``to``(device) |> ignore

    runTraining model imageData2D config device modelPath
    model

/// Continue training existing model
let continueTraining (existingModel: SimpleDenoiser) (imageData: float[]) (width: int) (height: int) (config: TrainingConfig) (modelPath: string) : SimpleDenoiser =

    let imageData2D = array1DTo2D imageData width height 0

    let device =
        if config.UseGpu && torch.cuda.is_available() then
            Log.Information("Using GPU (CUDA) for continued training")
            torch.CUDA
        else
            Log.Information("Using CPU for continued training")
            torch.CPU

    existingModel.``to``(device) |> ignore

    runTraining existingModel imageData2D config device modelPath
    existingModel

/// Extract tile from image with bounds checking
let private extractTile (imageData2D: float32[,]) (startY: int) (startX: int) (tileSize: int) =
    let imgHeight = Array2D.length1 imageData2D
    let imgWidth = Array2D.length2 imageData2D

    let endY = min (startY + tileSize) imgHeight
    let endX = min (startX + tileSize) imgWidth

    let actualHeight = endY - startY
    let actualWidth = endX - startX

    Array2D.init actualHeight actualWidth (fun y x -> imageData2D.[startY + y, startX + x])

/// Tile with position info for batch processing
type private TileInfo = {
    Tile: float32[,]
    StartY: int
    StartX: int
    OriginalHeight: int
    OriginalWidth: int
}

/// Pad tile to target size (for edge tiles)
let private padTile (tile: float32[,]) (targetSize: int) =
    let height = Array2D.length1 tile
    let width = Array2D.length2 tile

    if height = targetSize && width = targetSize then
        tile
    else
        Array2D.init targetSize targetSize (fun y x ->
            if y < height && x < width then tile.[y, x] else 0.0f)

/// Process batch of tiles through model with Self2Self test-time averaging
let private denoiseTileBatch (model: SimpleDenoiser) (tiles: TileInfo list) (tileSize: int) (device: torch.Device) =
    let batchSize = tiles.Length
    let numPasses = 10  // Average 10 masked predictions per Self2Self paper
    let dropoutRate = 0.5

    // Stack tiles into batch tensor [B, 1, tileSize, tileSize]
    // Pad edge tiles to ensure uniform size
    use batchTensor = torch.zeros([| int64 batchSize; 1L; int64 tileSize; int64 tileSize |],
                                   dtype=torch.ScalarType.Float32, device=device)

    for i in 0 .. batchSize - 1 do
        let paddedTile = padTile tiles.[i].Tile tileSize
        use tileTensor = patchToTensor paddedTile device
        batchTensor.[int64 i] <- tileTensor.squeeze(0L)

    use _ = torch.no_grad()

    // Z-score normalization based on batch statistics (compute once, reuse for all passes)
    use batchMean = batchTensor.mean()
    use batchStd = batchTensor.std()
    let epsilon = TorchSharp.Scalar.op_Implicit(1e-6f)
    use stdWithEps = batchStd + epsilon
    use normalized = (batchTensor - batchMean) / stdWithEps

    // Self2Self inference: average multiple masked predictions
    // Simple averaging strategy as per expert recommendation
    use accumulator = torch.zeros_like(normalized)

    let mutable passNum = 0
    for _ in 1 .. numPasses do
        passNum <- passNum + 1

        // Create random dropout mask (same as training)
        use keepMask = createDropoutMask (int64 tileSize) (int64 tileSize) dropoutRate device
        use keepMaskBatch = keepMask.expand([| int64 batchSize; 1L; int64 tileSize; int64 tileSize |])

        // Mask normalized input
        use maskedInput = normalized * keepMaskBatch

        // Forward pass: DIRECT PREDICTION of denoised image
        use predictedImage = model.forward(maskedInput)

        // DEBUG first pass of first batch
        if passNum = 1 && batchSize > 0 then
            Log.Information("  [INFERENCE DEBUG] Pass 1:")
            Log.Information("    Input range: [{Min:F2}, {Max:F2}]", batchTensor.min().ToSingle(), batchTensor.max().ToSingle())
            Log.Information("    Z-score stats: mean={Mean:F2}, std={Std:F2}", batchMean.ToSingle(), batchStd.ToSingle())
            Log.Information("    Normalized range: [{Min:F4}, {Max:F4}]", normalized.min().ToSingle(), normalized.max().ToSingle())
            Log.Information("    Predicted image range: [{Min:F4}, {Max:F4}]", predictedImage.min().ToSingle(), predictedImage.max().ToSingle())

        // Accumulate predictions in Z-score space
        accumulator.add_(predictedImage) |> ignore

    // Average the predictions in Z-score space
    let avgFactor = TorchSharp.Scalar.op_Implicit(float32 numPasses)
    use denoisedNormalized = accumulator / avgFactor

    // Denormalize to pixel space
    use denoisedBatch = denoisedNormalized * batchStd + batchMean

    Log.Information("  [INFERENCE DEBUG] After averaging:")
    Log.Information("    Accumulator range: [{Min:F2}, {Max:F2}]", accumulator.min().ToSingle(), accumulator.max().ToSingle())
    Log.Information("    Final output range: [{Min:F2}, {Max:F2}]", denoisedBatch.min().ToSingle(), denoisedBatch.max().ToSingle())

    use denoisedCpu = denoisedBatch.cpu()

    // Unpack results and crop back to original size
    [for i in 0 .. batchSize - 1 do
        let origHeight = tiles.[i].OriginalHeight
        let origWidth = tiles.[i].OriginalWidth
        let result = Array2D.zeroCreate<float32> origHeight origWidth
        for y in 0 .. origHeight - 1 do
            for x in 0 .. origWidth - 1 do
                result.[y, x] <- denoisedCpu.[int64 i, 0L, int64 y, int64 x].ToSingle()
        yield (result, tiles.[i].StartY, tiles.[i].StartX)]

/// Add denoised tile to accumulation buffers with blending
let private accumulateTile (denoisedTile: float32[,]) (startY: int) (startX: int)
                          (accumulator: float32[,]) (countMap: int[,]) =
    let tileHeight = Array2D.length1 denoisedTile
    let tileWidth = Array2D.length2 denoisedTile

    for y in 0 .. tileHeight - 1 do
        for x in 0 .. tileWidth - 1 do
            let imgY = startY + y
            let imgX = startX + x
            accumulator.[imgY, imgX] <- accumulator.[imgY, imgX] + denoisedTile.[y, x]
            countMap.[imgY, imgX] <- countMap.[imgY, imgX] + 1

/// Apply trained model using overlapping tiles with batch processing
let applyModel (model: SimpleDenoiser) (imageData: float[]) (width: int) (height: int) (useGpu: bool) : float[] =

    let tileSize = 512
    let overlap = 64
    let stride = tileSize - overlap
    let batchSize = 16

    let imageData2D = array1DTo2D imageData width height 0

    let device =
        if useGpu && torch.cuda.is_available() then
            Log.Information("Using GPU for inference")
            torch.CUDA
        else
            Log.Information("Using CPU for inference")
            torch.CPU

    model.``to``(device) |> ignore
    model.eval()

    // Calculate tile grid
    let tilesY = (height + stride - 1) / stride
    let tilesX = (width + stride - 1) / stride
    let totalTiles = tilesY * tilesX

    Log.Information("Processing image in {Count} tiles ({TilesX}×{TilesY}, size={TileSize}, overlap={Overlap}, batch={BatchSize})",
                   totalTiles, tilesX, tilesY, tileSize, overlap, batchSize)

    // Accumulation buffers for blending
    let accumulator = Array2D.zeroCreate<float32> height width
    let countMap = Array2D.zeroCreate<int> height width

    // Collect all tiles
    let allTiles = [
        for tileY in 0 .. tilesY - 1 do
            for tileX in 0 .. tilesX - 1 do
                let startY = tileY * stride
                let startX = tileX * stride
                let tile = extractTile imageData2D startY startX tileSize
                yield {
                    Tile = tile
                    StartY = startY
                    StartX = startX
                    OriginalHeight = Array2D.length1 tile
                    OriginalWidth = Array2D.length2 tile
                }
    ]

    // Process tiles in batches
    let mutable processedTiles = 0
    let batches = allTiles |> List.chunkBySize batchSize

    for batch in batches do
        let denoisedBatch = denoiseTileBatch model batch tileSize device

        for (denoisedTile, startY, startX) in denoisedBatch do
            accumulateTile denoisedTile startY startX accumulator countMap

        processedTiles <- processedTiles + batch.Length
        if processedTiles % (batchSize * 5) = 0 || processedTiles = totalTiles then
            Log.Information("  Processed {Count}/{Total} tiles", processedTiles, totalTiles)

    // Average overlapping regions
    let result2D = Array2D.zeroCreate<float32> height width
    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let count = countMap.[y, x]
            result2D.[y, x] <- if count > 0 then accumulator.[y, x] / float32 count else 0.0f

    array2DTo1D result2D

/// Save trained model to file
let saveModel (model: SimpleDenoiser) (path: string) : unit =
    Log.Information("Saving model to {Path}", path)
    model.save(path) |> ignore
    Log.Information("Model saved successfully")

/// Load model from file
let loadModel (path: string) (numLayers: int) (numFilters: int) : SimpleDenoiser =
    Log.Information("Loading model from {Path}", path)
    let model = new SimpleDenoiser(numLayers, numFilters)
    model.load(path) |> ignore |> ignore
    Log.Information("Model loaded successfully")
    model
