module Algorithms.Self2SelfDenoise

open System
open TorchSharp
open Serilog

// ============================================================================
// SIMPLE CNN DENOISER
// ============================================================================

/// Simple CNN for denoising with normalized input/output
type SimpleDenoiser(numLayers: int, numFilters: int) as this =
    inherit torch.nn.Module<torch.Tensor, torch.Tensor>("SimpleDenoiser")

    let normalizationFactor = 65535.0f

    let layers =
        let modules = ResizeArray<torch.nn.Module<torch.Tensor, torch.Tensor>>()

        // First conv: 1 channel -> numFilters
        modules.Add(torch.nn.Conv2d(1L, int64 numFilters, 3L, padding=1L))
        modules.Add(torch.nn.ReLU())

        // Middle layers
        for _ in 2 .. numLayers - 1 do
            modules.Add(torch.nn.Conv2d(int64 numFilters, int64 numFilters, 3L, padding=1L))
            modules.Add(torch.nn.BatchNorm2d(int64 numFilters))
            modules.Add(torch.nn.ReLU())

        // Final conv: numFilters -> 1 (predict denoised image in [0,1] range)
        modules.Add(torch.nn.Conv2d(int64 numFilters, 1L, 3L, padding=1L))

        torch.nn.Sequential(modules)

    do
        this.RegisterComponents()

    override _.forward(input) =
        // Normalize input to [0, 1] range
        let normalized = input / normalizationFactor

        // Direct prediction: output denoised image in [0, 1] range
        let denoisedNormalized = layers.forward(normalized)

        // Denormalize back to original range
        denoisedNormalized * normalizationFactor

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
}

let defaultConfig = {
    NumEpochs = 100
    NumLayers = 5
    NumFilters = 48
    LearningRate = 0.001
    DropoutRate = 0.3
    UseGpu = true
}

// ============================================================================
// SELF2SELF TRAINING
// ============================================================================

/// Create random binary dropout mask
let private createDropoutMask (height: int64) (width: int64) (dropoutRate: float) (device: torch.Device) =
    use randTensor = torch.rand([|1L; 1L; height; width|], device=device)
    let threshold = TorchSharp.Scalar.op_Implicit(dropoutRate)
    randTensor.gt(threshold) // Returns 1 where pixel is kept, 0 where dropped

/// Single training step with random masking (Self2Self)
let private trainStep
    (model: SimpleDenoiser)
    (optimizer: torch.optim.Optimizer)
    (image: torch.Tensor)
    (dropoutRate: float) =

    optimizer.zero_grad()

    let height = image.shape.[2]
    let width = image.shape.[3]
    let device = image.device

    // Create dropout mask: 1 = keep pixel, 0 = drop pixel
    use keepMask = createDropoutMask height width dropoutRate device

    // Mask input: drop some pixels (set to 0)
    use maskedInput = image * keepMask

    // Forward pass: model tries to reconstruct full image from partial input
    use output = model.forward(maskedInput)

    // Compute MSE loss ONLY on dropped pixels (where keepMask = 0)
    // This forces the model to predict dropped pixels from neighbors
    use dropMask = keepMask.logical_not().to_type(torch.float32)
    use diff = (output - image) * dropMask
    let exponent = TorchSharp.Scalar.op_Implicit(2.0)
    use pixelLoss = diff.pow(exponent).mean()

    // Soft range penalty: encourage output to stay in [0, 65535] range (or [0, 1] normalized)
    // This doesn't hard-clip, just guides the model during training
    use outputNormalized = output / 65535.0f
    use tooLow = outputNormalized.clamp_max(0.0f).pow(exponent)
    use tooHigh = (outputNormalized - 1.0f).clamp_min(0.0f).pow(exponent)
    use rangePenalty = (tooLow.mean() + tooHigh.mean()) * 10.0f

    // Total loss
    use totalLoss = pixelLoss + rangePenalty

    // Backward pass
    totalLoss.backward()
    optimizer.step() |> ignore

    pixelLoss.ToSingle()  // Return pixel loss for logging (not penalty)

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

/// Core training loop on random patches
let private runTraining (model: SimpleDenoiser) (imageData2D: float32[,]) (config: TrainingConfig) (device: torch.Device) =
    let patchSize = 512
    use optimizer = torch.optim.AdamW(model.parameters(), config.LearningRate)
    let rng = Random()

    Log.Information("Training Self2Self model for {Epochs} epochs using {Size}x{Size} patches",
                   config.NumEpochs, patchSize, patchSize)
    model.train()

    for epoch in 1 .. config.NumEpochs do
        let patch = extractRandomPatch imageData2D patchSize rng
        use patchTensor = patchToTensor patch device

        try
            let loss = trainStep model optimizer patchTensor config.DropoutRate

            if epoch % 10 = 0 || epoch = 1 then
                Log.Information("Epoch {Epoch}/{Total}: Loss = {Loss:F6}", epoch, config.NumEpochs, loss)
        with ex ->
            Log.Error("Training failed at epoch {Epoch}: {Error}", epoch, ex.Message)
            Log.Error("Stack trace: {StackTrace}", ex.StackTrace)
            reraise()

    Log.Information("Training complete")

/// Train new Self2Self model from scratch
let trainModel (imageData: float[]) (width: int) (height: int) (config: TrainingConfig) : SimpleDenoiser =

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

    runTraining model imageData2D config device
    model

/// Continue training existing model
let continueTraining (existingModel: SimpleDenoiser) (imageData: float[]) (width: int) (height: int) (config: TrainingConfig) : SimpleDenoiser =

    let imageData2D = array1DTo2D imageData width height 0

    let device =
        if config.UseGpu && torch.cuda.is_available() then
            Log.Information("Using GPU (CUDA) for continued training")
            torch.CUDA
        else
            Log.Information("Using CPU for continued training")
            torch.CPU

    existingModel.``to``(device) |> ignore

    runTraining existingModel imageData2D config device
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

/// Process batch of tiles through model
let private denoiseTileBatch (model: SimpleDenoiser) (tiles: TileInfo list) (tileSize: int) (device: torch.Device) =
    let batchSize = tiles.Length

    // Stack tiles into batch tensor [B, 1, tileSize, tileSize]
    // Pad edge tiles to ensure uniform size
    use batchTensor = torch.zeros([| int64 batchSize; 1L; int64 tileSize; int64 tileSize |],
                                   dtype=torch.ScalarType.Float32, device=device)

    for i in 0 .. batchSize - 1 do
        let paddedTile = padTile tiles.[i].Tile tileSize
        use tileTensor = patchToTensor paddedTile device
        batchTensor.[int64 i] <- tileTensor.squeeze(0L)

    // Process entire batch in one forward pass
    use _ = torch.no_grad()
    use denoisedBatch = model.forward(batchTensor)
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
