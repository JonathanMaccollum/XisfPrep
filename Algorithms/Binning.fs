module Algorithms.Binning

open Serilog

// --- Types ---

/// Binning methods for pixel aggregation
type BinningMethod =
    | Average
    | Median
    | Sum

    member this.Description =
        match this with
        | Average -> "Average binning (preserves flux)"
        | Median -> "Median binning (robust to outliers)"
        | Sum -> "Sum binning (adds pixel values)"

    override this.ToString() =
        match this with
        | Average -> "average"
        | Median -> "median"
        | Sum -> "sum"

/// Binning errors
type BinningError =
    | DimensionNotDivisible of width: int * height: int * factor: int
    | InvalidFactor of factor: int
    | InvalidDimensions of width: int * height: int * channels: int

    override this.ToString() =
        match this with
        | DimensionNotDivisible (w, h, f) ->
            $"Image dimensions ({w}x{h}) not evenly divisible by factor {f} - output will be truncated"
        | InvalidFactor f ->
            $"Invalid binning factor: {f}. Must be between 2 and 6"
        | InvalidDimensions (w, h, c) ->
            $"Invalid image dimensions: {w}x{h}x{c}"

/// Configuration for binning operation
type BinningConfig = {
    Factor: int
    Method: BinningMethod
}

/// Result of binning operation
type BinningResult = {
    BinnedPixels: float[]
    NewWidth: int
    NewHeight: int
    NewChannels: int
    TruncatedPixels: bool
}

// --- Validation ---

/// Validate binning configuration
let validateConfig (config: BinningConfig) : Result<unit, BinningError> =
    if config.Factor < 2 || config.Factor > 6 then
        Error (InvalidFactor config.Factor)
    else
        Ok ()

/// Validate image dimensions for binning
let validateDimensions (width: int) (height: int) (channels: int) (factor: int) : Result<bool, BinningError> =
    if width <= 0 || height <= 0 || channels <= 0 then
        Error (InvalidDimensions (width, height, channels))
    elif width < factor || height < factor then
        Error (InvalidDimensions (width, height, channels))
    else
        let truncated = (width % factor <> 0) || (height % factor <> 0)
        if truncated then
            Log.Warning("Image dimensions ({Width}x{Height}) not divisible by factor {Factor} - will truncate", width, height, factor)
        Ok truncated

// --- Core Binning Algorithm ---

/// Apply binning method to a block of pixels
let private applyBinningMethod (pixels: float array) (method: BinningMethod) : float =
    if Array.isEmpty pixels then
        0.0
    else
        match method with
        | Average ->
            Array.average pixels
        | Median ->
            let sorted = Array.sort pixels
            let mid = sorted.Length / 2
            if sorted.Length % 2 = 0 then
                (sorted.[mid - 1] + sorted.[mid]) / 2.0
            else
                sorted.[mid]
        | Sum ->
            // Clamp to prevent overflow in 16-bit range
            Array.sum pixels |> min 65535.0

/// Bin pixels from source image
/// Pixels are expected in row-major order: [c0, c1, c2, c0, c1, c2, ...]
let binPixels
    (pixels: float[])
    (width: int)
    (height: int)
    (channels: int)
    (config: BinningConfig) : BinningResult =

    let factor = config.Factor
    let newWidth = width / factor
    let newHeight = height / factor

    // Get pixel value at specific position and channel
    let getPixel x y channel =
        let offset = (y * width + x) * channels + channel
        pixels.[offset]

    let binnedFloats = Array.zeroCreate (newWidth * newHeight * channels)

    // Pre-allocate block buffer and reuse it (performance optimization)
    let blockPixels = Array.zeroCreate (factor * factor)

    for newY = 0 to newHeight - 1 do
        for newX = 0 to newWidth - 1 do
            for ch = 0 to channels - 1 do
                // Collect pixels from source block
                let mutable pixelCount = 0
                for dy in 0 .. factor - 1 do
                    for dx in 0 .. factor - 1 do
                        let srcX = newX * factor + dx
                        let srcY = newY * factor + dy
                        // Handle edge cases where dimensions don't divide evenly
                        if srcX < width && srcY < height then
                            blockPixels.[pixelCount] <- getPixel srcX srcY ch
                            pixelCount <- pixelCount + 1

                // Apply binning method to collected block
                let binnedValue =
                    if pixelCount = 0 then
                        0.0
                    else
                        applyBinningMethod blockPixels.[0 .. pixelCount - 1] config.Method

                let offset = (newY * newWidth + newX) * channels + ch
                binnedFloats.[offset] <- binnedValue

    let truncated = (width % factor <> 0) || (height % factor <> 0)

    {
        BinnedPixels = binnedFloats
        NewWidth = newWidth
        NewHeight = newHeight
        NewChannels = channels
        TruncatedPixels = truncated
    }
