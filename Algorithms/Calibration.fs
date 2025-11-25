module Algorithms.Calibration

open System
open System.IO
open XisfLib.Core
open Algorithms.Statistics

// --- Types ---

/// Loaded master calibration frames
type MasterFrames = {
    BiasData: float[] option
    DarkData: float[] option
    FlatData: float[] option
    FlatMedian: float option
    Width: int
    Height: int
    Channels: int
    MaxPrecisionFormat: XisfSampleFormat
}

/// Configuration for calibration
type CalibrationConfig = {
    BiasFrame: string option
    BiasLevel: float option
    DarkFrame: string option
    FlatFrame: string option
    UncalibratedDark: bool
    UncalibratedFlat: bool
    OptimizeDark: bool
    OutputPedestal: int
}

/// Result of pixel calibration
type CalibrationResult = {
    CalibratedPixels: float[]
    ClippedPixels: int
    FlatZeroPixels: int
    DarkScale: float
}

// --- Format Precision ---

/// Get precision rank for sample format (higher = more precise)
let formatPrecision (format: XisfSampleFormat) =
    match format with
    | XisfSampleFormat.UInt8 -> 1
    | XisfSampleFormat.UInt16 -> 2
    | XisfSampleFormat.UInt32 -> 3
    | XisfSampleFormat.Float32 -> 4
    | XisfSampleFormat.Float64 -> 5
    | _ -> 0

/// Select the most precise format from a list
let maxPrecisionFormat (formats: XisfSampleFormat list) =
    formats |> List.maxBy formatPrecision

// --- Master Frame Loading ---

/// Load a single XISF frame as float array
let loadFrameAsFloat (path: string) : Async<float[] * int * int * int * XisfSampleFormat> =
    async {
        let reader = new XisfReader()
        let! metadata = reader.ReadAsync(path) |> Async.AwaitTask

        if metadata.Images.Count = 0 then
            failwithf "No images found in file: %s" path

        let img = metadata.Images.[0]
        let width = int img.Geometry.Width
        let height = int img.Geometry.Height
        let channels = int img.Geometry.ChannelCount
        let sampleFormat = img.SampleFormat

        let floatData = PixelIO.readPixelsAsFloat img

        return (floatData, width, height, channels, sampleFormat)
    }

/// Load master calibration frames from disk
let loadMasterFrames (config: CalibrationConfig) : Async<MasterFrames> =
    async {
        let mutable width = 0
        let mutable height = 0
        let mutable channels = 0
        let mutable biasData = None
        let mutable darkData = None
        let mutable flatData = None
        let mutable flatMedian = None
        let mutable formats = []

        // Load bias frame or use constant bias level
        match config.BiasFrame, config.BiasLevel with
        | Some path, _ ->
            let! (data, w, h, c, fmt) = loadFrameAsFloat path
            width <- w
            height <- h
            channels <- c
            biasData <- Some data
            formats <- fmt :: formats
        | None, Some _level ->
            // Constant bias level - will be applied during calibration
            biasData <- None
        | None, None ->
            ()

        // Load dark frame
        match config.DarkFrame with
        | Some path ->
            let! (data, w, h, c, fmt) = loadFrameAsFloat path
            if width > 0 && (w <> width || h <> height || c <> channels) then
                failwithf "Dark dimension mismatch: Expected %dx%dx%d, got %dx%dx%d" width height channels w h c
            width <- w
            height <- h
            channels <- c
            darkData <- Some data
            formats <- fmt :: formats
        | None ->
            ()

        // Load flat frame
        match config.FlatFrame with
        | Some path ->
            let! (data, w, h, c, fmt) = loadFrameAsFloat path
            if width > 0 && (w <> width || h <> height || c <> channels) then
                failwithf "Flat dimension mismatch: Expected %dx%dx%d, got %dx%dx%d" width height channels w h c
            width <- w
            height <- h
            channels <- c

            // Calculate median for normalization
            let median = calculateMedian data
            flatData <- Some data
            flatMedian <- Some median
            formats <- fmt :: formats
        | None ->
            ()

        // Default to UInt16 if no masters loaded (pedestal-only case)
        let maxFormat =
            if List.isEmpty formats then XisfSampleFormat.UInt16
            else maxPrecisionFormat formats

        return {
            BiasData = biasData
            DarkData = darkData
            FlatData = flatData
            FlatMedian = flatMedian
            Width = width
            Height = height
            Channels = channels
            MaxPrecisionFormat = maxFormat
        }
    }

// --- Dark Optimization ---

/// Golden section search to find optimal dark scaling factor
/// Minimizes noise in calibrated output
let findOptimalDarkScale
    (lightPixels: float[])
    (darkPixels: float[])
    (biasPixels: float[] option)
    (biasLevel: float option)
    (flatPixels: float[] option)
    (flatMedian: float option)
    (pedestal: int) : float =

    let goldenRatio = 0.618034  // (sqrt(5) - 1) / 2
    let tolerance = 0.001  // 1/1000 fractional accuracy
    let testSample = 5000  // use subset for speed

    let indices =
        if lightPixels.Length <= testSample then
            Array.init lightPixels.Length id
        else
            let step = lightPixels.Length / testSample
            Array.init testSample (fun i -> i * step)

    // Get bias value for a given pixel index
    let getBias idx =
        match biasPixels, biasLevel with
        | Some data, _ -> data.[idx]
        | None, Some level -> level
        | None, None -> 0.0

    // Calibrate with given dark scale k
    let calibrateWithScale k =
        indices
        |> Array.map (fun idx ->
            let lightVal = lightPixels.[idx]
            let darkVal = darkPixels.[idx]
            let biasVal = getBias idx
            let afterBias = lightVal - biasVal
            let afterDark = afterBias - (k * (darkVal - biasVal))
            let afterFlat =
                match flatPixels, flatMedian with
                | Some flatData, Some m when flatData.[idx] > 0.0 -> afterDark * (m / flatData.[idx])
                | _ -> afterDark
            max 0.0 (min 65535.0 (afterFlat + float pedestal))
        )
        |> estimateNoiseKSigma

    // Find bracketing interval
    let mutable k_lo = 0.0
    let mutable k_hi = 2.0
    let mutable noise_lo = calibrateWithScale k_lo
    let mutable noise_hi = calibrateWithScale k_hi

    // Expand bracket if needed
    let mutable iterations = 0
    while noise_hi < noise_lo && iterations < 20 do
        k_hi <- k_hi * 2.0
        noise_hi <- calibrateWithScale k_hi
        iterations <- iterations + 1

    // Golden section search
    let mutable a = k_lo
    let mutable b = k_hi
    let mutable fa = noise_lo
    let mutable fb = noise_hi

    while (b - a) / max (abs b) 1.0 > tolerance do
        let x1 = b - goldenRatio * (b - a)
        let x2 = a + goldenRatio * (b - a)

        let fx1 = calibrateWithScale x1
        let fx2 = calibrateWithScale x2

        if fx1 < fx2 then
            b <- x2
            fb <- fx2
        else
            a <- x1
            fa <- fx1

    (a + b) / 2.0

// --- Calibration Logic ---

/// Get bias value for a pixel index
let getBiasValue (masters: MasterFrames) (config: CalibrationConfig) (index: int) =
    match masters.BiasData, config.BiasLevel with
    | Some data, _ -> data.[index]
    | None, Some level -> level
    | None, None -> 0.0

/// Calibrate a single pixel value
let calibratePixel
    (lightValue: float)
    (biasValue: float)
    (darkValue: float option)
    (flatValue: float option)
    (flatMedian: float option)
    (pedestal: int)
    (uncalibratedDark: bool)
    (uncalibratedFlat: bool)
    (darkScale: float) : float =

    let afterBias = lightValue - biasValue

    let afterDark =
        match darkValue with
        | Some d ->
            // If dark is uncalibrated (raw), it contains bias that must be subtracted
            let effectiveDark = if uncalibratedDark then (d - biasValue) * darkScale else d * darkScale
            afterBias - effectiveDark
        | None -> afterBias

    let afterFlat =
        match flatValue, flatMedian with
        | Some f, Some m ->
            if f > 0.0 then
                let effectiveFlat =
                    if uncalibratedFlat then
                        // Flat is uncalibrated, need to calibrate it first
                        let flatAfterBias = f - biasValue
                        let flatAfterDark =
                            match darkValue with
                            | Some d ->
                                let effectiveDark = if uncalibratedDark then (d - biasValue) else d
                                flatAfterBias - effectiveDark
                            | None -> flatAfterBias
                        max 0.01 flatAfterDark  // Avoid division by zero
                    else
                        f
                afterDark * (m / effectiveFlat)
            else
                afterDark  // Skip division if flat pixel is zero
        | _ -> afterDark

    let final = afterFlat + float pedestal
    max 0.0 (min 65535.0 final)

/// Apply calibration to a light frame's pixels
let calibratePixels
    (lightPixels: float[])
    (masters: MasterFrames)
    (config: CalibrationConfig) : CalibrationResult =

    // Compute optimal dark scale if requested
    let darkScale =
        if config.OptimizeDark && masters.DarkData.IsSome then
            let darkPixels = masters.DarkData.Value
            findOptimalDarkScale lightPixels darkPixels masters.BiasData config.BiasLevel masters.FlatData masters.FlatMedian config.OutputPedestal
        else
            1.0

    let pixelCount = lightPixels.Length
    let calibratedFloats = Array.zeroCreate pixelCount
    let mutable zeroCount = 0
    let mutable flatZeroCount = 0

    for i = 0 to pixelCount - 1 do
        let lightValue = lightPixels.[i]
        let biasValue = getBiasValue masters config i
        let darkValue = masters.DarkData |> Option.map (fun data -> data.[i])
        let flatValue = masters.FlatData |> Option.map (fun data -> data.[i])

        if flatValue.IsSome && flatValue.Value = 0.0 then
            flatZeroCount <- flatZeroCount + 1

        let calibrated =
            calibratePixel
                lightValue biasValue darkValue flatValue masters.FlatMedian
                config.OutputPedestal config.UncalibratedDark config.UncalibratedFlat darkScale

        if calibrated = 0.0 then
            zeroCount <- zeroCount + 1

        calibratedFloats.[i] <- calibrated

    {
        CalibratedPixels = calibratedFloats
        ClippedPixels = zeroCount
        FlatZeroPixels = flatZeroCount
        DarkScale = darkScale
    }

// --- Metadata Generation ---

/// Build FITS HISTORY entries for calibration provenance
let buildCalibrationHistory (config: CalibrationConfig) =
    [
        yield "Calibrated with xisfprep"

        match config.BiasFrame with
        | Some path -> yield sprintf "Master bias: %s" (Path.GetFileName(path))
        | None -> ()

        match config.BiasLevel with
        | Some level -> yield sprintf "Bias level: %.0f" level
        | None -> ()

        match config.DarkFrame with
        | Some path ->
            yield sprintf "Master dark: %s" (Path.GetFileName(path))
            if config.UncalibratedDark then
                yield "Dark: uncalibrated (raw)"
            else
                yield "Dark: calibrated (bias-subtracted)"
        | None -> ()

        match config.FlatFrame with
        | Some path ->
            yield sprintf "Master flat: %s" (Path.GetFileName(path))
            if config.UncalibratedFlat then
                yield "Flat: uncalibrated (raw)"
            else
                yield "Flat: calibrated (bias/dark-subtracted)"
        | None -> ()

        if config.OutputPedestal <> 0 then
            yield sprintf "Output pedestal: %d" config.OutputPedestal
    ]

/// Build XISF properties for calibration metadata
let buildCalibrationProperties (config: CalibrationConfig) (darkScale: float) =
    [
        match config.BiasFrame with
        | Some path -> yield XisfStringProperty("Calibration:BiasFrame", path) :> XisfProperty
        | None -> ()

        match config.BiasLevel with
        | Some level -> yield XisfScalarProperty<float>("Calibration:BiasLevel", level) :> XisfProperty
        | None -> ()

        match config.DarkFrame with
        | Some path ->
            yield XisfStringProperty("Calibration:DarkFrame", path) :> XisfProperty
            yield XisfScalarProperty<bool>("Calibration:DarkCalibrated", not config.UncalibratedDark) :> XisfProperty
            if config.OptimizeDark then
                yield XisfScalarProperty<float>("Calibration:DarkScaleFactor", darkScale) :> XisfProperty
        | None -> ()

        match config.FlatFrame with
        | Some path ->
            yield XisfStringProperty("Calibration:FlatFrame", path) :> XisfProperty
            yield XisfScalarProperty<bool>("Calibration:FlatCalibrated", not config.UncalibratedFlat) :> XisfProperty
        | None -> ()

        if config.OutputPedestal <> 0 then
            yield XisfScalarProperty<int>("Calibration:OutputPedestal", config.OutputPedestal) :> XisfProperty
    ]

// --- Validation ---

/// Validate master frame dimensions match target image
let validateDimensions (masters: MasterFrames) (width: int) (height: int) (channels: int) : Result<unit, string> =
    if masters.Width > 0 && (width <> masters.Width || height <> masters.Height || channels <> masters.Channels) then
        Error (sprintf "Dimension mismatch: Expected %dx%dx%d, got %dx%dx%d"
            masters.Width masters.Height masters.Channels width height channels)
    else
        Ok ()
