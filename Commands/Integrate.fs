module Commands.Integrate

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Statistics

// --- Type Definitions ---

type CombinationMethod =
    | Average
    | Median

type NormalizationMethod =
    | NoNormalization
    | Additive
    | Multiplicative
    | AdditiveWithScaling
    | MultiplicativeWithScaling

type RejectionMethod =
    | NoRejection
    | MinMaxClipping of lowCount: int * highCount: int
    | SigmaClipping of lowSigma: float * highSigma: float
    | LinearFitClipping of lowSigma: float * highSigma: float

type RejectionNormalization =
    | NoRejectionNormalization
    | ScaleAndZeroOffset
    | EqualizeFluxes

type IntegrationSettings = {
    Combination: CombinationMethod
    Normalization: NormalizationMethod
    Rejection: RejectionMethod
    RejectionNormalization: RejectionNormalization
    Iterations: int
}

type InlineCalibrationConfig = {
    BiasFrame: string option
    BiasLevel: float option
    DarkFrame: string option
    UncalibratedDark: bool
    Pedestal: int
}

// --- Defaults ---
let private defaultCombination = Average
let private defaultNormalization = Multiplicative
let private defaultRejection = NoRejection
let private defaultRejectionNormalization = NoRejectionNormalization
let private defaultLowSigma = 2.5
let private defaultHighSigma = 2.0
let private defaultIterations = 3
// ---

let showHelp() =
    printfn "integrate - Stack/combine multiple images with rejection and normalization"
    printfn ""
    printfn "Usage: xisfprep integrate [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files to stack (wildcards supported)"
    printfn "  --output, -o <file>       Output integrated file path"
    printfn ""
    printfn "Optional:"
    printfn $"  --combination <method>    Pixel combination method (default: {defaultCombination.ToString().ToLower()})"
    printfn "                              average - Mean combination"
    printfn "                              median  - Median combination"
    printfn $"  --normalization <method>  Image normalization (default: {defaultNormalization.ToString().ToLower()})"
    printfn "                              none                    - No normalization"
    printfn "                              additive                - P' = P + (K - m_i)"
    printfn "                              multiplicative          - P' = P * (K / m_i)"
    printfn "                              additive-scaling        - P' = P*s_i + (K - m_i)"
    printfn "                              multiplicative-scaling  - P' = P*s_i * (K / m_i)"
    printfn "  --rejection <algorithm>   Pixel rejection algorithm (default: none)"
    printfn "                              none      - No rejection"
    printfn "                              minmax    - Min/max clipping (requires --low-count, --high-count)"
    printfn "                              sigma     - Iterative sigma clipping (requires --low-sigma, --high-sigma)"
    printfn "                              linearfit - Linear fit clipping (requires --low-sigma, --high-sigma)"
    printfn "  --rejection-norm <method> Normalize pixels before rejection stats (default: none)"
    printfn "                              none            - Use raw pixel values for statistics"
    printfn "                              scale-offset    - (P - mean) / stddev normalization"
    printfn "                              equalize-flux   - P / median normalization"
    printfn $"  --low-sigma <value>       Low rejection threshold (default: {defaultLowSigma})"
    printfn $"  --high-sigma <value>      High rejection threshold (default: {defaultHighSigma})"
    printfn "  --low-count <n>           Drop N lowest pixels (for minmax, default: 1)"
    printfn "  --high-count <n>          Drop N highest pixels (for minmax, default: 1)"
    printfn $"  --iterations <n>          Rejection iterations (default: {defaultIterations})"
    printfn "  --overwrite               Overwrite existing output file (default: skip with warning)"
    printfn "  --output-format <format>  Output sample format (default: float32 for stacking precision)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Inline Calibration (for master dark/flat creation):"
    printfn "  --bias, -b <file>         Master bias frame for inline calibration"
    printfn "  --bias-level <value>      Constant bias value (alternative to --bias)"
    printfn "  --dark, -d <file>         Master dark frame for inline calibration"
    printfn "  --uncalibrated-dark       Dark is raw (not bias-subtracted)"
    printfn "  --pedestal <value>        Pedestal added after calibration [0-65535] (default: 0)"
    printfn ""
    printfn "Process:"
    printfn "  1. Load all input images into memory"
    printfn "  2. Calculate per-channel statistics for normalization"
    printfn "  3. Process each pixel position across image stack in parallel:"
    printfn "     - Apply normalization"
    printfn "     - Apply rejection algorithm"
    printfn "     - Combine surviving pixels"
    printfn "  4. Output stacked image with integration metadata"
    printfn ""
    printfn "Validation:"
    printfn "  - All images must have same dimensions"
    printfn "  - All images must have same channel count"
    printfn "  - All images must have same sample format"
    printfn "  - Minimum 3 images required for rejection algorithms"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep integrate -i \"subs/*.xisf\" -o \"master.xisf\""
    printfn "  xisfprep integrate -i \"subs/*.xisf\" -o \"master.xisf\" --rejection linearfit"
    printfn "  xisfprep integrate -i \"bias/*.xisf\" -o \"master_bias.xisf\" --normalization none --rejection sigma"

type private ParseState = {
    Input: string option
    Output: string option
    Combination: CombinationMethod option
    Normalization: NormalizationMethod option
    Rejection: RejectionMethod option
    RejectionNormalization: RejectionNormalization option
    LowSigma: float option
    HighSigma: float option
    LowCount: int option
    HighCount: int option
    Iterations: int option
    Overwrite: bool
    OutputFormat: XisfSampleFormat option
    // Inline calibration
    BiasFrame: string option
    BiasLevel: float option
    DarkFrame: string option
    UncalibratedDark: bool
    Pedestal: int option
}

let parseArgs (args: string array) =
    let initialState : ParseState = {
        Input = None
        Output = None
        Combination = None
        Normalization = None
        Rejection = None
        RejectionNormalization = None
        LowSigma = None
        HighSigma = None
        LowCount = None
        HighCount = None
        Iterations = None
        Overwrite = false
        OutputFormat = None
        // Inline calibration
        BiasFrame = None
        BiasLevel = None
        DarkFrame = None
        UncalibratedDark = false
        Pedestal = None
    }

    let rec parse (state: ParseState) (args: string list) =
        match args with
        // The base case now contains ALL validation and default logic
        | [] ->
            // Validate required arguments
            let input = match state.Input with Some v -> v | None -> failwith "Required argument: --input"
            let output = match state.Output with Some v -> v | None -> failwith "Required argument: --output"

            // Apply defaults for all optional values
            let combination = state.Combination |> Option.defaultValue defaultCombination
            let normalization = state.Normalization |> Option.defaultValue defaultNormalization
            let rejectionNormalization = state.RejectionNormalization |> Option.defaultValue defaultRejectionNormalization
            let lowSigma = state.LowSigma |> Option.defaultValue defaultLowSigma
            let highSigma = state.HighSigma |> Option.defaultValue defaultHighSigma
            let lowCount = state.LowCount |> Option.defaultValue 1
            let highCount = state.HighCount |> Option.defaultValue 1
            let iterations = state.Iterations |> Option.defaultValue defaultIterations

            // Update rejection with actual sigma/count values
            let rejection =
                match state.Rejection with
                | Some (MinMaxClipping _) -> MinMaxClipping (lowCount, highCount)
                | Some (SigmaClipping _) -> SigmaClipping (lowSigma, highSigma)
                | Some (LinearFitClipping _) -> LinearFitClipping (lowSigma, highSigma)
                | Some NoRejection -> NoRejection
                | None -> defaultRejection

            // Validation
            if iterations < 1 then
                failwith "Iterations must be at least 1"

            // Inline calibration validation
            if state.BiasFrame.IsSome && state.BiasLevel.IsSome then
                failwith "--bias and --bias-level are mutually exclusive"

            if state.UncalibratedDark && state.DarkFrame.IsNone then
                failwith "--uncalibrated-dark requires --dark"

            if state.UncalibratedDark && state.BiasFrame.IsNone && state.BiasLevel.IsNone then
                failwith "--uncalibrated-dark requires --bias or --bias-level"

            let pedestal = state.Pedestal |> Option.defaultValue 0
            if pedestal < 0 || pedestal > 65535 then
                failwith "Pedestal must be in range [0, 65535]"

            // Build the final settings record
            let settings = {
                Combination = combination
                Normalization = normalization
                Rejection = rejection
                RejectionNormalization = rejectionNormalization
                Iterations = iterations
            }

            // Build inline calibration config
            let inlineCalibration =
                if state.BiasFrame.IsSome || state.BiasLevel.IsSome || state.DarkFrame.IsSome || pedestal > 0 then
                    Some {
                        BiasFrame = state.BiasFrame
                        BiasLevel = state.BiasLevel
                        DarkFrame = state.DarkFrame
                        UncalibratedDark = state.UncalibratedDark
                        Pedestal = pedestal
                    }
                else
                    None

            // Return the 6-tuple that the 'run' function expects
            (input, output, settings, state.Overwrite, state.OutputFormat, inlineCalibration)

        // Recursive calls are now clean and unambiguous
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse { state with Input = Some value } rest
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse { state with Output = Some value } rest
        | "--combination" :: value :: rest ->
            let c = match value.ToLower() with
                    | "average" -> Average
                    | "median" -> Median
                    | _ -> failwithf "Unknown combination method: %s" value
            parse { state with Combination = Some c } rest
        | "--normalization" :: value :: rest ->
            let n = match value.ToLower() with
                    | "none" -> NoNormalization
                    | "additive" -> Additive
                    | "multiplicative" -> Multiplicative
                    | "additive-scaling" -> AdditiveWithScaling
                    | "multiplicative-scaling" -> MultiplicativeWithScaling
                    | _ -> failwithf "Unknown normalization method: %s" value
            parse { state with Normalization = Some n } rest
        | "--rejection" :: value :: rest ->
            let r = match value.ToLower() with
                    | "none" -> NoRejection
                    | "minmax" -> MinMaxClipping (0, 0) // Placeholder, corrected in base case
                    | "sigma" -> SigmaClipping (0.0, 0.0) // Placeholder
                    | "linearfit" -> LinearFitClipping (0.0, 0.0) // Placeholder
                    | _ -> failwithf "Unknown rejection method: %s" value
            parse { state with Rejection = Some r } rest
        | "--rejection-norm" :: value :: rest ->
            let rn = match value.ToLower() with
                     | "none" -> NoRejectionNormalization
                     | "scale-offset" -> ScaleAndZeroOffset
                     | "equalize-flux" -> EqualizeFluxes
                     | _ -> failwithf "Unknown rejection normalization: %s" value
            parse { state with RejectionNormalization = Some rn } rest
        | "--low-sigma" :: value :: rest ->
            parse { state with LowSigma = Some (float value) } rest
        | "--high-sigma" :: value :: rest ->
            parse { state with HighSigma = Some (float value) } rest
        | "--low-count" :: value :: rest ->
            parse { state with LowCount = Some (int value) } rest
        | "--high-count" :: value :: rest ->
            parse { state with HighCount = Some (int value) } rest
        | "--iterations" :: value :: rest ->
            parse { state with Iterations = Some (int value) } rest
        | "--overwrite" :: rest ->
            parse { state with Overwrite = true } rest
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> parse { state with OutputFormat = Some fmt } rest
            | None -> failwithf "Unknown output format: %s (supported: uint8, uint16, uint32, float32, float64)" value
        // Inline calibration
        | "--bias" :: value :: rest | "-b" :: value :: rest ->
            parse { state with BiasFrame = Some value } rest
        | "--bias-level" :: value :: rest ->
            parse { state with BiasLevel = Some (float value) } rest
        | "--dark" :: value :: rest | "-d" :: value :: rest ->
            parse { state with DarkFrame = Some value } rest
        | "--uncalibrated-dark" :: rest ->
            parse { state with UncalibratedDark = true } rest
        | "--pedestal" :: value :: rest ->
            parse { state with Pedestal = Some (int value) } rest
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    // The initial call is now simple and type-safe
    parse initialState (List.ofArray args)

// --- Statistics Module ---
module Stats =
    let calculate (data: float[]) =
        if Array.isEmpty data then (0.0, 0.0, 0.0)
        else
            let n = float data.Length
            let mean = Array.average data
            let stdDev =
                data
                |> Array.sumBy (fun x -> (x - mean) * (x - mean))
                |> (fun sumSq -> sqrt (sumSq / n))

            let sorted = Array.copy data
            Array.sortInPlace sorted
            let median =
                let mid = data.Length / 2
                if data.Length % 2 = 0 then
                    (sorted.[mid - 1] + sorted.[mid]) / 2.0
                else
                    sorted.[mid]

            (mean, median, stdDev)

// --- Linear Fit Module ---
module LinearFit =
    let fitLine (points: (float * float)[]) : float * float * float[] =
        if Array.isEmpty points then (0.0, 0.0, [||])
        else
            let n = float points.Length
            let sumX = points |> Array.sumBy fst
            let sumY = points |> Array.sumBy snd
            let sumXY = points |> Array.sumBy (fun (x, y) -> x * y)
            let sumXX = points |> Array.sumBy (fun (x, _) -> x * x)

            let slope =
                let denominator = n * sumXX - sumX * sumX
                if abs denominator < 1e-10 then 0.0
                else (n * sumXY - sumX * sumY) / denominator

            let intercept = (sumY - slope * sumX) / n

            let residuals =
                points
                |> Array.map (fun (x, y) -> y - (slope * x + intercept))

            (slope, intercept, residuals)

    let rejectOutliers (values: float[]) (lowSigma: float) (highSigma: float) (iterations: int) : bool[] =
        let n = values.Length

        if n = 0 then [||]
        else
            let indexedValues = Array.mapi (fun i v -> (i, v)) values

            let rec iterate (currentPoints: (int * float)[]) (iter: int) =
                if iter >= iterations || Array.isEmpty currentPoints then
                    currentPoints
                else
                    let fitData = currentPoints |> Array.map (fun (i, v) -> (float i, v))
                    let (slope, intercept, residuals) = fitLine fitData

                    let (meanRes, _, stdDevRes) = Stats.calculate residuals

                    if stdDevRes = 0.0 then
                        currentPoints
                    else
                        let lowThreshold = meanRes - lowSigma * stdDevRes
                        let highThreshold = meanRes + highSigma * stdDevRes

                        let nextPoints =
                            Array.zip currentPoints residuals
                            |> Array.choose (fun ((idx, value), res) ->
                                if res >= lowThreshold && res <= highThreshold
                                then Some (idx, value)
                                else None)

                        if nextPoints.Length < currentPoints.Length then
                            iterate nextPoints (iter + 1)
                        else
                            currentPoints

            let surviving = iterate indexedValues iterations

            let survivingSet = surviving |> Array.map fst |> Set.ofArray
            Array.init n (fun i -> survivingSet.Contains i)

// --- Inline Calibration Module ---
module InlineCalibration =
    type LoadedMasters = {
        BiasData: float[] option
        DarkData: float[] option
        Width: int
        Height: int
        Channels: int
    }

    let loadFrameAsFloat (path: string) : Async<float[] * int * int * int> =
        async {
            let reader = new XisfReader()
            let! metadata = reader.ReadAsync(path) |> Async.AwaitTask

            if metadata.Images.Count = 0 then
                failwithf "No images found in file: %s" path

            let img = metadata.Images.[0]
            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount
            let floatData = PixelIO.readPixelsAsFloat img

            return (floatData, width, height, channels)
        }

    let loadMasters (config: InlineCalibrationConfig) : Async<LoadedMasters option> =
        async {
            let mutable width = 0
            let mutable height = 0
            let mutable channels = 0
            let mutable biasData = None
            let mutable darkData = None

            // Load bias frame
            match config.BiasFrame with
            | Some path ->
                Log.Information("Loading master bias for inline calibration: {Path}", path)
                let! (data, w, h, c) = loadFrameAsFloat path
                width <- w
                height <- h
                channels <- c
                biasData <- Some data
            | None -> ()

            // Load dark frame
            match config.DarkFrame with
            | Some path ->
                Log.Information("Loading master dark for inline calibration: {Path}", path)
                let! (data, w, h, c) = loadFrameAsFloat path
                if width > 0 && (w <> width || h <> height || c <> channels) then
                    failwithf "Dark dimension mismatch: Expected %dx%dx%d, got %dx%dx%d" width height channels w h c
                width <- w
                height <- h
                channels <- c
                darkData <- Some data
            | None -> ()

            if biasData.IsNone && darkData.IsNone && config.BiasLevel.IsNone && config.Pedestal = 0 then
                return None
            else
                return Some {
                    BiasData = biasData
                    DarkData = darkData
                    Width = width
                    Height = height
                    Channels = channels
                }
        }

    let calibratePixels (pixels: float[]) (masters: LoadedMasters) (config: InlineCalibrationConfig) : float[] =
        let result = Array.zeroCreate pixels.Length

        for i = 0 to pixels.Length - 1 do
            let lightValue = pixels.[i]

            // Get bias value
            let biasValue =
                match masters.BiasData, config.BiasLevel with
                | Some data, _ -> data.[i]
                | None, Some level -> level
                | None, None -> 0.0

            let afterBias = lightValue - biasValue

            // Apply dark subtraction
            let afterDark =
                match masters.DarkData with
                | Some darkData ->
                    let darkValue = darkData.[i]
                    if config.UncalibratedDark then
                        // Dark contains bias, subtract it
                        afterBias - (darkValue - biasValue)
                    else
                        afterBias - darkValue
                | None -> afterBias

            // Add pedestal and clamp
            let final = afterDark + float config.Pedestal
            result.[i] <- max 0.0 (min 65535.0 final)

        result

// --- Main Integration Logic ---

let validateImages (images: XisfImage[]) =
    let first = images.[0]
    let width = first.Geometry.Width
    let height = first.Geometry.Height
    let channels = first.Geometry.ChannelCount
    let sampleFormat = first.SampleFormat

    images |> Array.iteri (fun i img ->
        if img.Geometry.Width <> width || img.Geometry.Height <> height then
            failwithf "Image %d has different dimensions: %dx%d (expected %dx%d)" i img.Geometry.Width img.Geometry.Height width height
        if img.Geometry.ChannelCount <> channels then
            failwithf "Image %d has different channel count: %d (expected %d)" i img.Geometry.ChannelCount channels
        if img.SampleFormat <> sampleFormat then
            failwithf "Image %d has different sample format: %A (expected %A)" i img.SampleFormat sampleFormat
    )

let calculateImageStats (pixelArrays: float[][]) (channels: int) (pixelCount: int) =
    printfn "Calculating per-channel image statistics for normalization..."

    let getImgStats (pixelData: float[]) =
        Array.init channels (fun ch ->
            let values = Array.init pixelCount (fun pix ->
                pixelData.[pix * channels + ch]
            )
            let (mean, median, _) = Stats.calculate values
            (mean, median)
        )

    let allStats = pixelArrays |> Array.Parallel.map getImgStats
    let refStats = allStats.[0]
    let map = allStats |> Array.mapi (fun i stats -> (i, stats)) |> Map.ofArray

    (map, refStats)

let processPixel
    (pixelArrays: float[][])
    (pix: int)
    (channels: int)
    (combination: CombinationMethod)
    (normalization: NormalizationMethod)
    (rejection: RejectionMethod)
    (rejectionNormalization: RejectionNormalization)
    (iterations: int)
    (imageStats: Map<int, (float * float)[]>)
    (referenceStats: (float * float)[])
    : float[] =

    Array.init channels (fun channel ->
        let pixelIndex = pix * channels + channel
        let (refMean, refMedian) = referenceStats.[channel]

        // Get normalized pixel stack
        let rawPixelValues =
            pixelArrays
            |> Array.mapi (fun i arr ->
                let rawValue = arr.[pixelIndex]

                match normalization with
                | NoNormalization -> rawValue
                | _ ->
                    let (imgMean, imgMedian) = imageStats.[i].[channel]
                    let m_i = if imgMedian = 0.0 then 1.0 else imgMedian
                    let s_i = if imgMean = 0.0 then 1.0 else refMean / imgMean

                    match normalization with
                    | Additive -> rawValue + (refMedian - imgMedian)
                    | Multiplicative -> rawValue * (refMedian / m_i)
                    | AdditiveWithScaling -> (rawValue * s_i) + (refMedian - imgMedian)
                    | MultiplicativeWithScaling -> (rawValue * s_i) * (refMedian / m_i)
                    | _ -> rawValue
            )

        // Apply rejection
        let pixelsToAverage =
            match rejection with
            | NoRejection -> rawPixelValues

            | MinMaxClipping (lowCount, highCount) ->
                if (lowCount + highCount) >= rawPixelValues.Length then
                    [||]
                else
                    let sorted = Array.copy rawPixelValues
                    Array.sortInPlace sorted
                    sorted.[lowCount .. (sorted.Length - highCount - 1)]

            | SigmaClipping (lowSigma, highSigma) ->
                let mutable currentPixels = rawPixelValues
                let mutable lastCount = currentPixels.Length + 1
                let mutable i = 0

                while i < iterations && currentPixels.Length < lastCount && currentPixels.Length > 0 do
                    lastCount <- currentPixels.Length
                    i <- i + 1

                    // Get data to calculate stats on (potentially normalized for rejection)
                    let dataForStats =
                        match rejectionNormalization with
                        | NoRejectionNormalization -> currentPixels
                        | ScaleAndZeroOffset ->
                            let (mean, _, stdDev) = Stats.calculate currentPixels
                            if stdDev = 0.0 then currentPixels
                            else currentPixels |> Array.map (fun p -> (p - mean) / stdDev)
                        | EqualizeFluxes ->
                            let (_, median, _) = Stats.calculate currentPixels
                            if median = 0.0 then currentPixels
                            else currentPixels |> Array.map (fun p -> p / median)

                    let (mean, _, stdDev) = Stats.calculate dataForStats

                    if stdDev > 0.0 then
                        let lowThreshold = mean - lowSigma * stdDev
                        let highThreshold = mean + highSigma * stdDev

                        // Filter based on 'dataForStats', but keep 'currentPixels' values
                        let zipped = Array.zip currentPixels dataForStats
                        currentPixels <-
                            zipped
                            |> Array.filter (fun (raw, norm) -> norm >= lowThreshold && norm <= highThreshold)
                            |> Array.map fst

                currentPixels

            | LinearFitClipping (lowSigma, highSigma) ->
                let keepMask = LinearFit.rejectOutliers rawPixelValues lowSigma highSigma iterations
                rawPixelValues
                |> Array.mapi (fun i v -> (i, v))
                |> Array.filter (fun (i, _) -> keepMask.[i])
                |> Array.map snd

        // Combine pixels
        if Array.isEmpty pixelsToAverage then 0.0
        else
            match combination with
            | Average ->
                let sum = Array.sum pixelsToAverage
                sum / (float pixelsToAverage.Length)
            | Median ->
                let sorted = Array.copy pixelsToAverage
                Array.sortInPlace sorted
                let mid = sorted.Length / 2
                if sorted.Length % 2 = 0 then
                    (sorted.[mid - 1] + sorted.[mid]) / 2.0
                else
                    sorted.[mid]
    )

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let (inputPattern, outputPath, settings, overwrite, outputFormatOverride, inlineCalConfig) = parseArgs args

                // Check if output exists
                if File.Exists(outputPath) && not overwrite then
                    Log.Warning($"Output file '{Path.GetFileName outputPath}' already exists, skipping (use --overwrite to replace)")
                    return 0
                else

                if File.Exists(outputPath) && overwrite then
                    Log.Information($"Overwriting existing file '{Path.GetFileName outputPath}'")

                Log.Information("Starting image integration")

                // Load inline calibration masters if configured
                let! inlineMasters =
                    match inlineCalConfig with
                    | Some config -> InlineCalibration.loadMasters config
                    | None -> async { return None }

                // Find input files
                let inputDir = Path.GetDirectoryName(inputPattern)
                let pattern = Path.GetFileName(inputPattern)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else

                let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                if files.Length = 0 then
                    Log.Error($"No files found matching pattern: {inputPattern}")
                    return 1
                else

                // Validate minimum image count for rejection
                match settings.Rejection with
                | NoRejection -> ()
                | _ when files.Length < 3 ->
                    Log.Warning($"Only {files.Length} images found - rejection algorithms work best with 3+ images")
                | _ -> ()

                printfn $"Found {files.Length} files to integrate"
                printfn "Loading images into memory..."

                // Load all images
                let reader = new XisfReader()
                let! units =
                    files
                    |> Array.map (fun f -> reader.ReadAsync(f) |> Async.AwaitTask)
                    |> Async.Parallel

                let images = units |> Array.map (fun u -> u.Images.[0])

                printfn $"Loaded {images.Length} images"

                // Validate
                validateImages images

                let firstImg = images.[0]
                let width = int firstImg.Geometry.Width
                let height = int firstImg.Geometry.Height
                let channels = int firstImg.Geometry.ChannelCount
                let pixelCount = width * height
                let inputFormat = firstImg.SampleFormat

                printfn $"Image format: {width}x{height}, {channels} channel(s), {inputFormat}"
                printfn $"Normalization: {settings.Normalization}"
                printfn $"Rejection: {settings.Rejection}"
                printfn $"Combination: {settings.Combination}"

                // Get pixel data using PixelIO (handles all sample formats)
                // Apply inline calibration if configured
                let pixelArrays =
                    match inlineMasters, inlineCalConfig with
                    | Some masters, Some config ->
                        printfn "Applying inline calibration to input images..."
                        // Validate dimensions match
                        if masters.Width > 0 && (masters.Width <> width || masters.Height <> height || masters.Channels <> channels) then
                            failwithf "Master frame dimension mismatch: Expected %dx%dx%d, got %dx%dx%d"
                                width height channels masters.Width masters.Height masters.Channels
                        images
                        |> Array.Parallel.map (fun img ->
                            let rawPixels = PixelIO.readPixelsAsFloat img
                            InlineCalibration.calibratePixels rawPixels masters config
                        )
                    | _ ->
                        images |> Array.map PixelIO.readPixelsAsFloat

                // Calculate stats if normalization needed
                let (imageStats, referenceStats) =
                    if settings.Normalization = NoNormalization then
                        (Map.empty, Array.create channels (0.0, 0.0))
                    else
                        calculateImageStats pixelArrays channels pixelCount

                printfn $"Processing {pixelCount} pixels in parallel..."

                // Progress tracking
                let mutable progressCounter = 0
                let progressLock = obj()
                let reportProgress () =
                    lock progressLock (fun () ->
                        progressCounter <- progressCounter + 1
                        if progressCounter % 500000 = 0 then
                            printfn $"   {progressCounter * 100 / pixelCount}%%"
                    )

                // Process all pixels
                let pixelResults =
                    Array.Parallel.init pixelCount (fun pix ->
                        if pix % 500000 = 0 then reportProgress ()
                        processPixel pixelArrays pix channels settings.Combination settings.Normalization settings.Rejection settings.RejectionNormalization settings.Iterations imageStats referenceStats
                    )

                // Flatten results into float array
                let stackedFloats = Array.zeroCreate (pixelCount * channels)
                for pix = 0 to pixelCount - 1 do
                    let offset = pix * channels
                    let channelValues = pixelResults.[pix]
                    Array.blit channelValues 0 stackedFloats offset channels

                // Determine output format
                // Default: UInt16 → Float32 (preserves fractional precision from averaging)
                // Float32/Float64 → preserve
                let (outputFormat, normalize) =
                    match outputFormatOverride with
                    | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
                    | None ->
                        match inputFormat with
                        | XisfSampleFormat.UInt8 | XisfSampleFormat.UInt16 | XisfSampleFormat.UInt32 ->
                            // Integer inputs → Float32 output (preserve stacking precision)
                            (XisfSampleFormat.Float32, true)
                        | XisfSampleFormat.Float32 | XisfSampleFormat.Float64 ->
                            // Float inputs → preserve format
                            PixelIO.getRecommendedOutputFormat inputFormat
                        | _ ->
                            (XisfSampleFormat.Float32, true)

                // Convert to output format using PixelIO
                let stacked = PixelIO.writePixelsFromFloat stackedFloats outputFormat normalize

                printfn $"Output format: {outputFormat}"

                printfn "Building output metadata..."

                // Build output metadata
                let combString = sprintf "%A" settings.Combination
                let normString = sprintf "%A" settings.Normalization
                let rejectString = sprintf "%A" settings.Rejection

                let firstProps =
                    if isNull firstImg.Properties then [||]
                    else firstImg.Properties :> seq<_> |> Seq.toArray

                let findProp id = firstProps |> Array.tryFind (fun p -> p.Id = id)

                let firstFits =
                    if isNull firstImg.AssociatedElements then [||]
                    else
                        firstImg.AssociatedElements :> seq<_>
                        |> Seq.toArray
                        |> Array.choose (fun e -> if e :? XisfFitsKeyword then Some (e :?> XisfFitsKeyword) else None)

                let findFits name = firstFits |> Array.tryFind (fun f -> f.Name = name)

                let outputProps =
                    [
                        findProp "Instrument:Camera:Name"
                        findProp "Instrument:Camera:Gain"
                        findProp "Instrument:Camera:XBinning"
                        findProp "Instrument:Camera:YBinning"
                        findProp "Instrument:Sensor:XPixelSize"
                        findProp "Instrument:Sensor:YPixelSize"

                        Some (XisfScalarProperty<int>("ImageIntegration:NumberOfImages", images.Length, "Number of integrated frames") :> XisfProperty)
                        Some (XisfStringProperty("ImageIntegration:PixelCombination", combString, "Pixel combination method") :> XisfProperty)
                        Some (XisfStringProperty("ImageIntegration:PixelRejection", rejectString, "Pixel rejection algorithm") :> XisfProperty)
                        Some (XisfStringProperty("ImageIntegration:OutputNormalization", normString, "Image normalization method") :> XisfProperty)
                    ]
                    |> List.choose id
                    |> Array.ofList

                let outputFits =
                    [
                        findFits "INSTRUME" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "EGAIN" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "XBINNING" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "YBINNING" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "XPIXSZ" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "YPIXSZ" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "BAYERPAT" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "XBAYROFF" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "YBAYROFF" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "TELESCOP" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "FOCALLEN" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "APTDIA" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "EQUINOX" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "OBSGEO-L" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "OBSGEO-B" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "LONG-OBS" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "LAT-OBS" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "DATE-OBS" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "DATE-END" |> Option.map (fun f -> f :> XisfCoreElement)
                        findFits "IMAGETYP" |> Option.map (fun f -> f :> XisfCoreElement)

                        Some (XisfFitsKeyword("HISTORY", "", "Integration with XisfPrep v1.0") :> XisfCoreElement)
                        Some (XisfFitsKeyword("HISTORY", "", $"ImageIntegration.pixelCombination: {combString}") :> XisfCoreElement)
                        Some (XisfFitsKeyword("HISTORY", "", $"ImageIntegration.outputNormalization: {normString}") :> XisfCoreElement)
                        Some (XisfFitsKeyword("HISTORY", "", $"ImageIntegration.pixelRejection: {rejectString}") :> XisfCoreElement)
                        Some (XisfFitsKeyword("HISTORY", "", $"ImageIntegration.numberOfImages: {images.Length}") :> XisfCoreElement)
                        Some (XisfFitsKeyword("HISTORY", "", $"ImageIntegration.totalPixels: {pixelCount * images.Length}") :> XisfCoreElement)
                    ]
                    |> List.choose id
                    |> Array.ofList

                // Get bounds per XISF spec: Some for Float32/Float64, None for integer formats
                let bounds =
                    match PixelIO.getBoundsForFormat outputFormat with
                    | Some b -> b
                    | None -> Unchecked.defaultof<XisfImageBounds>  // null for integer formats

                let dataBlock = InlineDataBlock(ReadOnlyMemory(stacked), XisfEncoding.Base64)
                let outImage = XisfImage(
                    firstImg.Geometry,
                    outputFormat,
                    firstImg.ColorSpace,
                    dataBlock,
                    bounds,
                    firstImg.PixelStorage,
                    firstImg.ImageType,
                    firstImg.Offset,
                    firstImg.Orientation,
                    firstImg.ImageId,
                    firstImg.Uuid,
                    outputProps,
                    outputFits
                )

                let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Integrate v1.0")
                let outUnit = XisfFactory.CreateMonolithic(metadata, outImage)

                printfn "Writing output..."
                let writer = new XisfWriter()
                do! writer.WriteAsync(outUnit, outputPath) |> Async.AwaitTask

                let sizeMB = (FileInfo outputPath).Length / 1024L / 1024L
                printfn ""
                printfn $"Integration complete: {Path.GetFileName outputPath} ({sizeMB} MB)"
                printfn $"  Integrated {images.Length} images ({width}x{height}, {channels} channel(s))"

                return 0
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep integrate --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
