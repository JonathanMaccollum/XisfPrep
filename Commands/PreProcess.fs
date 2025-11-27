module Commands.PreProcess

open System
open System.Diagnostics
open System.IO
open Serilog
open FsToolkit.ErrorHandling
open XisfLib.Core
open Algorithms
open Algorithms.Alignment
open Algorithms.Binning
open Algorithms.Calibration
open Algorithms.StarDetection
open Algorithms.TriangleMatch
open Algorithms.SpatialMatch
open Algorithms.Interpolation
open Algorithms.RBFTransform
open Algorithms.Painting
open Algorithms.InputValidation
open Algorithms.OutputImage

type PipelineError =
    | LoadFailed of string
    | CalibrationFailed of Algorithms.Calibration.CalibrationError
    | BinningFailed of Algorithms.Binning.BinningError
    | DetectionFailed of string
    | MatchingFailed of Algorithms.Alignment.AlignmentError
    | TransformFailed of Algorithms.Alignment.AlignmentError
    | SaveFailed of string

    override this.ToString() =
        match this with
        | LoadFailed msg -> $"Load failed: {msg}"
        | CalibrationFailed err -> $"Calibration failed: {err}"
        | BinningFailed err -> $"Binning failed: {err}"
        | DetectionFailed msg -> $"Detection failed: {msg}"
        | MatchingFailed err -> $"Matching failed: {err}"
        | TransformFailed err -> $"Transform failed: {err}"
        | SaveFailed msg -> $"Save failed: {msg}"

type OutputMode =
    | Detect
    | Match
    | Align
    | Distortion

/// Distortion correction kernel
type DistortionCorrection =
    | NoDistortion
    | Wendland
    | TPS
    | IMQ

let private defaultInterpolation = Lanczos3
let private defaultSuffix = "_a"
let private defaultParallel = Environment.ProcessorCount
// Detection defaults
let private defaultThreshold = 5.0
let private defaultGridSize = 128
let private defaultMinFWHM = 1.5
let private defaultMaxFWHM = 20.0
let private defaultMaxEccentricity = 0.5
let private defaultMaxStars = 20000
// Matching defaults
let private defaultRatioTolerance = 0.05
let private defaultMaxStarsTriangles = 100
let private defaultMinVotes = 3
// Visualization defaults
let private defaultIntensity = 1.0
// Algorithm defaults
let private defaultAlgorithm = Expanding
let private defaultAnchorStars = 12
let private defaultAnchorDistribution = SpatialMatch.Center
// Distortion correction defaults
let private defaultDistortion = NoDistortion
let private defaultRBFSupportFactor = 3.0
let private defaultRBFRegularization = 1e-6

type PreProcessOptions = {
    Input: string
    Output: string
    Reference: string option
    AutoReference: bool
    OutputMode: OutputMode
    Interpolation: InterpolationMethod
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
    // Detection parameters
    Threshold: float
    GridSize: int
    MinFWHM: float
    MaxFWHM: float
    MaxEccentricity: float
    MaxStars: int
    // Matching parameters
    RatioTolerance: float
    MaxStarsTriangles: int
    MinVotes: int
    // Visualization
    Intensity: float
    // Algorithm selection
    Algorithm: MatchAlgorithm
    AnchorStars: int
    AnchorDistribution: SpatialMatch.AnchorDistribution
    // Distortion correction
    Distortion: DistortionCorrection
    RBFSupportFactor: float
    RBFRegularization: float
    // Additional outputs
    ShowDistortionStats: bool
    IncludeDistortionModel: bool
    IncludeDetectionModel: bool
    // Calibration
    BiasFrame: string option
    BiasLevel: float option
    DarkFrame: string option
    FlatFrame: string option
    UncalibratedDark: bool
    UncalibratedFlat: bool
    OptimizeDark: bool
    OutputPedestal: int
    // Binning
    BinFactor: int option
    BinMethod: BinningMethod
}

type PreProcessValidationError =
    | MissingInput
    | MissingOutput
    | ReferenceOptionsConflict
    | InvalidParallelCount of value: int
    | InvalidThreshold of value: float
    | InvalidGridSize of value: int
    | InvalidMinFWHM of value: float
    | InvalidMaxFWHM of min: float * max: float
    | InvalidMaxEccentricity of value: float
    | InvalidMaxStars of value: int
    | InvalidRatioTolerance of value: float
    | InvalidMaxStarsTriangles of value: int
    | InvalidMinVotes of value: int
    | InvalidIntensity of value: float
    | InvalidAnchorStars of value: int
    | InvalidRBFSupportFactor of value: float
    | InvalidRBFRegularization of value: float
    | InvalidBinFactor of value: int

    override this.ToString() =
        match this with
        | MissingInput -> "Required argument: --input"
        | MissingOutput -> "Required argument: --output"
        | ReferenceOptionsConflict -> "--reference and --auto-reference are mutually exclusive"
        | InvalidParallelCount value -> $"Parallel count must be at least 1, got {value}"
        | InvalidThreshold value -> $"Threshold must be positive, got {value}"
        | InvalidGridSize value -> $"Grid size must be at least 16, got {value}"
        | InvalidMinFWHM value -> $"Min FWHM must be positive, got {value}"
        | InvalidMaxFWHM (min, max) -> $"Max FWHM ({max}) must be greater than min FWHM ({min})"
        | InvalidMaxEccentricity value -> $"Max eccentricity must be between 0 and 1, got {value}"
        | InvalidMaxStars value -> $"Max stars must be at least 1, got {value}"
        | InvalidRatioTolerance value -> $"Ratio tolerance must be positive, got {value}"
        | InvalidMaxStarsTriangles value -> $"Max stars for triangles must be at least 3, got {value}"
        | InvalidMinVotes value -> $"Min votes must be at least 1, got {value}"
        | InvalidIntensity value -> $"Intensity must be between 0 and 1, got {value}"
        | InvalidAnchorStars value -> $"Anchor stars must be at least 4, got {value}"
        | InvalidRBFSupportFactor value -> $"RBF support factor must be at least 1.0, got {value}"
        | InvalidRBFRegularization value -> $"RBF regularization must be non-negative, got {value}"
        | InvalidBinFactor value -> $"Bin factor must be between 2 and 6, got {value}"

type RunError =
    | ValidationFailed of PreProcessValidationError
    | CalibrationValidationFailed of Algorithms.Calibration.CalibrationValidationError
    | InputResolutionFailed of InputValidationError
    | MasterLoadFailed of string
    | ReferenceAnalysisFailed of PipelineError
    | InsufficientStars of found: int * required: int
    | ProcessingFailed of string

    override this.ToString() =
        match this with
        | ValidationFailed err -> err.ToString()
        | CalibrationValidationFailed err -> $"Calibration validation failed: {err}"
        | InputResolutionFailed err -> err.ToString()
        | MasterLoadFailed msg -> $"Failed to load calibration masters: {msg}"
        | ReferenceAnalysisFailed err -> $"Reference analysis failed: {err}"
        | InsufficientStars (found, required) -> $"Insufficient stars in reference image: found {found}, need at least {required}"
        | ProcessingFailed msg -> msg

type BinningConfig = {
    Factor: int option
    Method: BinningMethod
}

type OutputConfig = {
    Mode: OutputMode
    Intensity: float
    Format: XisfSampleFormat option
    IncludeDetectionModel: bool
    IncludeDistortionModel: bool
    ShowDistortionStats: bool
    RefImageDims: int * int
}

type PipelineConfigs = {
    Calibration: (MasterFrames * Algorithms.Calibration.CalibrationConfig) option
    Binning: BinningConfig option
    Detection: DetectionConfig
    Matching: MatchingConfig option
    Distortion: DistortionConfig option
    Transform: TransformConfig option
    Output: OutputConfig
}

/// Reference image metadata
type ReferenceData = {
    Frame: Alignment.ReferenceFrame
    FileName: string
    DetectionParams: DetectionParams
}

let validatePreProcessOptions (opts: PreProcessOptions) : Result<unit, PreProcessValidationError> =
    if String.IsNullOrEmpty opts.Input then
        Error MissingInput
    elif String.IsNullOrEmpty opts.Output then
        Error MissingOutput
    elif opts.Reference.IsSome && opts.AutoReference then
        Error ReferenceOptionsConflict
    elif opts.MaxParallel < 1 then
        Error (InvalidParallelCount opts.MaxParallel)
    elif opts.Threshold <= 0.0 then
        Error (InvalidThreshold opts.Threshold)
    elif opts.GridSize < 16 then
        Error (InvalidGridSize opts.GridSize)
    elif opts.MinFWHM <= 0.0 then
        Error (InvalidMinFWHM opts.MinFWHM)
    elif opts.MaxFWHM <= opts.MinFWHM then
        Error (InvalidMaxFWHM (opts.MinFWHM, opts.MaxFWHM))
    elif opts.MaxEccentricity < 0.0 || opts.MaxEccentricity > 1.0 then
        Error (InvalidMaxEccentricity opts.MaxEccentricity)
    elif opts.MaxStars < 1 then
        Error (InvalidMaxStars opts.MaxStars)
    elif opts.RatioTolerance <= 0.0 then
        Error (InvalidRatioTolerance opts.RatioTolerance)
    elif opts.MaxStarsTriangles < 3 then
        Error (InvalidMaxStarsTriangles opts.MaxStarsTriangles)
    elif opts.MinVotes < 1 then
        Error (InvalidMinVotes opts.MinVotes)
    elif opts.Intensity < 0.0 || opts.Intensity > 1.0 then
        Error (InvalidIntensity opts.Intensity)
    elif opts.AnchorStars < 4 then
        Error (InvalidAnchorStars opts.AnchorStars)
    elif opts.RBFSupportFactor < 1.0 then
        Error (InvalidRBFSupportFactor opts.RBFSupportFactor)
    elif opts.RBFRegularization < 0.0 then
        Error (InvalidRBFRegularization opts.RBFRegularization)
    elif opts.BinFactor.IsSome && (opts.BinFactor.Value < 2 || opts.BinFactor.Value > 6) then
        Error (InvalidBinFactor opts.BinFactor.Value)
    else
        Ok ()

let validateCalibrationConfig (opts: PreProcessOptions) : Result<unit, Algorithms.Calibration.CalibrationValidationError> =
    if opts.BiasFrame.IsNone && opts.BiasLevel.IsNone && opts.DarkFrame.IsNone && opts.FlatFrame.IsNone then
        Ok ()
    else
        let calConfig: Algorithms.Calibration.CalibrationConfig = {
            BiasFrame = opts.BiasFrame
            BiasLevel = opts.BiasLevel
            DarkFrame = opts.DarkFrame
            FlatFrame = opts.FlatFrame
            UncalibratedDark = opts.UncalibratedDark
            UncalibratedFlat = opts.UncalibratedFlat
            OptimizeDark = opts.OptimizeDark
            OutputPedestal = opts.OutputPedestal
        }
        Calibration.validateConfig calConfig

let validateReferenceStars (count: int) : Result<unit, string> =
    if count < 10 then
        Error $"Insufficient stars in reference image: found {count}, need at least 10"
    else
        Ok ()

let resolveReferenceFile (files: string[]) (opts: PreProcessOptions) : string =
    match opts.Reference with
    | Some r -> r
    | None ->
        if opts.AutoReference then
            printfn "Auto-reference selection not yet implemented, using first file"
        files.[0]

let parseArgs (args: string array) : PreProcessOptions =
    let rec parse (args: string list) (opts: PreProcessOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = value }
        | "--reference" :: value :: rest | "-r" :: value :: rest ->
            parse rest { opts with Reference = Some value }
        | "--auto-reference" :: rest ->
            parse rest { opts with AutoReference = true }
        | "--output-mode" :: value :: rest ->
            let mode = match value.ToLower() with
                       | "detect" -> Detect
                       | "match" -> Match
                       | "align" -> Align
                       | "distortion" -> Distortion
                       | _ -> failwithf "Unknown output mode: %s" value
            parse rest { opts with OutputMode = mode }
        | "--interpolation" :: value :: rest ->
            let interp = match value.ToLower() with
                         | "nearest" -> Nearest
                         | "bilinear" -> Bilinear
                         | "bicubic" -> Bicubic
                         | "lanczos" | "lanczos3" -> Lanczos3
                         | _ -> failwithf "Unknown interpolation method: %s" value
            parse rest { opts with Interpolation = interp }
        | "--suffix" :: value :: rest ->
            parse rest { opts with Suffix = value }
        | "--overwrite" :: rest ->
            parse rest { opts with Overwrite = true }
        | "--parallel" :: value :: rest ->
            parse rest { opts with MaxParallel = int value }
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> parse rest { opts with OutputFormat = Some fmt }
            | None -> failwithf "Unknown output format: %s" value
        | "--threshold" :: value :: rest ->
            parse rest { opts with Threshold = float value }
        | "--grid-size" :: value :: rest ->
            parse rest { opts with GridSize = int value }
        | "--min-fwhm" :: value :: rest ->
            parse rest { opts with MinFWHM = float value }
        | "--max-fwhm" :: value :: rest ->
            parse rest { opts with MaxFWHM = float value }
        | "--max-eccentricity" :: value :: rest ->
            parse rest { opts with MaxEccentricity = float value }
        | "--max-stars" :: value :: rest ->
            parse rest { opts with MaxStars = int value }
        | "--algorithm" :: value :: rest ->
            let algo = match value.ToLower() with
                       | "triangle" -> Triangle
                       | "expanding" -> Expanding
                       | _ -> failwithf "Unknown algorithm: %s" value
            parse rest { opts with Algorithm = algo }
        | "--anchor-stars" :: value :: rest ->
            parse rest { opts with AnchorStars = int value }
        | "--anchor-spread" :: value :: rest ->
            let dist = match value.ToLower() with
                       | "center" -> SpatialMatch.Center
                       | "grid" -> SpatialMatch.Grid
                       | _ -> failwithf "Unknown anchor spread: %s" value
            parse rest { opts with AnchorDistribution = dist }
        | "--ratio-tolerance" :: value :: rest ->
            parse rest { opts with RatioTolerance = float value }
        | "--max-stars-triangles" :: value :: rest ->
            parse rest { opts with MaxStarsTriangles = int value }
        | "--min-votes" :: value :: rest ->
            parse rest { opts with MinVotes = int value }
        | "--intensity" :: value :: rest ->
            parse rest { opts with Intensity = float value }
        | "--distortion" :: value :: rest ->
            let dist = match value.ToLower() with
                       | "none" -> NoDistortion
                       | "wendland" -> Wendland
                       | "tps" -> TPS
                       | "imq" -> IMQ
                       | _ -> failwithf "Unknown distortion: %s" value
            parse rest { opts with Distortion = dist }
        | "--rbf-support" :: value :: rest ->
            parse rest { opts with RBFSupportFactor = float value }
        | "--rbf-regularization" :: value :: rest ->
            parse rest { opts with RBFRegularization = float value }
        | "--show-distortion-stats" :: rest ->
            parse rest { opts with ShowDistortionStats = true }
        | "--include-distortion-model" :: rest ->
            parse rest { opts with IncludeDistortionModel = true }
        | "--include-detection-model" :: rest ->
            parse rest { opts with IncludeDetectionModel = true }
        | "--bias" :: value :: rest | "-b" :: value :: rest ->
            parse rest { opts with BiasFrame = Some value }
        | "--bias-level" :: value :: rest ->
            parse rest { opts with BiasLevel = Some (float value) }
        | "--dark" :: value :: rest | "-d" :: value :: rest ->
            parse rest { opts with DarkFrame = Some value }
        | "--flat" :: value :: rest | "-f" :: value :: rest ->
            parse rest { opts with FlatFrame = Some value }
        | "--uncalibrated-dark" :: rest ->
            parse rest { opts with UncalibratedDark = true }
        | "--uncalibrated-flat" :: rest ->
            parse rest { opts with UncalibratedFlat = true }
        | "--optimize-dark" :: rest ->
            parse rest { opts with OptimizeDark = true }
        | "--pedestal" :: value :: rest ->
            parse rest { opts with OutputPedestal = int value }
        | "--bin-factor" :: value :: rest ->
            parse rest { opts with BinFactor = Some (int value) }
        | "--bin-method" :: value :: rest ->
            let method = match value.ToLower() with
                         | "average" -> Average
                         | "median" -> Median
                         | "sum" -> Sum
                         | _ -> failwithf "Unknown binning method: %s" value
            parse rest { opts with BinMethod = method }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Reference = None
        AutoReference = false
        OutputMode = Align
        Interpolation = defaultInterpolation
        Suffix = defaultSuffix
        Overwrite = false
        MaxParallel = defaultParallel
        OutputFormat = None
        Threshold = defaultThreshold
        GridSize = defaultGridSize
        MinFWHM = defaultMinFWHM
        MaxFWHM = defaultMaxFWHM
        MaxEccentricity = defaultMaxEccentricity
        MaxStars = defaultMaxStars
        RatioTolerance = defaultRatioTolerance
        MaxStarsTriangles = defaultMaxStarsTriangles
        MinVotes = defaultMinVotes
        Intensity = defaultIntensity
        Algorithm = defaultAlgorithm
        AnchorStars = defaultAnchorStars
        AnchorDistribution = defaultAnchorDistribution
        Distortion = defaultDistortion
        RBFSupportFactor = defaultRBFSupportFactor
        RBFRegularization = defaultRBFRegularization
        ShowDistortionStats = false
        IncludeDistortionModel = false
        IncludeDetectionModel = false
        BiasFrame = None
        BiasLevel = None
        DarkFrame = None
        FlatFrame = None
        UncalibratedDark = false
        UncalibratedFlat = false
        OptimizeDark = false
        OutputPedestal = 0
        BinFactor = None
        BinMethod = Average
    }

    parse (List.ofArray args) defaults

let magnitudeToAscii (mag: float) : char =
    if mag < 2.0 then '·'
    elif mag < 4.0 then '░'
    elif mag < 6.0 then '▒'
    elif mag < 8.0 then '▓'
    else '█'

/// Map magnitude to color (blue -> cyan -> green -> yellow -> red)
let magnitudeToColor (mag: float) (maxMag: float) : uint16 * uint16 * uint16 =
    let t = min 1.0 (mag / maxMag)
    let (r, g, b) =
        if t < 0.25 then
            (0.0, t * 4.0, 1.0)
        elif t < 0.5 then
            (0.0, 1.0, 1.0 - (t - 0.25) * 4.0)
        elif t < 0.75 then
            ((t - 0.5) * 4.0, 1.0, 0.0)
        else
            (1.0, 1.0 - (t - 0.75) * 4.0, 0.0)
    (uint16 (r * 65535.0), uint16 (g * 65535.0), uint16 (b * 65535.0))

/// Print console heatmap of distortion field
let printDistortionHeatmap
    (width: int) (height: int)
    (transform: SimilarityTransform)
    (rbfCoeffs: RBFCoefficients option)
    (cols: int) =

    let aspectRatio = float height / float width
    let rows = max 1 (int (float cols * aspectRatio))

    let cellWidth = float width / float cols
    let cellHeight = float height / float rows

    printfn ""
    printfn "Distortion Field Heatmap (sampled at %dx%d grid)" cols rows
    printfn "─────────────────────────────────────────────────────────────────"
    printfn "Scale: █ ≥8px  ▓ 6-8px  ▒ 4-6px  ░ 2-4px  · <2px"
    printfn ""

    // Sample distortion at grid points
    let samples = Array2D.zeroCreate rows cols
    let mutable minMag = Double.MaxValue
    let mutable maxMag = Double.MinValue
    let mutable sumMag = 0.0
    let mutable count = 0

    for row in 0 .. rows - 1 do
        for col in 0 .. cols - 1 do
            // Sample at center of cell
            let ox = (float col + 0.5) * cellWidth
            let oy = (float row + 0.5) * cellHeight

            let (tx, ty) = RBFTransform.applyFullInverseTransform transform rbfCoeffs ox oy 5 0.01

            let dx = ox - tx
            let dy = oy - ty
            let mag = sqrt (dx * dx + dy * dy)

            samples.[row, col] <- mag
            minMag <- min minMag mag
            maxMag <- max maxMag mag
            sumMag <- sumMag + mag
            count <- count + 1

    let meanMag = sumMag / float count

    // Compute standard deviation
    let mutable sumSqDiff = 0.0
    for row in 0 .. rows - 1 do
        for col in 0 .. cols - 1 do
            let diff = samples.[row, col] - meanMag
            sumSqDiff <- sumSqDiff + diff * diff
    let stdDev = sqrt (sumSqDiff / float count)

    // Print values
    for row in 0 .. rows - 1 do
        // Print numeric values
        for col in 0 .. cols - 1 do
            printf "%5.1f " samples.[row, col]
        printfn ""

        // Print ASCII representation
        for col in 0 .. cols - 1 do
            let c = magnitudeToAscii samples.[row, col]
            printf "  %c   " c
        printfn ""
        printfn ""

    printfn "─────────────────────────────────────────────────────────────────"
    printfn "Statistics:"
    printfn "  Min: %.2fpx  Max: %.2fpx  Mean: %.2fpx  StdDev: %.2fpx" minMag maxMag meanMag stdDev

let calibrate (masters: (MasterFrames * Algorithms.Calibration.CalibrationConfig) option) (img: ImageData) : Result<ImageData, PipelineError> =
    match masters with
    | None -> Ok img
    | Some (m, calConfig) ->
        try
            let result = calibratePixels img.Pixels m calConfig
            Ok { img with Pixels = result.CalibratedPixels }
        with ex ->
            Error (CalibrationFailed (Algorithms.Calibration.LoadFailed ("calibration", ex.Message)))

let bin (config: BinningConfig) (img: ImageData) : Result<ImageData, PipelineError> =
    match config.Factor with
    | None -> Ok img
    | Some factor ->
        result {
            let binConfig: Algorithms.Binning.BinningConfig = {
                Factor = factor
                Method = config.Method
            }

            do! Binning.validateConfig binConfig
                |> Result.mapError BinningFailed

            do! Binning.validateDimensions img.Width img.Height img.Channels factor
                |> Result.mapError BinningFailed
                |> Result.map ignore

            let result = binPixels img.Pixels img.Width img.Height img.Channels binConfig

            return {
                Pixels = result.BinnedPixels
                Width = result.NewWidth
                Height = result.NewHeight
                Channels = img.Channels
            }
        }

let preprocessImage
    (masters: (MasterFrames * Algorithms.Calibration.CalibrationConfig) option)
    (binConfig: BinningConfig)
    (img: ImageData)
    : Result<ImageData, PipelineError> =
    result {
        let! calibrated = calibrate masters img
        let! binned = bin binConfig calibrated
        return binned
    }

let matchStars
    (config: MatchingConfig option)
    (fileName: string)
    (detected: DetectedImage)
    : Result<MatchedImage option, PipelineError> =
    match config with
    | None -> Ok None
    | Some cfg ->
        Alignment.matchToReference cfg fileName detected
        |> Result.mapError MatchingFailed
        |> Result.map Some

let applyDistortion
    (config: DistortionConfig option)
    (matched: MatchedImage option)
    : Result<DistortionResult option, PipelineError> =
    match (config, matched) with
    | (None, _) | (_, None) -> Ok None
    | (Some cfg, Some m) ->
        let (_, result) = Alignment.computeDistortion (Some cfg) m
        Ok result

let applyTransform
    (config: TransformConfig option)
    (matched: MatchedImage option)
    (distortion: DistortionResult option)
    : Result<ImageData option, PipelineError> =
    match (config, matched) with
    | (None, _) | (_, None) -> Ok None
    | (Some cfg, Some m) ->
        let rbfCoeffs = distortion |> Option.bind (fun r -> r.Coefficients)
        Alignment.transform cfg (m, rbfCoeffs)
        |> Result.mapError TransformFailed
        |> Result.map Some

let buildPipelineConfigs
    (mode: OutputMode)
    (opts: PreProcessOptions)
    (masters: (MasterFrames * Algorithms.Calibration.CalibrationConfig) option)
    (refData: ReferenceData)
    : PipelineConfigs =
    {
        Calibration = masters
        Binning =
            match opts.BinFactor with
            | None -> None
            | Some factor -> Some { Factor = Some factor; Method = opts.BinMethod }
        Detection = {
            Threshold = opts.Threshold
            GridSize = opts.GridSize
            MinFWHM = opts.MinFWHM
            MaxFWHM = opts.MaxFWHM
            MaxEccentricity = opts.MaxEccentricity
            MaxStars = opts.MaxStars
        }
        Matching =
            match mode with
            | Detect -> None
            | _ -> Some {
                RefStars = refData.Frame.Detected.Stars
                RefTriangles = refData.Frame.Triangles
                Params = {
                    Algorithm = opts.Algorithm
                    RatioTolerance = opts.RatioTolerance
                    MaxStarsTriangles = opts.MaxStarsTriangles
                    MinVotes = opts.MinVotes
                    AnchorStars = opts.AnchorStars
                    AnchorDistribution = opts.AnchorDistribution
                }
            }
        Distortion =
            match (mode, opts.Distortion) with
            | (Align, d) | (Distortion, d) when d <> NoDistortion ->
                let kernel = match d with
                             | Wendland -> RBFTransform.Wendland 1
                             | TPS -> RBFTransform.ThinPlateSpline
                             | IMQ -> RBFTransform.InverseMultiquadric
                             | NoDistortion -> RBFTransform.Wendland 1
                Some { Kernel = kernel; SupportFactor = opts.RBFSupportFactor; Regularization = opts.RBFRegularization }
            | _ -> None
        Transform =
            match mode with
            | Align -> Some { Interpolation = opts.Interpolation; ApplyRBF = opts.Distortion <> NoDistortion }
            | _ -> None
        Output = {
            Mode = mode
            Intensity = opts.Intensity
            Format = opts.OutputFormat
            IncludeDetectionModel = opts.IncludeDetectionModel
            IncludeDistortionModel = opts.IncludeDistortionModel
            ShowDistortionStats = opts.ShowDistortionStats
            RefImageDims = (refData.Frame.Detected.Image.Width, refData.Frame.Detected.Image.Height)
        }
    }

let loadImage (filePath: string) : Async<Result<ImageData, string>> =
    async {
        try
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(filePath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount
            let pixels = PixelIO.readPixelsAsFloat img

            return Ok {
                Pixels = pixels
                Width = width
                Height = height
                Channels = channels
            }
        with ex ->
            return Error ex.Message
    }

let createBlackPixels (width: int) (height: int) (channels: int) (format: XisfSampleFormat) : byte[] =
    let bytesPerSample =
        match format with
        | XisfSampleFormat.UInt8 -> 1
        | XisfSampleFormat.UInt16 -> 2
        | XisfSampleFormat.UInt32 | XisfSampleFormat.Float32 -> 4
        | XisfSampleFormat.Float64 -> 8
        | _ -> 2
    Array.zeroCreate (width * height * channels * bytesPerSample)

/// Generate detect mode output - paint detected stars on black background
let generateDetectOutput
    (detected: DetectedImage)
    (intensity: float)
    (outputFormat: XisfSampleFormat)
    : byte[] =

    let img = detected.Image
    let pixels = createBlackPixels img.Width img.Height img.Channels outputFormat

    // Paint detected stars
    for star in detected.Stars do
        let radius = star.FWHM * 2.0
        for ch in 0 .. img.Channels - 1 do
            Painting.paintCircle pixels img.Width img.Height img.Channels ch star.X star.Y radius intensity outputFormat

    pixels

/// Generate match mode output - paint match visualization
let generateMatchOutput
    (matched: MatchedImage)
    (refStars: DetectedStar[])
    (intensity: float)
    (outputFormat: XisfSampleFormat)
    : byte[] =

    let img = matched.Detected.Image
    let pixels = createBlackPixels img.Width img.Height img.Channels outputFormat

    // Paint reference stars as faint gaussians
    let refIntensity = intensity * 0.3
    for star in refStars do
        for ch in 0 .. img.Channels - 1 do
            Painting.paintGaussian pixels img.Width img.Height img.Channels ch star.X star.Y star.FWHM refIntensity outputFormat

    // Paint target stars - circles for matched, X for unmatched
    let matchedTargetIndices =
        matched.MatchedPairs
        |> Array.map snd
        |> Set.ofArray

    for i, star in matched.Detected.Stars |> Array.indexed do
        let size = star.FWHM * 2.0
        if matchedTargetIndices.Contains i then
            for ch in 0 .. img.Channels - 1 do
                Painting.paintCircle pixels img.Width img.Height img.Channels ch star.X star.Y size intensity outputFormat
        else
            for ch in 0 .. img.Channels - 1 do
                Painting.paintX pixels img.Width img.Height img.Channels ch star.X star.Y size intensity outputFormat

    pixels

/// Generate align mode output - transformed pixels
let generateAlignOutput (transformed: ImageData) : byte[] =
    // Output as Float32 for interpolation precision
    PixelIO.writePixelsFromFloat transformed.Pixels XisfSampleFormat.Float32 true

/// Create XISF output image
let createOutputImage
    (originalImg: XisfImage)
    (pixels: byte[])
    (width: int)
    (height: int)
    (channels: int)
    (format: XisfSampleFormat)
    (headers: XisfCoreElement[])
    : XisfImage =

    let geometry = XisfImageGeometry([| uint32 width; uint32 height |], uint32 channels)
    let dataBlock = InlineDataBlock(ReadOnlyMemory(pixels), XisfEncoding.Base64)
    let bounds = PixelIO.getBoundsForFormat format |> Option.defaultValue (Unchecked.defaultof<XisfImageBounds>)

    let inPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
    let existingFits =
        if isNull originalImg.AssociatedElements then [||]
        else originalImg.AssociatedElements |> Seq.toArray

    let preservedFits =
        existingFits
        |> Array.filter (fun elem ->
            match elem with
            | :? XisfFitsKeyword as kw -> not (inPlaceKeys.Contains kw.Name)
            | _ -> true)

    let allFits = Array.append preservedFits headers

    XisfImage(
        geometry, format, originalImg.ColorSpace, dataBlock, bounds,
        originalImg.PixelStorage, originalImg.ImageType, originalImg.Offset,
        originalImg.Orientation, originalImg.ImageId, originalImg.Uuid,
        originalImg.Properties, allFits
    )

/// Write XISF file
let writeOutputFile (path: string) (image: XisfImage) : Async<Result<unit, string>> =
    async {
        try
            let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Align v1.0")
            let unit = XisfFactory.CreateMonolithic(metadata, image)
            let writer = new XisfWriter()
            do! writer.WriteAsync(unit, path) |> Async.AwaitTask
            return Ok ()
        with ex ->
            return Error ex.Message
    }

let loadCalibrationMasters (opts: PreProcessOptions)
    : Async<Result<(MasterFrames * Algorithms.Calibration.CalibrationConfig) option, string>> =
    async {
        if opts.BiasFrame.IsNone && opts.BiasLevel.IsNone && opts.DarkFrame.IsNone && opts.FlatFrame.IsNone then
            return Ok None
        else
            printfn "Loading calibration masters..."

            let calConfig: Algorithms.Calibration.CalibrationConfig = {
                BiasFrame = opts.BiasFrame
                BiasLevel = opts.BiasLevel
                DarkFrame = opts.DarkFrame
                FlatFrame = opts.FlatFrame
                UncalibratedDark = opts.UncalibratedDark
                UncalibratedFlat = opts.UncalibratedFlat
                OptimizeDark = opts.OptimizeDark
                OutputPedestal = opts.OutputPedestal
            }

            let! mastersResult = Calibration.loadMasterFrames calConfig

            return
                mastersResult
                |> Result.map (fun masters -> Some (masters, calConfig))
                |> Result.mapError (fun err -> err.ToString())
    }

/// Process reference image through pipeline
let analyzeReference
    (filePath: string)
    (masters: (MasterFrames * Algorithms.Calibration.CalibrationConfig) option)
    (binConfig: BinningConfig)
    (detConfig: DetectionConfig)
    (maxStarsTriangles: int)
    : Async<Result<ReferenceData, PipelineError>> =

    async {
        let! loadResult = loadImage filePath

        return
            loadResult
            |> Result.mapError LoadFailed
            |> Result.bind (fun img ->
                let origWidth = img.Width
                let origHeight = img.Height

                printfn $"Image dimensions: {origWidth}x{origHeight}, {img.Channels} channel(s)"

                preprocessImage masters binConfig img
                |> Result.map (fun preprocessed ->
                    if binConfig.Factor.IsSome && preprocessed.Width <> origWidth then
                        printfn $"Binned reference: {origWidth}x{origHeight} -> {preprocessed.Width}x{preprocessed.Height} ({binConfig.Factor.Value}x{binConfig.Factor.Value} {binConfig.Method})"

                    let frame = Alignment.prepareReference detConfig maxStarsTriangles preprocessed
                    printfn $"Detecting stars in reference (MAD: {frame.Detected.MAD:F2})..."
                    printfn $"Reference triangles formed: {frame.Triangles.Length}"

                    {
                        Frame = frame
                        FileName = Path.GetFileName filePath
                        DetectionParams = {
                            Threshold = detConfig.Threshold
                            GridSize = detConfig.GridSize
                            MinFWHM = detConfig.MinFWHM
                            MaxFWHM = detConfig.MaxFWHM
                            MaxEccentricity = detConfig.MaxEccentricity
                            MaxStars = Some detConfig.MaxStars
                        }
                    }
                )
            )
    }

/// Process single file in distortion visualization mode
let determineSuffix (mode: OutputMode) (defaultSuffix: string) : string =
    if defaultSuffix <> "_a" then defaultSuffix
    else match mode with
         | Detect -> "_det"
         | Match -> "_mat"
         | Distortion -> "_dist"
         | Align -> "_a"

let generateOutput
    (config: OutputConfig)
    (originalImg: XisfImage)
    (detConfig: DetectionConfig)
    (detected: DetectedImage)
    (matched: MatchedImage option)
    (distortion: DistortionResult option)
    (transformed: ImageData option)
    : Result<XisfImage, PipelineError> =
    match config.Mode with
    | Detect ->
        let format = config.Format |> Option.defaultValue originalImg.SampleFormat
        let pixels = generateDetectOutput detected config.Intensity format
        let headers = [|
            XisfFitsKeyword("STARCOUNT", detected.Stars.Length.ToString(), "Number of detected stars") :> XisfCoreElement
            XisfFitsKeyword("STARTHRES", sprintf "%.1f" detConfig.Threshold, "Detection threshold (sigma)") :> XisfCoreElement
            XisfFitsKeyword("STARGRID", detConfig.GridSize.ToString(), "Background grid size") :> XisfCoreElement
            XisfFitsKeyword("HISTORY", "", "Star detection by XisfPrep Align (detect mode)") :> XisfCoreElement
        |]
        Ok (createOutputImage originalImg pixels detected.Image.Width detected.Image.Height detected.Image.Channels format headers)

    | Match ->
        match matched with
        | None -> Error (MatchingFailed (TransformationFailed "No matching results"))
        | Some m ->
            let format = config.Format |> Option.defaultValue originalImg.SampleFormat
            let pixels = generateMatchOutput m m.Detected.Stars config.Intensity format
            let matchedCount = m.MatchedPairs.Length
            let unmatchedCount = detected.Stars.Length - matchedCount
            let baseHeaders = [|
                XisfFitsKeyword("IMAGETYP", "MATCHMAP", "Type of image") :> XisfCoreElement
                XisfFitsKeyword("SWCREATE", "XisfPrep Align", "Software") :> XisfCoreElement
                XisfFitsKeyword("TGTstars", detected.Stars.Length.ToString(), "Target stars") :> XisfCoreElement
                XisfFitsKeyword("MATCHED", matchedCount.ToString(), "Matched pairs") :> XisfCoreElement
                XisfFitsKeyword("UNMATCHD", unmatchedCount.ToString(), "Unmatched stars") :> XisfCoreElement
            |]
            let transformHeaders = [|
                XisfFitsKeyword("XSHIFT", sprintf "%.2f" m.Transform.Dx, "X shift (px)") :> XisfCoreElement
                XisfFitsKeyword("YSHIFT", sprintf "%.2f" m.Transform.Dy, "Y shift (px)") :> XisfCoreElement
                XisfFitsKeyword("ROTATION", sprintf "%.4f" m.Transform.Rotation, "Rotation (deg)") :> XisfCoreElement
                XisfFitsKeyword("SCALE", sprintf "%.6f" m.Transform.Scale, "Scale") :> XisfCoreElement
                XisfFitsKeyword("HISTORY", "", "Match by XisfPrep") :> XisfCoreElement
            |]
            let headers = Array.append baseHeaders transformHeaders
            Ok (createOutputImage originalImg pixels detected.Image.Width detected.Image.Height detected.Image.Channels format headers)

    | Align ->
        match (matched, transformed) with
        | (Some m, Some t) ->
            let pixels = generateAlignOutput t
            let headers = [|
                XisfFitsKeyword("ALIGNED", "T", "Aligned") :> XisfCoreElement
                XisfFitsKeyword("XSHIFT", sprintf "%.2f" m.Transform.Dx, "X shift (px)") :> XisfCoreElement
                XisfFitsKeyword("YSHIFT", sprintf "%.2f" m.Transform.Dy, "Y shift (px)") :> XisfCoreElement
                XisfFitsKeyword("ROTATION", sprintf "%.4f" m.Transform.Rotation, "Rotation (deg)") :> XisfCoreElement
                XisfFitsKeyword("SCALE", sprintf "%.6f" m.Transform.Scale, "Scale") :> XisfCoreElement
                XisfFitsKeyword("STARMTCH", m.MatchedPairs.Length.ToString(), "Matched pairs") :> XisfCoreElement
                XisfFitsKeyword("HISTORY", "", "Aligned by XisfPrep") :> XisfCoreElement
            |]
            Ok (createOutputImage originalImg pixels t.Width t.Height t.Channels XisfSampleFormat.Float32 headers)
        | _ -> Error (TransformFailed (TransformationFailed "Align requires transformation"))

    | Distortion ->
        match (matched, distortion) with
        | (Some m, Some rbfResult) ->
            match rbfResult.Coefficients with
            | None -> Error (MatchingFailed (TransformationFailed "No RBF coefficients"))
            | Some coeffs ->
                let width = detected.Image.Width
                let height = detected.Image.Height
                let pixels = Array.zeroCreate<byte> (width * height * 3 * 2)
                let maxMag = 10.0
                Array.Parallel.iter (fun pixIdx ->
                    let ox = pixIdx % width
                    let oy = pixIdx / width
                    let (tx, ty) = RBFTransform.applyFullInverseTransform m.Transform (Some coeffs) (float ox) (float oy) 5 0.01
                    let dx = float ox - tx
                    let dy = float oy - ty
                    let mag = sqrt (dx * dx + dy * dy)
                    let (r, g, b) = magnitudeToColor mag maxMag
                    let byteIdx = pixIdx * 6
                    pixels.[byteIdx] <- byte (r &&& 0xFFus)
                    pixels.[byteIdx + 1] <- byte (r >>> 8)
                    pixels.[byteIdx + 2] <- byte (g &&& 0xFFus)
                    pixels.[byteIdx + 3] <- byte (g >>> 8)
                    pixels.[byteIdx + 4] <- byte (b &&& 0xFFus)
                    pixels.[byteIdx + 5] <- byte (b >>> 8)
                ) [| 0 .. (width * height - 1) |]
                for (cx, cy) in coeffs.ControlPoints do
                    for ch in 0 .. 2 do
                        Painting.paintCircle pixels width height 3 ch cx cy 3.0 1.0 XisfSampleFormat.UInt16
                let geometry = XisfImageGeometry([| uint32 width; uint32 height |], 3u)
                let dataBlock = InlineDataBlock(ReadOnlyMemory(pixels), XisfEncoding.Base64)
                let bounds = PixelIO.getBoundsForFormat XisfSampleFormat.UInt16 |> Option.defaultValue (Unchecked.defaultof<XisfImageBounds>)
                let distImage = XisfImage(
                    geometry, XisfSampleFormat.UInt16, XisfColorSpace.RGB, dataBlock, bounds,
                    originalImg.PixelStorage, originalImg.ImageType, originalImg.Offset,
                    originalImg.Orientation, originalImg.ImageId, originalImg.Uuid,
                    originalImg.Properties, [||]
                )
                Ok distImage
        | _ -> Error (MatchingFailed (TransformationFailed "Distortion requires results"))


let processFile
    (inputPath: string)
    (outputPath: string)
    (configs: PipelineConfigs)
    : Async<Result<unit, PipelineError>> =
    asyncResult {
        let fileName = Path.GetFileName inputPath

        let reader = new XisfReader()
        let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask |> AsyncResult.ofAsync
        let originalImg = unit.Images.[0]
        let rawImage = {
            Pixels = PixelIO.readPixelsAsFloat originalImg
            Width = int originalImg.Geometry.Width
            Height = int originalImg.Geometry.Height
            Channels = int originalImg.Geometry.ChannelCount
        }

        let! calibrated = calibrate configs.Calibration rawImage |> AsyncResult.ofResult
        let binConfig = configs.Binning |> Option.defaultValue { Factor = None; Method = Average }
        let! binned = bin binConfig calibrated |> AsyncResult.ofResult

        let detected = Alignment.detect configs.Detection binned
        let! matched = matchStars configs.Matching fileName detected |> AsyncResult.ofResult
        let! distortion = applyDistortion configs.Distortion matched |> AsyncResult.ofResult
        let! transformed = applyTransform configs.Transform matched distortion |> AsyncResult.ofResult

        let! outputImage = generateOutput configs.Output originalImg configs.Detection detected matched distortion transformed |> AsyncResult.ofResult
        do! writeOutputFile outputPath outputImage |> AsyncResult.mapError SaveFailed
    }

let processBatch
    (files: string[])
    (opts: PreProcessOptions)
    (configs: PipelineConfigs)
    : Async<int> =
    async {
        Directory.CreateDirectory opts.Output |> ignore
        let suffix = determineSuffix configs.Output.Mode opts.Suffix

        let tasks = files |> Array.map (fun f ->
            let baseName = Path.GetFileNameWithoutExtension f
            let outPath = Path.Combine(opts.Output, $"{baseName}{suffix}.xisf")
            if File.Exists outPath && not opts.Overwrite then
                async { return Ok () }
            else
                processFile f outPath configs
        )

        let sw = Stopwatch.StartNew()
        let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)
        sw.Stop()

        let successCount = results |> Array.filter Result.isOk |> Array.length
        let failCount = files.Length - successCount
        results |> Array.iter (fun r -> match r with Error err -> Log.Error($"{err}") | Ok () -> ())
        printfn ""
        printfn $"Processed {successCount} images, {failCount} failed in {sw.Elapsed.TotalSeconds:F1}s"

        return if successCount < files.Length then 1 else 0
    }

let showHelp() =
    printfn "preprocess - Unified preprocessing pipeline with calibration, binning, and registration"
    printfn ""
    printfn "Usage: xisfprep preprocess [options]"
    printfn ""
    printfn "Description:"
    printfn "  Railway-oriented preprocessing pipeline supporting multiple output modes."
    printfn "  Pipeline: Load → Calibrate → Bin → Detect → Match → Distort → Transform → Save"
    printfn "  Each step is optional and configurable based on output mode."
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for processed files"
    printfn ""
    printfn "Reference Selection:"
    printfn "  --reference, -r <file>    Reference frame for registration (first file if omitted)"
    printfn "  --auto-reference          Auto-select best reference (highest star count/SNR)"
    printfn ""
    printfn "Output Mode:"
    printfn "  --output-mode <mode>      Processing pipeline mode (default: align)"
    printfn "                              detect     - Star detection only (calibrate → bin → detect)"
    printfn "                              match      - Star matching visualization (+ match)"
    printfn "                              align      - Full registration (+ distort → transform)"
    printfn "                              distortion - Distortion field visualization"
    printfn ""
    printfn "Alignment Parameters:"
    printfn "  --interpolation <method>  Resampling method (default: lanczos3)"
    printfn "                              nearest  - Nearest neighbor (preserves values)"
    printfn "                              bilinear - Bilinear (smooth, fast)"
    printfn "                              bicubic  - Bicubic (4x4 Catmull-Rom)"
    printfn "                              lanczos3 - Lanczos (6x6, best quality)"
    printfn ""
    printfn "Detection Parameters:"
    printfn $"  --threshold <sigma>       Detection threshold in sigma (default: {defaultThreshold})"
    printfn $"  --grid-size <px>          Background grid size in pixels (default: {defaultGridSize})"
    printfn $"  --min-fwhm <px>           Minimum FWHM filter in pixels (default: {defaultMinFWHM})"
    printfn $"  --max-fwhm <px>           Maximum FWHM filter in pixels (default: {defaultMaxFWHM})"
    printfn $"  --max-eccentricity <val>  Maximum eccentricity filter (default: {defaultMaxEccentricity})"
    printfn $"  --max-stars <n>           Maximum stars to detect (default: {defaultMaxStars})"
    printfn ""
    printfn "Matching Parameters:"
    printfn "  --algorithm <type>        Matching algorithm (default: expanding)"
    printfn "                              triangle  - Triangle ratio matching"
    printfn "                              expanding - Center-seeded expanding match"
    printfn $"  --anchor-stars <n>        Anchor stars for expanding algorithm (default: {defaultAnchorStars})"
    printfn "  --anchor-spread <type>    Anchor distribution (default: center)"
    printfn "                              center - Central region only"
    printfn "                              grid   - Distributed across 3x3 grid"
    printfn $"  --ratio-tolerance <val>   Triangle ratio matching tolerance (default: {defaultRatioTolerance})"
    printfn $"  --max-stars-triangles <n> Stars used for triangle formation (default: {defaultMaxStarsTriangles})"
    printfn $"  --min-votes <n>           Minimum votes for valid correspondence (default: {defaultMinVotes})"
    printfn ""
    printfn "Distortion Correction:"
    printfn "  --distortion <type>       Correction method (default: none)"
    printfn "                              none     - Similarity transform only"
    printfn "                              wendland - Local RBF correction (recommended)"
    printfn "                              tps      - Thin-plate spline (global)"
    printfn "                              imq      - Inverse multiquadric"
    printfn $"  --rbf-support <factor>    Wendland support radius factor (default: {defaultRBFSupportFactor})"
    printfn $"  --rbf-regularization <v>  Regularization parameter (default: {defaultRBFRegularization})"
    printfn ""
    printfn "Visualization (detect/match modes):"
    printfn $"  --intensity <0-1>         Marker brightness (default: {defaultIntensity})"
    printfn ""
    printfn "Output:"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Diagnostic Outputs:"
    printfn "  --show-distortion-stats   Print console heatmap in distortion mode"
    printfn "  --include-distortion-model Output distortion heatmap alongside aligned images"
    printfn "  --include-detection-model  Output star detection visualization alongside aligned images"
    printfn ""
    printfn "Calibration (applied before star detection):"
    printfn "  --bias, -b <file>         Master bias frame"
    printfn "  --bias-level <value>      Constant bias value (alternative to --bias)"
    printfn "  --dark, -d <file>         Master dark frame"
    printfn "  --flat, -f <file>         Master flat frame"
    printfn "  --uncalibrated-dark       Dark is raw (not bias-subtracted)"
    printfn "  --uncalibrated-flat       Flat is raw (not bias/dark-subtracted)"
    printfn "  --optimize-dark           Optimize dark scaling for temperature/exposure differences"
    printfn "  --pedestal <value>        Output pedestal [0-65535] (default: 0)"
    printfn ""
    printfn "Binning (applied after calibration, before star detection):"
    printfn "  --bin-factor <2-6>        Bin images by NxN factor (default: no binning)"
    printfn "                              2 - 2x2 binning (4x faster)"
    printfn "                              3 - 3x3 binning (9x faster)"
    printfn "                              4 - 4x4 binning (16x faster)"
    printfn "  --bin-method <method>     Binning method (default: average)"
    printfn "                              average - Average binning (recommended)"
    printfn "                              median  - Median binning (robust to outliers)"
    printfn "                              sum     - Sum binning (adds pixel values)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\""
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"validation/\" --output-mode detect"
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"validation/\" --output-mode match --intensity 0.8"
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"aligned/\" --bin-factor 2  # Quick 2x2 binned alignment"
    printfn "  xisfprep preprocess -i \"images/*.xisf\" -o \"preview/\" --bin-factor 4 --bin-method median  # Fast preview"

let run (args: string array) =
    if args |> Array.exists (fun a -> a = "--help" || a = "-h") then
        showHelp()
        0
    else
        let setupResult = asyncResult {
            let opts = parseArgs args

            do! validatePreProcessOptions opts |> Result.mapError ValidationFailed
            do! validateCalibrationConfig opts |> Result.mapError CalibrationValidationFailed

            let! files = InputValidation.resolveInputFiles opts.Input
                         |> Result.mapError InputResolutionFailed
            printfn $"Found {files.Length} files"
            printfn ""

            let! masters = loadCalibrationMasters opts |> AsyncResult.mapError MasterLoadFailed

            let binConfig: BinningConfig = {
                Factor = opts.BinFactor
                Method = opts.BinMethod
            }

            let detConfig: DetectionConfig = {
                Threshold = opts.Threshold
                GridSize = opts.GridSize
                MinFWHM = opts.MinFWHM
                MaxFWHM = opts.MaxFWHM
                MaxEccentricity = opts.MaxEccentricity
                MaxStars = opts.MaxStars
            }

            let refFile = resolveReferenceFile files opts
            printfn $"Reference: {Path.GetFileName refFile}"

            let! refData = analyzeReference refFile masters binConfig detConfig opts.MaxStarsTriangles
                           |> AsyncResult.mapError ReferenceAnalysisFailed

            do! validateReferenceStars refData.Frame.Detected.Stars.Length
                |> Result.mapError (fun _ -> InsufficientStars (refData.Frame.Detected.Stars.Length, 10))

            printfn $"Reference stars detected: {refData.Frame.Detected.Stars.Length}"
            printfn $"Reference triangles formed: {refData.Frame.Triangles.Length}"
            printfn ""

            let pipelineConfigs = buildPipelineConfigs opts.OutputMode opts masters refData

            printfn $"Output mode: {opts.OutputMode}"
            printfn ""

            return (opts, files, pipelineConfigs)
        }

        async {
            match! setupResult with
            | Error err ->
                Log.Error(err.ToString())
                printfn ""
                printfn "Run 'xisfprep preprocess --help' for usage information"
                return 1
            | Ok (opts, files, configs) ->
                let! exitCode = processBatch files opts configs
                return exitCode
        }
        |> Async.RunSynchronously
