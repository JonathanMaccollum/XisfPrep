module Commands.Align

open System
open System.Diagnostics
open System.IO
open Serilog
open XisfLib.Core
open Algorithms
open Algorithms.Painting
open Algorithms.Statistics
open Algorithms.OutputImage
open Algorithms.TriangleMatch
open Algorithms.SpatialMatch
open Algorithms.Interpolation
open Algorithms.RBFTransform
open Algorithms.Calibration
open Algorithms.InputValidation
open Algorithms.Binning

// Type aliases for star detection types
type DetectedStar = Algorithms.StarDetection.DetectedStar
type DetectionParams = Algorithms.StarDetection.DetectionParams

// Algorithm selection
type MatchAlgorithm =
    | Triangle   // Original triangle matching (default)
    | Expanding  // Center-seeded expanding match

// Re-export interpolation method from Algorithms
type InterpolationMethod = Algorithms.Interpolation.InterpolationMethod

type OutputMode =
    | Detect      // Show detected stars only
    | Match       // Show matched star correspondences
    | Align       // Apply transformation (default)
    | Distortion  // Visualize distortion field

type DistortionCorrection =
    | NoDistortion   // Similarity only
    | Wendland       // Compact support (recommended)
    | TPS            // Thin-plate spline
    | IMQ            // Inverse multiquadric

// --- Defaults ---
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

type AlignOptions = {
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

// --- Helper Functions ---

/// Apply calibration if masters available
let private applyCalibration (pixelsRaw: float[]) (masters: (MasterFrames * CalibrationConfig) option) : float[] =
    match masters with
    | Some (m, calConfig) ->
        let result = calibratePixels pixelsRaw m calConfig
        result.CalibratedPixels
    | None ->
        pixelsRaw

/// Apply binning if factor specified (after calibration)
let private applyBinning
    (pixels: float[])
    (width: int)
    (height: int)
    (channels: int)
    (binFactor: int option)
    (binMethod: BinningMethod)
    : (float[] * int * int) =
    match binFactor with
    | None ->
        (pixels, width, height)
    | Some factor ->
        let config: BinningConfig = {
            Factor = factor
            Method = binMethod
        }

        // Validate and bin
        match validateConfig config with
        | Error err ->
            Log.Warning($"Binning validation failed: {err}, skipping binning")
            (pixels, width, height)
        | Ok () ->
            match validateDimensions width height channels factor with
            | Error err ->
                Log.Warning($"Binning dimensions invalid: {err}, skipping binning")
                (pixels, width, height)
            | Ok _truncated ->
                let result = binPixels pixels width height channels config
                (result.BinnedPixels, result.NewWidth, result.NewHeight)

/// Calculate median and MAD for an array
let private calculateChannelMAD (values: float[]) =
    if values.Length = 0 then 0.0
    else
        let sorted = Array.sort values
        let mid = sorted.Length / 2
        let median =
            if sorted.Length % 2 = 0 then (sorted.[mid - 1] + sorted.[mid]) / 2.0
            else sorted.[mid]
        calculateMAD values median

open FsToolkit.ErrorHandling

// Type alias for cleaner signatures
type AsyncResult<'T, 'E> = Async<Result<'T, 'E>>

/// Domain errors for align command
type AlignError =
    | InputValidation of InputValidationError
    | InsufficientStars of count: int * required: int
    | ImageLoadFailed of file: string * reason: string
    | CalibrationFailed of CalibrationError

    override this.ToString() =
        match this with
        | InputValidation err ->
            err.ToString()
        | InsufficientStars (count, required) ->
            $"Insufficient stars in reference image: found {count}, need at least {required}"
        | ImageLoadFailed (file, reason) ->
            $"Failed to load image '{file}': {reason}"
        | CalibrationFailed err ->
            $"Failed to load calibration masters: {err}"

/// Align command validation errors
type AlignValidationError =
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

/// Reference image metadata
type ReferenceMetadata = {
    FileName: string
    Width: int
    Height: int
    Channels: int
}

/// Complete reference analysis result
type ReferenceData = {
    Stars: DetectedStar[]
    Triangles: Triangle[]
    Metadata: ReferenceMetadata
    DetectionParams: DetectionParams
    Masters: (MasterFrames * CalibrationConfig) option
}

/// Encapsulates all state needed for batch processing
type ProcessingContext = {
    Opts: AlignOptions
    Files: string[]
    OutputDirectory: string
    Reference: ReferenceData
}

/// Result of processing a batch
type ProcessingResult = {
    TotalFiles: int
    SuccessCount: int
    FailedCount: int
    Duration: System.TimeSpan
}

/// Load calibration masters if configured
let private loadCalibrationMasters (opts: AlignOptions) : AsyncResult<(MasterFrames * CalibrationConfig) option, AlignError> =
    asyncResult {
        if opts.BiasFrame.IsNone && opts.DarkFrame.IsNone && opts.FlatFrame.IsNone then
            return None
        else
            let calConfig: CalibrationConfig = {
                BiasFrame = opts.BiasFrame
                BiasLevel = opts.BiasLevel
                DarkFrame = opts.DarkFrame
                FlatFrame = opts.FlatFrame
                UncalibratedDark = opts.UncalibratedDark
                UncalibratedFlat = opts.UncalibratedFlat
                OptimizeDark = opts.OptimizeDark
                OutputPedestal = opts.OutputPedestal
            }

            printfn "Loading calibration masters..."

            let! m = loadMasterFrames calConfig
                     |> AsyncResult.mapError CalibrationFailed

            return Some (m, calConfig)
    }

/// Analyze reference image: load, calibrate, detect stars
let private analyzeReference
    (file: string)
    (opts: AlignOptions)
    : AsyncResult<ReferenceData, AlignError> =
    asyncResult {
        printfn $"Reference: {Path.GetFileName file}"

        try
            // 1. Load reference image
            let reader = new XisfReader()
            let! refUnit = reader.ReadAsync(file)
                           |> Async.AwaitTask
                           |> AsyncResult.ofAsync
            let refImage = refUnit.Images.[0]

            let width = int refImage.Geometry.Width
            let height = int refImage.Geometry.Height
            let channels = int refImage.Geometry.ChannelCount

            printfn $"Image dimensions: {width}x{height}, {channels} channel(s)"

            // 2. Load calibration masters
            let! masters = loadCalibrationMasters opts

            // 3. Calibrate pixels
            let refPixelsRaw = PixelIO.readPixelsAsFloat refImage
            let refPixels = applyCalibration refPixelsRaw masters

            // 4. Apply binning (after calibration)
            let (refPixelsFinal, widthFinal, heightFinal) =
                applyBinning refPixels width height channels opts.BinFactor opts.BinMethod

            if opts.BinFactor.IsSome then
                printfn $"Binned reference: {width}x{height} -> {widthFinal}x{heightFinal} ({opts.BinFactor.Value}x{opts.BinFactor.Value} {opts.BinMethod})"

            // 5. Extract luminance channel
            let refChannel0 =
                if channels = 1 then refPixelsFinal
                else Array.init (widthFinal * heightFinal) (fun i -> refPixelsFinal.[i * channels])

            let refMAD = calculateChannelMAD refChannel0
            printfn $"Detecting stars in reference (MAD: {refMAD:F2})..."

            // 6. Build detection params
            let detectionParams: DetectionParams = {
                Threshold = opts.Threshold
                GridSize = opts.GridSize
                MinFWHM = opts.MinFWHM
                MaxFWHM = opts.MaxFWHM
                MaxEccentricity = opts.MaxEccentricity
                MaxStars = Some opts.MaxStars
            }

            // 7. Detect stars (using binned dimensions)
            let refStarsResult =
                StarDetection.detectStarsInChannel
                    refChannel0 widthFinal heightFinal refMAD 0 "Luminance" detectionParams

            let stars = refStarsResult.Stars
            printfn $"Reference stars detected: {stars.Length}"

            // 8. Validate star count (railway switch point)
            if stars.Length < 10 then
                return! Error (InsufficientStars (stars.Length, 10)) |> AsyncResult.ofResult
            else
                // 9. Form triangles
                let triangles = formTriangles stars opts.MaxStarsTriangles
                printfn $"Reference triangles formed: {triangles.Length}"
                printfn ""

                return {
                    Stars = stars
                    Triangles = triangles
                    Metadata = {
                        FileName = Path.GetFileName file
                        Width = widthFinal  // Use binned dimensions
                        Height = heightFinal
                        Channels = channels
                    }
                    DetectionParams = detectionParams
                    Masters = masters
                }
        with ex ->
            return! Error (ImageLoadFailed (Path.GetFileName file, ex.Message)) |> AsyncResult.ofResult
    }

/// Determine reference file from options
let private resolveReferenceFile (files: string[]) (opts: AlignOptions) : string =
    match opts.Reference with
    | Some r -> r
    | None ->
        if opts.AutoReference then
            printfn "Auto-reference selection not yet implemented, using first file"
        files.[0]

// --- Batch Processing Orchestration ---

/// Execute batch processing for any mode
/// Takes a task factory function and returns structured results
let private runBatch
    (ctx: ProcessingContext)
    (suffix: string)
    (mode: string)
    (taskFactory: string -> Async<bool>)
    : AsyncResult<ProcessingResult, AlignError> =
    asyncResult {
        // 1. Setup output directory
        if not (Directory.Exists ctx.OutputDirectory) then
            Directory.CreateDirectory(ctx.OutputDirectory) |> ignore
            Log.Information($"Created output directory: {ctx.OutputDirectory}")

        printfn $"Output mode: {mode}"
        printfn ""

        // 2. Execute tasks in parallel
        let sw = Stopwatch.StartNew()
        let tasks = ctx.Files |> Array.map taskFactory
        let! results = Async.Parallel(tasks, maxDegreeOfParallelism = ctx.Opts.MaxParallel)
                       |> AsyncResult.ofAsync
        sw.Stop()

        // 3. Aggregate results
        let successCount = results |> Array.filter id |> Array.length
        let failCount = results.Length - successCount

        // 4. Report
        printfn ""
        let verb = if mode.Contains("align") then "Aligned" else "Processed"
        printfn $"{verb} {successCount} images, {failCount} failed in {sw.Elapsed.TotalSeconds:F1}s"

        return {
            TotalFiles = results.Length
            SuccessCount = successCount
            FailedCount = failCount
            Duration = sw.Elapsed
        }
    }

// --- Image Transformation ---

/// Transform image pixels using similarity transform and interpolation
/// Applies inverse transform to resample target image into reference frame
let transformImage
    (targetPixels: float[])
    (width: int)
    (height: int)
    (channels: int)
    (transform: SimilarityTransform)
    (rbfCoeffs: RBFTransform.RBFCoefficients option)
    (interpolation: InterpolationMethod)
    : float[] =

    let pixelCount = width * height

    // Pre-extract channels for efficient sampling
    let channelArrays =
        if channels = 1 then
            [| targetPixels |]
        else
            [| for ch in 0 .. channels - 1 ->
                Array.init pixelCount (fun i -> targetPixels.[i * channels + ch]) |]

    // Transform in parallel - each output pixel is independent
    let outputPixels = Array.zeroCreate (pixelCount * channels)

    Array.Parallel.iter (fun pixIdx ->
        let ox = pixIdx % width
        let oy = pixIdx / width

        // Transform output coords to target coords (with RBF if available)
        let (tx, ty) = RBFTransform.applyFullInverseTransform transform rbfCoeffs (float ox) (float oy) 5 0.01

        // Sample each channel
        for ch in 0 .. channels - 1 do
            let value = sample interpolation channelArrays.[ch] width height tx ty
            outputPixels.[pixIdx * channels + ch] <- value
    ) [| 0 .. pixelCount - 1 |]

    outputPixels

// --- Output Image Functions ---

/// Create FITS headers for detection output
let createDetectHeaders (starCount: int) (threshold: float) (gridSize: int) =
    [|
        XisfFitsKeyword("STARCOUNT", starCount.ToString(), "Number of detected stars") :> XisfCoreElement
        XisfFitsKeyword("STARTHRES", sprintf "%.1f" threshold, "Detection threshold (sigma)") :> XisfCoreElement
        XisfFitsKeyword("STARGRID", gridSize.ToString(), "Background grid size") :> XisfCoreElement
        XisfFitsKeyword("HISTORY", "", "Star detection by XisfPrep Align (detect mode)") :> XisfCoreElement
    |]

/// Paint detected stars as circles on black background
let paintDetectedStars (pixels: byte[]) (width: int) (height: int) (channels: int) (format: XisfSampleFormat) (stars: DetectedStar[]) (intensity: float) =
    for star in stars do
        let radius = star.FWHM * 2.0
        for ch in 0 .. channels - 1 do
            paintCircle pixels width height channels ch star.X star.Y radius intensity format

/// Process single file in detect mode
let processDetectFile (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool) (outputFormat: XisfSampleFormat option) (detectionParams: DetectionParams) (intensity: float) (masters: (MasterFrames * CalibrationConfig) option) (binFactor: int option) (binMethod: BinningMethod) =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            if File.Exists(outPath) && not overwrite then
                Log.Warning($"Output '{outFileName}' exists, skipping (use --overwrite)")
                return true
            else

            // Read image
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Get pixel data, calibrate, bin, and detect stars
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let (pixelFloatsFinal, widthFinal, heightFinal) =
                applyBinning pixelFloats width height channels binFactor binMethod

            let channel0 =
                if channels = 1 then pixelFloatsFinal
                else Array.init (widthFinal * heightFinal) (fun i -> pixelFloatsFinal.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 widthFinal heightFinal mad 0 "Luminance" detectionParams
            let stars = starsResult.Stars

            // Create output (using binned dimensions)
            let format = outputFormat |> Option.defaultValue img.SampleFormat
            let pixels = createBlackPixels widthFinal heightFinal channels format
            paintDetectedStars pixels widthFinal heightFinal channels format stars intensity

            let headers = createDetectHeaders stars.Length detectionParams.Threshold detectionParams.GridSize
            let inPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
            let outputImage = createOutputImage img pixels format headers inPlaceKeys

            do! writeOutputFile outPath outputImage "XisfPrep Align v1.0"

            printfn $"  {stars.Length} stars -> {outFileName}"
            return true
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(inputPath)}: {ex.Message}")
            return false
    }

/// Create FITS headers for match output
let createMatchHeaders (targetStars: int) (matchedCount: int) (unmatchedCount: int) (refStars: int) (diag: AlignmentDiagnostics) =
    [|
        XisfFitsKeyword("IMAGETYP", "MATCHMAP", "Type of image") :> XisfCoreElement
        XisfFitsKeyword("SWCREATE", "XisfPrep Align", "Software that created this file") :> XisfCoreElement
        XisfFitsKeyword("TGTstars", targetStars.ToString(), "Stars detected in target") :> XisfCoreElement
        XisfFitsKeyword("REFSTARS", refStars.ToString(), "Stars in reference") :> XisfCoreElement
        XisfFitsKeyword("MATCHED", matchedCount.ToString(), "Matched star pairs") :> XisfCoreElement
        XisfFitsKeyword("UNMATCHD", unmatchedCount.ToString(), "Unmatched target stars") :> XisfCoreElement
        XisfFitsKeyword("TRIFORME", diag.TrianglesFormed.ToString(), "Triangles formed") :> XisfCoreElement
        XisfFitsKeyword("TRIMATCD", diag.TrianglesMatched.ToString(), "Triangles matched") :> XisfCoreElement
        XisfFitsKeyword("MATCHPCT", sprintf "%.1f" diag.MatchPercentage, "Triangle match percentage") :> XisfCoreElement
        XisfFitsKeyword("TOPVOTES", diag.TopVoteCount.ToString(), "Top correspondence votes") :> XisfCoreElement
        match diag.EstimatedTransform with
        | Some (dx, dy, rot, scale) ->
            XisfFitsKeyword("XSHIFT", sprintf "%.2f" dx, "Estimated X shift (pixels)") :> XisfCoreElement
            XisfFitsKeyword("YSHIFT", sprintf "%.2f" dy, "Estimated Y shift (pixels)") :> XisfCoreElement
            XisfFitsKeyword("ROTATION", sprintf "%.4f" rot, "Estimated rotation (degrees)") :> XisfCoreElement
            XisfFitsKeyword("SCALE", sprintf "%.6f" scale, "Estimated scale factor") :> XisfCoreElement
        | None -> ()
        XisfFitsKeyword("HISTORY", "", "Match visualization by XisfPrep Align (match mode)") :> XisfCoreElement
    |]

/// Paint match visualization: circles=matched, X=unmatched, gaussian=reference
let paintMatchVisualization
    (pixels: byte[]) (width: int) (height: int) (channels: int) (format: XisfSampleFormat)
    (targetStars: DetectedStar[]) (refStars: DetectedStar[])
    (matchedIndices: Set<int>) (intensity: float) =

    // Paint reference stars as gaussians (faint, shows where stars should be)
    let refIntensity = intensity * 0.3
    for star in refStars do
        for ch in 0 .. channels - 1 do
            paintGaussian pixels width height channels ch star.X star.Y star.FWHM refIntensity format

    // Paint target stars
    for i, star in targetStars |> Array.indexed do
        let size = star.FWHM * 2.0
        if matchedIndices.Contains i then
            // Matched: circle
            for ch in 0 .. channels - 1 do
                paintCircle pixels width height channels ch star.X star.Y size intensity format
        else
            // Unmatched: X marker
            for ch in 0 .. channels - 1 do
                paintX pixels width height channels ch star.X star.Y size intensity format

/// Process single file in match mode
let processMatchFile
    (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool)
    (outputFormat: XisfSampleFormat option) (detectionParams: DetectionParams)
    (refStars: DetectedStar[]) (refTriangles: Triangle[])
    (opts: AlignOptions) (masters: (MasterFrames * CalibrationConfig) option) =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            if File.Exists(outPath) && not overwrite then
                Log.Warning($"Output '{outFileName}' exists, skipping (use --overwrite)")
                return true
            else

            // Read image
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Get pixel data, calibrate, bin, and detect stars
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let (pixelFloatsFinal, widthFinal, heightFinal) =
                applyBinning pixelFloats width height channels opts.BinFactor opts.BinMethod

            let channel0 =
                if channels = 1 then pixelFloatsFinal
                else Array.init (widthFinal * heightFinal) (fun i -> pixelFloatsFinal.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 widthFinal heightFinal mad 0 "Luminance" detectionParams
            let targetStars = starsResult.Stars

            // Run matching based on selected algorithm
            let (diag, matchedTargetIndices) =
                match opts.Algorithm with
                | Triangle ->
                    // Original triangle matching
                    let diag = runDiagnostic refStars refTriangles targetStars fileName opts.RatioTolerance opts.MaxStarsTriangles opts.MinVotes

                    // Get matched target star indices from correspondences
                    let targetTriangles = formTriangles targetStars opts.MaxStarsTriangles
                    let matches = matchTriangles refTriangles targetTriangles opts.RatioTolerance
                    let correspondences = voteForCorrespondences matches

                    let matchedIndices =
                        correspondences
                        |> Map.toSeq
                        |> Seq.filter (fun ((_, _), votes) -> votes >= opts.MinVotes)
                        |> Seq.map (fun ((_, targetIdx), _) -> targetIdx)
                        |> Set.ofSeq

                    (diag, matchedIndices)

                | Expanding ->
                    // Center-seeded expanding match (use binned dimensions)
                    let config = {
                        defaultExpandingConfig with
                            AnchorStars = opts.AnchorStars
                            AnchorDistribution = opts.AnchorDistribution
                            RatioTolerance = opts.RatioTolerance
                            MinAnchorVotes = opts.MinVotes
                    }

                    let result = matchExpanding refStars targetStars widthFinal heightFinal config

                    // Convert to diagnostics format
                    let transform =
                        result.Transform
                        |> Option.map (fun t -> (t.Dx, t.Dy, t.Rotation, t.Scale))

                    let diag = {
                        FileName = fileName
                        DetectedStars = targetStars.Length
                        TrianglesFormed = 0  // Not applicable
                        TrianglesMatched = result.AnchorPairs
                        MatchPercentage = 0.0
                        TopVoteCount = result.TotalInliers
                        EstimatedTransform = transform
                    }

                    (diag, result.MatchedTargetIndices)

            let matchedCount = matchedTargetIndices.Count
            let unmatchedCount = targetStars.Length - matchedCount

            // Create output (using binned dimensions)
            let format = outputFormat |> Option.defaultValue img.SampleFormat
            let pixels = createBlackPixels widthFinal heightFinal channels format
            paintMatchVisualization pixels widthFinal heightFinal channels format targetStars refStars matchedTargetIndices opts.Intensity

            let headers = createMatchHeaders targetStars.Length matchedCount unmatchedCount refStars.Length diag
            let inPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
            let outputImage = createOutputImage img pixels format headers inPlaceKeys

            do! writeOutputFile outPath outputImage "XisfPrep Align v1.0"

            // Format transform info for output
            let transformStr =
                match diag.EstimatedTransform with
                | Some (dx, dy, rot, _) -> sprintf "dx=%.1f dy=%.1f rot=%.2f°" dx dy rot
                | None -> "N/A"

            let algoStr = match opts.Algorithm with Triangle -> "" | Expanding -> " [exp]"
            printfn $"  {matchedCount}/{targetStars.Length} matched -> {outFileName} ({transformStr}){algoStr}"
            return true
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(inputPath)}: {ex.Message}")
            return false
    }

/// Create FITS headers for aligned output
let createAlignHeaders (transform: SimilarityTransform) (matchedCount: int) (refFileName: string) (interpolation: string) =
    [|
        XisfFitsKeyword("ALIGNED", "T", "Image has been aligned") :> XisfCoreElement
        XisfFitsKeyword("ALIGNREF", refFileName, "Reference frame") :> XisfCoreElement
        XisfFitsKeyword("XSHIFT", sprintf "%.2f" transform.Dx, "X shift (pixels)") :> XisfCoreElement
        XisfFitsKeyword("YSHIFT", sprintf "%.2f" transform.Dy, "Y shift (pixels)") :> XisfCoreElement
        XisfFitsKeyword("ROTATION", sprintf "%.4f" transform.Rotation, "Rotation (degrees)") :> XisfCoreElement
        XisfFitsKeyword("SCALE", sprintf "%.6f" transform.Scale, "Scale factor") :> XisfCoreElement
        XisfFitsKeyword("STARMTCH", matchedCount.ToString(), "Matched star pairs") :> XisfCoreElement
        XisfFitsKeyword("INTERP", interpolation, "Interpolation method") :> XisfCoreElement
        XisfFitsKeyword("HISTORY", "", "Aligned by XisfPrep Align v1.0") :> XisfCoreElement
    |]

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

/// Process single file in align mode - apply transformation
let processAlignFile
    (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool)
    (detectionParams: DetectionParams)
    (refStars: DetectedStar[]) (refTriangles: Triangle[])
    (refFileName: string) (imageWidth: int) (imageHeight: int)
    (opts: AlignOptions) (masters: (MasterFrames * CalibrationConfig) option)
    : Async<bool> =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            if File.Exists(outPath) && not overwrite then
                Log.Warning($"Output '{outFileName}' exists, skipping (use --overwrite)")
                return true
            else

            // Read image
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Get pixel data, calibrate, and bin
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let (pixelFloatsFinal, widthFinal, heightFinal) =
                applyBinning pixelFloats width height channels opts.BinFactor opts.BinMethod

            // Verify dimensions match reference (after binning)
            if widthFinal <> imageWidth || heightFinal <> imageHeight then
                Log.Error($"Image dimensions {widthFinal}x{heightFinal} don't match reference {imageWidth}x{imageHeight}: {fileName}")
                return false
            else

            // Detect stars in binned image
            let channel0 =
                if channels = 1 then pixelFloatsFinal
                else Array.init (widthFinal * heightFinal) (fun i -> pixelFloatsFinal.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 widthFinal heightFinal mad 0 "Luminance" detectionParams
            let targetStars = starsResult.Stars

            // Run matching based on selected algorithm
            let matchResult =
                match opts.Algorithm with
                | Triangle ->
                    // Use triangle matching
                    let targetTriangles = formTriangles targetStars opts.MaxStarsTriangles
                    let matches = matchTriangles refTriangles targetTriangles opts.RatioTolerance
                    let correspondences = voteForCorrespondences matches
                    let sortedVotes = correspondences |> Map.toArray |> Array.sortByDescending snd

                    // Build pairs for transform estimation
                    let aboveThreshold = sortedVotes |> Array.filter (fun (_, v) -> v >= opts.MinVotes)
                    let bestPerRef =
                        aboveThreshold
                        |> Array.groupBy (fun ((refIdx, _), _) -> refIdx)
                        |> Array.map (fun (_, group) -> group |> Array.maxBy snd)
                    let bijective =
                        bestPerRef
                        |> Array.groupBy (fun ((_, targetIdx), _) -> targetIdx)
                        |> Array.map (fun (_, group) -> group |> Array.maxBy snd)

                    let matchedPairs = bijective |> Array.map (fun ((refIdx, targetIdx), _) -> (refIdx, targetIdx))

                    let pairs =
                        matchedPairs
                        |> Array.map (fun (refIdx, targetIdx) ->
                            let r = refStars.[refIdx]
                            let t = targetStars.[targetIdx]
                            (r.X, r.Y, t.X, t.Y))

                    match estimateTransformFromPairs pairs with
                    | Some t -> Some (t, matchedPairs)
                    | None -> None

                | Expanding ->
                    // Use expanding match algorithm (use binned dimensions)
                    let config = {
                        defaultExpandingConfig with
                            AnchorStars = opts.AnchorStars
                            AnchorDistribution = opts.AnchorDistribution
                            RatioTolerance = opts.RatioTolerance
                            MinAnchorVotes = opts.MinVotes
                    }

                    let result = matchExpanding refStars targetStars widthFinal heightFinal config

                    // Extract matched pairs by finding correspondences via inverse transform
                    match result.Transform with
                    | Some transform ->
                        let inverse = invertTransform transform
                        let targetTree =
                            targetStars
                            |> Array.indexed
                            |> Array.map (fun (i, s) -> (s, i))
                            |> buildKdTree

                        let matchedPairs = ResizeArray<int * int>()
                        for refIdx in 0 .. refStars.Length - 1 do
                            let refStar = refStars.[refIdx]
                            let (predX, predY) = applyTransform inverse refStar.X refStar.Y
                            match findNearest targetTree predX predY with
                            | Some (_, targetIdx, dist) when dist < 15.0 ->
                                matchedPairs.Add((refIdx, targetIdx))
                            | _ -> ()

                        Some (transform, matchedPairs.ToArray())
                    | None -> None

            match matchResult with
            | None ->
                Log.Error($"Failed to compute transform for {fileName}")
                return false
            | Some (transform, matchedPairs) ->
                // Compute RBF distortion correction if enabled
                let (rbfCoeffs, rbfInfo) =
                    match opts.Distortion with
                    | NoDistortion -> (None, "")
                    | _ ->
                        let rbfKernel =
                            match opts.Distortion with
                            | Wendland -> RBFTransform.Wendland 1
                            | TPS -> RBFTransform.ThinPlateSpline
                            | IMQ -> RBFTransform.InverseMultiquadric
                            | NoDistortion -> RBFTransform.Wendland 1

                        let rbfConfig: RBFTransform.RBFConfig = {
                            Kernel = rbfKernel
                            SupportRadiusFactor = opts.RBFSupportFactor
                            ShapeFactor = 0.5
                            Regularization = opts.RBFRegularization
                            MinControlPoints = 20
                            MaxControlPoints = 500
                        }

                        let rbfResult = RBFTransform.setupRBF refStars targetStars matchedPairs transform rbfConfig widthFinal heightFinal

                        match rbfResult.Coefficients with
                        | Some coeffs ->
                            let distName = match opts.Distortion with Wendland -> "wend" | TPS -> "tps" | IMQ -> "imq" | _ -> ""
                            (Some coeffs, sprintf " [%s rms=%.2f]" distName rbfResult.ResidualRMS)
                        | None ->
                            Log.Warning($"RBF setup failed for {fileName}, using similarity only")
                            (None, " [rbf failed]")

                // Apply transformation (to binned pixels)
                let transformedPixels = transformImage pixelFloatsFinal widthFinal heightFinal channels transform rbfCoeffs opts.Interpolation

                // Output as float32 for interpolation precision
                // normalize=true converts [0, 65535] → [0, 1] for Float32
                let outputFormat = XisfSampleFormat.Float32
                let outputBytes = PixelIO.writePixelsFromFloat transformedPixels outputFormat true

                // Get interpolation name for header
                let interpName =
                    match opts.Interpolation with
                    | Nearest -> "nearest"
                    | Bilinear -> "bilinear"
                    | Bicubic -> "bicubic"
                    | Lanczos3 -> "lanczos3"

                let headers = createAlignHeaders transform matchedPairs.Length refFileName interpName
                let inPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
                let outputImage = createOutputImage img outputBytes outputFormat headers inPlaceKeys

                do! writeOutputFile outPath outputImage "XisfPrep Align v1.0"

                // Output additional diagnostic files if requested
                if opts.IncludeDetectionModel then
                    let detFileName = $"{baseName}_det.xisf"
                    let detPath = Path.Combine(outputDir, detFileName)
                    let detPixels = createBlackPixels widthFinal heightFinal channels XisfSampleFormat.UInt16
                    paintDetectedStars detPixels widthFinal heightFinal channels XisfSampleFormat.UInt16 targetStars opts.Intensity
                    let detHeaders = createDetectHeaders targetStars.Length detectionParams.Threshold detectionParams.GridSize
                    let detInPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
                    let detOutputImage = createOutputImage img detPixels XisfSampleFormat.UInt16 detHeaders detInPlaceKeys
                    do! writeOutputFile detPath detOutputImage "XisfPrep Align v1.0"

                if opts.IncludeDistortionModel && rbfCoeffs.IsSome then
                    let distFileName = $"{baseName}_dist.xisf"
                    let distPath = Path.Combine(outputDir, distFileName)

                    // Generate distortion heatmap (using binned dimensions)
                    let distPixels = Array.zeroCreate<byte> (widthFinal * heightFinal * 3 * 2) // RGB * UInt16
                    let maxMag = 10.0

                    Array.Parallel.iter (fun pixIdx ->
                        let ox = pixIdx % widthFinal
                        let oy = pixIdx / widthFinal
                        let (tx, ty) = RBFTransform.applyFullInverseTransform transform rbfCoeffs (float ox) (float oy) 5 0.01
                        let dx = float ox - tx
                        let dy = float oy - ty
                        let mag = sqrt (dx * dx + dy * dy)
                        let (r, g, b) = magnitudeToColor mag maxMag
                        let byteIdx = pixIdx * 6
                        distPixels.[byteIdx] <- byte (r &&& 0xFFus)
                        distPixels.[byteIdx + 1] <- byte (r >>> 8)
                        distPixels.[byteIdx + 2] <- byte (g &&& 0xFFus)
                        distPixels.[byteIdx + 3] <- byte (g >>> 8)
                        distPixels.[byteIdx + 4] <- byte (b &&& 0xFFus)
                        distPixels.[byteIdx + 5] <- byte (b >>> 8)
                    ) [| 0 .. widthFinal * heightFinal - 1 |]

                    // Overlay control points
                    match rbfCoeffs with
                    | Some coeffs ->
                        for (cx, cy) in coeffs.ControlPoints do
                            for ch in 0 .. 2 do
                                paintCircle distPixels widthFinal heightFinal 3 ch cx cy 3.0 1.0 XisfSampleFormat.UInt16
                    | None -> ()

                    // Create RGB output image (using binned dimensions)
                    let distGeometry = XisfImageGeometry([| uint32 widthFinal; uint32 heightFinal |], 3u)
                    let distDataBlock = InlineDataBlock(ReadOnlyMemory(distPixels), XisfEncoding.Base64)
                    let distBounds = PixelIO.getBoundsForFormat XisfSampleFormat.UInt16 |> Option.defaultValue (Unchecked.defaultof<XisfImageBounds>)

                    let distOutputImage = XisfImage(
                        distGeometry,
                        XisfSampleFormat.UInt16,
                        XisfColorSpace.RGB,
                        distDataBlock,
                        distBounds,
                        XisfPixelStorage.Normal,
                        img.ImageType,
                        0.0,
                        img.Orientation,
                        "",
                        Nullable<Guid>(),
                        img.Properties,
                        [||]
                    )

                    let distMetadata = XisfFactory.CreateMinimalMetadata("XisfPrep Align v1.0")
                    let distUnit = XisfFactory.CreateMonolithic(distMetadata, distOutputImage)
                    let distWriter = new XisfWriter()
                    do! distWriter.WriteAsync(distUnit, distPath) |> Async.AwaitTask

                // Tidy output: dx=1.2 dy=-0.8 rot=0.12° -> filename_a.xisf
                printfn $"  dx={transform.Dx:F1} dy={transform.Dy:F1} rot={transform.Rotation:F2}° -> {outFileName}{rbfInfo}"
                return true
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(inputPath)}: {ex.Message}")
            return false
    }

// --- Distortion Visualization ---

/// Map magnitude to ASCII character for console display
let magnitudeToAscii (mag: float) : char =
    if mag < 2.0 then '·'
    elif mag < 4.0 then '░'
    elif mag < 6.0 then '▒'
    elif mag < 8.0 then '▓'
    else '█'

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

/// Process single file in distortion visualization mode
let processDistortionFile
    (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool)
    (detectionParams: DetectionParams)
    (refStars: DetectedStar[]) (refTriangles: Triangle[])
    (imageWidth: int) (imageHeight: int)
    (opts: AlignOptions) (masters: (MasterFrames * CalibrationConfig) option)
    : Async<bool> =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            if File.Exists(outPath) && not overwrite then
                Log.Warning($"Output '{outFileName}' exists, skipping (use --overwrite)")
                return true
            else

            // Read image
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Calibrate and bin
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let (pixelFloatsFinal, widthFinal, heightFinal) =
                applyBinning pixelFloats width height channels opts.BinFactor opts.BinMethod

            if widthFinal <> imageWidth || heightFinal <> imageHeight then
                Log.Error($"Image dimensions don't match reference: {fileName}")
                return false
            else

            // Detect stars in binned image
            let channel0 =
                if channels = 1 then pixelFloatsFinal
                else Array.init (widthFinal * heightFinal) (fun i -> pixelFloatsFinal.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 widthFinal heightFinal mad 0 "Luminance" detectionParams
            let targetStars = starsResult.Stars

            // Run matching (use binned dimensions)
            let config = {
                defaultExpandingConfig with
                    AnchorStars = opts.AnchorStars
                    AnchorDistribution = opts.AnchorDistribution
                    RatioTolerance = opts.RatioTolerance
                    MinAnchorVotes = opts.MinVotes
            }

            let result = matchExpanding refStars targetStars widthFinal heightFinal config

            match result.Transform with
            | None ->
                Log.Error($"Failed to compute transform for {fileName}")
                return false
            | Some transform ->
                // Extract matched pairs
                let inverse = invertTransform transform
                let targetTree =
                    targetStars
                    |> Array.indexed
                    |> Array.map (fun (i, s) -> (s, i))
                    |> buildKdTree

                let matchedPairs = ResizeArray<int * int>()
                for refIdx in 0 .. refStars.Length - 1 do
                    let refStar = refStars.[refIdx]
                    let (predX, predY) = applyTransform inverse refStar.X refStar.Y
                    match findNearest targetTree predX predY with
                    | Some (_, targetIdx, dist) when dist < 15.0 ->
                        matchedPairs.Add((refIdx, targetIdx))
                    | _ -> ()

                // Compute RBF
                let rbfKernel =
                    match opts.Distortion with
                    | Wendland -> RBFTransform.Wendland 1
                    | TPS -> RBFTransform.ThinPlateSpline
                    | IMQ -> RBFTransform.InverseMultiquadric
                    | NoDistortion -> RBFTransform.Wendland 1

                let rbfConfig: RBFTransform.RBFConfig = {
                    Kernel = rbfKernel
                    SupportRadiusFactor = opts.RBFSupportFactor
                    ShapeFactor = 0.5
                    Regularization = opts.RBFRegularization
                    MinControlPoints = 20
                    MaxControlPoints = 500
                }

                let rbfResult = RBFTransform.setupRBF refStars targetStars (matchedPairs.ToArray()) transform rbfConfig widthFinal heightFinal

                // Print console heatmap for analysis if requested
                if opts.ShowDistortionStats then
                    printDistortionHeatmap widthFinal heightFinal transform rbfResult.Coefficients 10

                // Create RGB output for heatmap (using binned dimensions)
                let outputFormat = XisfSampleFormat.UInt16
                let pixelCount = widthFinal * heightFinal
                let pixels = Array.zeroCreate<byte> (pixelCount * 3 * 2) // RGB * UInt16

                // Sample distortion at each pixel and create heatmap
                let maxMag = 10.0 // Max distortion for color scale

                Array.Parallel.iter (fun pixIdx ->
                    let ox = pixIdx % widthFinal
                    let oy = pixIdx / widthFinal

                    let (tx, ty) = RBFTransform.applyFullInverseTransform transform rbfResult.Coefficients (float ox) (float oy) 5 0.01

                    let dx = float ox - tx
                    let dy = float oy - ty
                    let mag = sqrt (dx * dx + dy * dy)

                    let (r, g, b) = magnitudeToColor mag maxMag

                    let byteIdx = pixIdx * 6
                    // Little-endian UInt16
                    pixels.[byteIdx] <- byte (r &&& 0xFFus)
                    pixels.[byteIdx + 1] <- byte (r >>> 8)
                    pixels.[byteIdx + 2] <- byte (g &&& 0xFFus)
                    pixels.[byteIdx + 3] <- byte (g >>> 8)
                    pixels.[byteIdx + 4] <- byte (b &&& 0xFFus)
                    pixels.[byteIdx + 5] <- byte (b >>> 8)
                ) [| 0 .. pixelCount - 1 |]

                // Overlay control points as white circles
                match rbfResult.Coefficients with
                | Some coeffs ->
                    for (cx, cy) in coeffs.ControlPoints do
                        // Paint white circle
                        for ch in 0 .. 2 do
                            paintCircle pixels widthFinal heightFinal 3 ch cx cy 3.0 1.0 outputFormat
                | None -> ()

                // Create headers
                let distName = match opts.Distortion with Wendland -> "WENDLAND" | TPS -> "TPS" | IMQ -> "IMQ" | _ -> "NONE"
                let headers = [|
                    XisfFitsKeyword("IMAGETYP", "DISTMAP", "Distortion visualization") :> XisfCoreElement
                    XisfFitsKeyword("DISTCORR", distName, "Correction method") :> XisfCoreElement
                    XisfFitsKeyword("RBFPTS", rbfResult.ControlPointCount.ToString(), "Control points") :> XisfCoreElement
                    XisfFitsKeyword("RBFRMS", sprintf "%.3f" rbfResult.ResidualRMS, "RBF residual RMS (px)") :> XisfCoreElement
                    XisfFitsKeyword("RBFMAX", sprintf "%.3f" rbfResult.MaxResidual, "RBF max residual (px)") :> XisfCoreElement
                    XisfFitsKeyword("XSHIFT", sprintf "%.2f" transform.Dx, "X shift (pixels)") :> XisfCoreElement
                    XisfFitsKeyword("YSHIFT", sprintf "%.2f" transform.Dy, "Y shift (pixels)") :> XisfCoreElement
                    XisfFitsKeyword("ROTATION", sprintf "%.4f" transform.Rotation, "Rotation (degrees)") :> XisfCoreElement
                    XisfFitsKeyword("HISTORY", "", "Distortion map by XisfPrep Align") :> XisfCoreElement
                |]

                // Create RGB output image (using binned dimensions)
                let geometry = XisfImageGeometry([| uint32 widthFinal; uint32 heightFinal |], 3u)
                let dataBlock = InlineDataBlock(ReadOnlyMemory(pixels), XisfEncoding.Base64)
                let bounds = PixelIO.getBoundsForFormat outputFormat |> Option.defaultValue (Unchecked.defaultof<XisfImageBounds>)

                let outputImage = XisfImage(
                    geometry,
                    outputFormat,
                    XisfColorSpace.RGB,
                    dataBlock,
                    bounds,
                    XisfPixelStorage.Normal,
                    img.ImageType,
                    0.0,
                    img.Orientation,
                    "",
                    Nullable<Guid>(),
                    img.Properties,
                    headers
                )

                let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Align v1.0")
                let outUnit = XisfFactory.CreateMonolithic(metadata, outputImage)
                let writer = new XisfWriter()
                do! writer.WriteAsync(outUnit, outPath) |> Async.AwaitTask

                printfn $"  {rbfResult.ControlPointCount} pts, rms={rbfResult.ResidualRMS:F2}px -> {outFileName}"
                return true
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(inputPath)}: {ex.Message}")
            return false
    }

// ---

let showHelp() =
    printfn "align - Register images to reference frame"
    printfn ""
    printfn "Usage: xisfprep align [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for aligned files"
    printfn ""
    printfn "Reference Selection:"
    printfn "  --reference, -r <file>    Reference frame to align to (first file if omitted)"
    printfn "  --auto-reference          Auto-select best reference (highest star count/SNR)"
    printfn ""
    printfn "Output Mode:"
    printfn "  --output-mode <mode>      Output type (default: align)"
    printfn "                              detect     - Show detected stars only"
    printfn "                              match      - Show matched star correspondences"
    printfn "                              align      - Apply transformation (default)"
    printfn "                              distortion - Visualize distortion field"
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
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\""
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"validation/\" --output-mode detect"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"validation/\" --output-mode match --intensity 0.8"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --bin-factor 2  # Quick 2x2 binned alignment"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"preview/\" --bin-factor 4 --bin-method median  # Fast preview"

/// Validate align options
let private validateAlignOptions (opts: AlignOptions) : Result<unit, AlignValidationError> =
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

let parseArgs (args: string array) : AlignOptions =
    let rec parse (args: string list) (opts: AlignOptions) =
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
                       | _ -> failwithf "Unknown output mode: %s (supported: detect, match, align, distortion)" value
            parse rest { opts with OutputMode = mode }
        | "--interpolation" :: value :: rest ->
            let interp = match value.ToLower() with
                         | "nearest" -> Nearest
                         | "bilinear" -> Bilinear
                         | "bicubic" -> Bicubic
                         | "lanczos" | "lanczos3" -> Lanczos3
                         | _ -> failwithf "Unknown interpolation method: %s (supported: nearest, bilinear, bicubic, lanczos3)" value
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
            | None -> failwithf "Unknown output format: %s (supported: uint8, uint16, uint32, float32, float64)" value
        // Detection parameters
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
        // Matching parameters
        | "--algorithm" :: value :: rest ->
            let algo = match value.ToLower() with
                       | "triangle" -> Triangle
                       | "expanding" -> Expanding
                       | _ -> failwithf "Unknown algorithm: %s (supported: triangle, expanding)" value
            parse rest { opts with Algorithm = algo }
        | "--anchor-stars" :: value :: rest ->
            parse rest { opts with AnchorStars = int value }
        | "--anchor-spread" :: value :: rest ->
            let dist = match value.ToLower() with
                       | "center" -> SpatialMatch.Center
                       | "grid" -> SpatialMatch.Grid
                       | _ -> failwithf "Unknown anchor spread: %s (supported: center, grid)" value
            parse rest { opts with AnchorDistribution = dist }
        | "--ratio-tolerance" :: value :: rest ->
            parse rest { opts with RatioTolerance = float value }
        | "--max-stars-triangles" :: value :: rest ->
            parse rest { opts with MaxStarsTriangles = int value }
        | "--min-votes" :: value :: rest ->
            parse rest { opts with MinVotes = int value }
        // Visualization
        | "--intensity" :: value :: rest ->
            parse rest { opts with Intensity = float value }
        // Distortion correction
        | "--distortion" :: value :: rest ->
            let dist = match value.ToLower() with
                       | "none" -> NoDistortion
                       | "wendland" -> Wendland
                       | "tps" -> TPS
                       | "imq" -> IMQ
                       | _ -> failwithf "Unknown distortion: %s (supported: none, wendland, tps, imq)" value
            parse rest { opts with Distortion = dist }
        | "--rbf-support" :: value :: rest ->
            parse rest { opts with RBFSupportFactor = float value }
        | "--rbf-regularization" :: value :: rest ->
            parse rest { opts with RBFRegularization = float value }
        // Additional outputs
        | "--show-distortion-stats" :: rest ->
            parse rest { opts with ShowDistortionStats = true }
        | "--include-distortion-model" :: rest ->
            parse rest { opts with IncludeDistortionModel = true }
        | "--include-detection-model" :: rest ->
            parse rest { opts with IncludeDetectionModel = true }
        // Calibration
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
        // Binning
        | "--bin-factor" :: value :: rest ->
            parse rest { opts with BinFactor = Some (int value) }
        | "--bin-method" :: value :: rest ->
            let method = match value.ToLower() with
                         | "average" -> Average
                         | "median" -> Median
                         | "sum" -> Sum
                         | _ -> failwithf "Unknown binning method: %s (supported: average, median, sum)" value
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

    let opts = parse (List.ofArray args) defaults

    // Validate align options using structured validation
    match validateAlignOptions opts with
    | Error err -> failwith (err.ToString())
    | Ok () -> ()

    // Validate calibration configuration using shared validation (if any calibration specified)
    if opts.BiasFrame.IsSome || opts.BiasLevel.IsSome || opts.DarkFrame.IsSome || opts.FlatFrame.IsSome || opts.OutputPedestal > 0 then
        let calConfig: CalibrationConfig = {
            BiasFrame = opts.BiasFrame
            BiasLevel = opts.BiasLevel
            DarkFrame = opts.DarkFrame
            FlatFrame = opts.FlatFrame
            UncalibratedDark = opts.UncalibratedDark
            UncalibratedFlat = opts.UncalibratedFlat
            OptimizeDark = opts.OptimizeDark
            OutputPedestal = opts.OutputPedestal
        }
        match Calibration.validateConfig calConfig with
        | Error err -> failwith (err.ToString())
        | Ok () -> ()

    opts

// --- Mode Execution ---

/// Execute the appropriate processing mode
let private executeMode (ctx: ProcessingContext) : AsyncResult<ProcessingResult, AlignError> =
    let opts = ctx.Opts
    let refData = ctx.Reference

    match opts.OutputMode with
    | Detect ->
        let suffix = if opts.Suffix = defaultSuffix then "_det" else opts.Suffix
        runBatch ctx suffix "detect (star visualization)" (fun f ->
            processDetectFile
                f ctx.OutputDirectory suffix opts.Overwrite
                opts.OutputFormat refData.DetectionParams
                opts.Intensity refData.Masters opts.BinFactor opts.BinMethod)

    | Match ->
        let suffix = if opts.Suffix = defaultSuffix then "_mat" else opts.Suffix
        runBatch ctx suffix "match (correspondence visualization)" (fun f ->
            processMatchFile
                f ctx.OutputDirectory suffix opts.Overwrite
                opts.OutputFormat refData.DetectionParams
                refData.Stars refData.Triangles
                opts refData.Masters)

    | Align ->
        runBatch ctx opts.Suffix "align (image transformation)" (fun f ->
            processAlignFile
                f ctx.OutputDirectory opts.Suffix opts.Overwrite
                refData.DetectionParams refData.Stars refData.Triangles
                refData.Metadata.FileName refData.Metadata.Width
                refData.Metadata.Height opts refData.Masters)

    | Distortion ->
        let suffix = if opts.Suffix = defaultSuffix then "_dist" else opts.Suffix
        runBatch ctx suffix "distortion (field visualization)" (fun f ->
            processDistortionFile
                f ctx.OutputDirectory suffix opts.Overwrite
                refData.DetectionParams refData.Stars refData.Triangles
                refData.Metadata.Width refData.Metadata.Height
                opts refData.Masters)

// --- Main Pipeline ---

/// The main pipeline: pure railway-oriented composition
let private runPipeline (opts: AlignOptions) : AsyncResult<ProcessingResult, AlignError> =
    asyncResult {
        // 1. Validate everything upfront (fail fast)
        let! files = resolveInputFiles opts.Input
                     |> Result.mapError InputValidation
                     |> AsyncResult.ofResult
        printfn $"Found {files.Length} files"
        printfn ""

        // 2. Analyze reference image (loads calibration masters internally)
        let refFile = resolveReferenceFile files opts
        let! refData = analyzeReference refFile opts

        // 3. Build processing context
        let ctx = {
            Opts = opts
            Files = files
            OutputDirectory = opts.Output
            Reference = refData
        }

        // 4. Execute selected mode
        return! executeMode ctx
    }

/// Entry point: handle help and convert Result to exit code
let run (args: string array) =
    if args |> Array.exists (fun a -> a = "--help" || a = "-h") then
        showHelp()
        0
    else
        parseArgs args
        |> runPipeline
        |> Async.map (function
            | Ok result ->
                // Success - exit cleanly (failed files already reported)
                if result.FailedCount > 0 then 1 else 0
            | Error err ->
                // Log structured error and return exit code
                Log.Error(err.ToString())
                printfn ""
                printfn "Run 'xisfprep align --help' for usage information"
                1)
        |> Async.RunSynchronously
