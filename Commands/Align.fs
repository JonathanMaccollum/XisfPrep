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
// ---

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
let processDetectFile (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool) (outputFormat: XisfSampleFormat option) (detectionParams: DetectionParams) (intensity: float) (masters: (MasterFrames * CalibrationConfig) option) =
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

            // Get pixel data, calibrate, and detect stars
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let channel0 =
                if channels = 1 then pixelFloats
                else Array.init (width * height) (fun i -> pixelFloats.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 width height mad 0 "Luminance" detectionParams
            let stars = starsResult.Stars

            // Create output
            let format = outputFormat |> Option.defaultValue img.SampleFormat
            let pixels = createBlackPixels width height channels format
            paintDetectedStars pixels width height channels format stars intensity

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

            // Get pixel data, calibrate, and detect stars
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let channel0 =
                if channels = 1 then pixelFloats
                else Array.init (width * height) (fun i -> pixelFloats.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 width height mad 0 "Luminance" detectionParams
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
                    // Center-seeded expanding match
                    let config = {
                        defaultExpandingConfig with
                            AnchorStars = opts.AnchorStars
                            AnchorDistribution = opts.AnchorDistribution
                            RatioTolerance = opts.RatioTolerance
                            MinAnchorVotes = opts.MinVotes
                    }

                    let result = matchExpanding refStars targetStars width height config

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

            // Create output
            let format = outputFormat |> Option.defaultValue img.SampleFormat
            let pixels = createBlackPixels width height channels format
            paintMatchVisualization pixels width height channels format targetStars refStars matchedTargetIndices opts.Intensity

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

            // Verify dimensions match reference
            if width <> imageWidth || height <> imageHeight then
                Log.Error($"Image dimensions {width}x{height} don't match reference {imageWidth}x{imageHeight}: {fileName}")
                return false
            else

            // Get pixel data, calibrate, and detect stars
            let pixelFloatsRaw = PixelIO.readPixelsAsFloat img
            let pixelFloats = applyCalibration pixelFloatsRaw masters
            let channel0 =
                if channels = 1 then pixelFloats
                else Array.init (width * height) (fun i -> pixelFloats.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 width height mad 0 "Luminance" detectionParams
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
                    // Use expanding match algorithm
                    let config = {
                        defaultExpandingConfig with
                            AnchorStars = opts.AnchorStars
                            AnchorDistribution = opts.AnchorDistribution
                            RatioTolerance = opts.RatioTolerance
                            MinAnchorVotes = opts.MinVotes
                    }

                    let result = matchExpanding refStars targetStars width height config

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

                        let rbfResult = RBFTransform.setupRBF refStars targetStars matchedPairs transform rbfConfig width height

                        match rbfResult.Coefficients with
                        | Some coeffs ->
                            let distName = match opts.Distortion with Wendland -> "wend" | TPS -> "tps" | IMQ -> "imq" | _ -> ""
                            (Some coeffs, sprintf " [%s rms=%.2f]" distName rbfResult.ResidualRMS)
                        | None ->
                            Log.Warning($"RBF setup failed for {fileName}, using similarity only")
                            (None, " [rbf failed]")

                // Apply transformation
                let transformedPixels = transformImage pixelFloats width height channels transform rbfCoeffs opts.Interpolation

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
                    let detPixels = createBlackPixels width height channels XisfSampleFormat.UInt16
                    paintDetectedStars detPixels width height channels XisfSampleFormat.UInt16 targetStars opts.Intensity
                    let detHeaders = createDetectHeaders targetStars.Length detectionParams.Threshold detectionParams.GridSize
                    let detInPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
                    let detOutputImage = createOutputImage img detPixels XisfSampleFormat.UInt16 detHeaders detInPlaceKeys
                    do! writeOutputFile detPath detOutputImage "XisfPrep Align v1.0"

                if opts.IncludeDistortionModel && rbfCoeffs.IsSome then
                    let distFileName = $"{baseName}_dist.xisf"
                    let distPath = Path.Combine(outputDir, distFileName)

                    // Generate distortion heatmap
                    let distPixels = Array.zeroCreate<byte> (width * height * 3 * 2) // RGB * UInt16
                    let maxMag = 10.0

                    Array.Parallel.iter (fun pixIdx ->
                        let ox = pixIdx % width
                        let oy = pixIdx / width
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
                    ) [| 0 .. width * height - 1 |]

                    // Overlay control points
                    match rbfCoeffs with
                    | Some coeffs ->
                        for (cx, cy) in coeffs.ControlPoints do
                            for ch in 0 .. 2 do
                                paintCircle distPixels width height 3 ch cx cy 3.0 1.0 XisfSampleFormat.UInt16
                    | None -> ()

                    // Create RGB output image
                    let distGeometry = XisfImageGeometry([| uint32 width; uint32 height |], 3u)
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

            if width <> imageWidth || height <> imageHeight then
                Log.Error($"Image dimensions don't match reference: {fileName}")
                return false
            else

            // Detect stars
            let pixelFloats = PixelIO.readPixelsAsFloat img
            let channels = int img.Geometry.ChannelCount
            let channel0 =
                if channels = 1 then pixelFloats
                else Array.init (width * height) (fun i -> pixelFloats.[i * channels])

            let mad = calculateChannelMAD channel0
            let starsResult = StarDetection.detectStarsInChannel channel0 width height mad 0 "Luminance" detectionParams
            let targetStars = starsResult.Stars

            // Run matching
            let config = {
                defaultExpandingConfig with
                    AnchorStars = opts.AnchorStars
                    AnchorDistribution = opts.AnchorDistribution
                    RatioTolerance = opts.RatioTolerance
                    MinAnchorVotes = opts.MinVotes
            }

            let result = matchExpanding refStars targetStars width height config

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

                let rbfResult = RBFTransform.setupRBF refStars targetStars (matchedPairs.ToArray()) transform rbfConfig width height

                // Print console heatmap for analysis if requested
                if opts.ShowDistortionStats then
                    printDistortionHeatmap width height transform rbfResult.Coefficients 10

                // Create RGB output for heatmap
                let outputFormat = XisfSampleFormat.UInt16
                let pixelCount = width * height
                let pixels = Array.zeroCreate<byte> (pixelCount * 3 * 2) // RGB * UInt16

                // Sample distortion at each pixel and create heatmap
                let maxMag = 10.0 // Max distortion for color scale

                Array.Parallel.iter (fun pixIdx ->
                    let ox = pixIdx % width
                    let oy = pixIdx / width

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
                            paintCircle pixels width height 3 ch cx cy 3.0 1.0 outputFormat
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

                // Create RGB output image
                let geometry = XisfImageGeometry([| uint32 width; uint32 height |], 3u)
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
    printfn "Examples:"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\""
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"validation/\" --output-mode detect"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"validation/\" --output-mode match --intensity 0.8"

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
    }

    let opts = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"

    if opts.Reference.IsSome && opts.AutoReference then
        failwith "--reference and --auto-reference are mutually exclusive"

    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"
    if opts.Threshold <= 0.0 then failwith "Threshold must be positive"
    if opts.GridSize < 16 then failwith "Grid size must be at least 16"
    if opts.MinFWHM <= 0.0 then failwith "Min FWHM must be positive"
    if opts.MaxFWHM <= opts.MinFWHM then failwith "Max FWHM must be greater than min FWHM"
    if opts.MaxEccentricity < 0.0 || opts.MaxEccentricity > 1.0 then failwith "Max eccentricity must be between 0 and 1"
    if opts.MaxStars < 1 then failwith "Max stars must be at least 1"
    if opts.RatioTolerance <= 0.0 then failwith "Ratio tolerance must be positive"
    if opts.MaxStarsTriangles < 3 then failwith "Max stars for triangles must be at least 3"
    if opts.MinVotes < 1 then failwith "Min votes must be at least 1"
    if opts.Intensity < 0.0 || opts.Intensity > 1.0 then failwith "Intensity must be between 0 and 1"
    if opts.AnchorStars < 4 then failwith "Anchor stars must be at least 4"
    if opts.RBFSupportFactor < 1.0 then failwith "RBF support factor must be at least 1.0"
    if opts.RBFRegularization < 0.0 then failwith "RBF regularization must be non-negative"

    // Calibration validation (same as Commands.Calibrate)
    if opts.BiasFrame.IsSome && opts.BiasLevel.IsSome then
        failwith "--bias and --bias-level are mutually exclusive"
    if opts.UncalibratedDark && opts.BiasFrame.IsNone && opts.BiasLevel.IsNone then
        failwith "--uncalibrated-dark requires --bias or --bias-level"
    if opts.UncalibratedFlat && opts.BiasFrame.IsNone && opts.BiasLevel.IsNone then
        failwith "--uncalibrated-flat requires --bias or --bias-level"
    if opts.UncalibratedFlat && opts.DarkFrame.IsNone then
        failwith "--uncalibrated-flat requires --dark"
    if opts.OptimizeDark then
        if opts.DarkFrame.IsNone then
            failwith "--optimize-dark requires --dark"
        if not opts.UncalibratedDark then
            failwith "--optimize-dark requires --uncalibrated-dark"
        if opts.BiasFrame.IsNone && opts.BiasLevel.IsNone then
            failwith "--optimize-dark requires --bias or --bias-level"
    if opts.OutputPedestal < 0 || opts.OutputPedestal > 65535 then
        failwith "Pedestal must be in range [0, 65535]"

    opts

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let opts = parseArgs args

                // Find input files
                let inputDir = Path.GetDirectoryName(opts.Input)
                let pattern = Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else

                let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                if files.Length = 0 then
                    Log.Error($"No files found matching pattern: {opts.Input}")
                    return 1
                else

                printfn $"Found {files.Length} files"

                let stopwatch = Stopwatch.StartNew()

                // Determine reference
                let referenceFile =
                    match opts.Reference with
                    | Some r -> r
                    | None ->
                        if opts.AutoReference then
                            printfn "Auto-reference selection not yet implemented, using first file"
                        files.[0]

                printfn $"Reference: {Path.GetFileName referenceFile}"
                printfn ""

                // Load reference image
                let reader = new XisfReader()
                let! refUnit = reader.ReadAsync(referenceFile) |> Async.AwaitTask
                let refImage = refUnit.Images.[0]

                let width = int refImage.Geometry.Width
                let height = int refImage.Geometry.Height
                let channels = int refImage.Geometry.ChannelCount

                printfn $"Image dimensions: {width}x{height}, {channels} channel(s)"

                // Load calibration masters if configured
                let! masters =
                    if opts.BiasFrame.IsSome || opts.DarkFrame.IsSome || opts.FlatFrame.IsSome then
                        async {
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

                            // Log what was loaded
                            match calConfig.BiasFrame with
                            | Some path -> Log.Information("Master bias: {Path}", Path.GetFileName path)
                            | None -> ()
                            match calConfig.DarkFrame with
                            | Some path -> Log.Information("Master dark: {Path}", Path.GetFileName path)
                            | None -> ()
                            match calConfig.FlatFrame with
                            | Some path -> Log.Information("Master flat: {Path}", Path.GetFileName path)
                            | None -> ()

                            return Some (m, calConfig)
                        }
                    else
                        async { return None }

                // Get reference pixel data and calibrate
                let refPixelsRaw = PixelIO.readPixelsAsFloat refImage
                let refPixels = applyCalibration refPixelsRaw masters

                // Use first channel for star detection
                let refChannel0 =
                    if channels = 1 then refPixels
                    else Array.init (width * height) (fun i -> refPixels.[i * channels])

                let refMAD = calculateChannelMAD refChannel0

                printfn $"Detecting stars in reference (MAD: {refMAD:F2})..."

                // Build detection params from user options
                let detectionParams: DetectionParams = {
                    Threshold = opts.Threshold
                    GridSize = opts.GridSize
                    MinFWHM = opts.MinFWHM
                    MaxFWHM = opts.MaxFWHM
                    MaxEccentricity = opts.MaxEccentricity
                    MaxStars = Some opts.MaxStars
                }

                let refStarsResult =
                    StarDetection.detectStarsInChannel
                        refChannel0 width height refMAD 0 "Luminance" detectionParams

                let refStars = refStarsResult.Stars
                printfn $"Reference stars detected: {refStars.Length}"

                if refStars.Length < 10 then
                    Log.Error("Insufficient stars in reference image (need at least 10)")
                    return 1
                else

                // Form triangles from reference stars
                let refTriangles = formTriangles refStars opts.MaxStarsTriangles

                printfn $"Reference triangles formed: {refTriangles.Length}"
                printfn ""

                match opts.OutputMode with
                | Detect ->
                    // Detect mode: show detected stars
                    printfn "Output mode: detect (star visualization)"
                    printfn ""

                    // Create output directory if needed
                    if not (Directory.Exists(opts.Output)) then
                        Directory.CreateDirectory(opts.Output) |> ignore
                        Log.Information($"Created output directory: {opts.Output}")

                    // Use _det suffix for detect mode
                    let suffix = if opts.Suffix = defaultSuffix then "_det" else opts.Suffix

                    // Process all files in parallel
                    let tasks = files |> Array.map (fun f ->
                        processDetectFile f opts.Output suffix opts.Overwrite opts.OutputFormat detectionParams opts.Intensity masters)
                    let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                    let successCount = results |> Array.filter id |> Array.length
                    let failCount = results.Length - successCount

                    stopwatch.Stop()
                    let elapsed = stopwatch.Elapsed.TotalSeconds
                    printfn ""
                    printfn $"Processed {files.Length} images, wrote {successCount}, {failCount} failed in {elapsed:F1}s"

                    return if failCount > 0 then 1 else 0

                | Match ->
                    // Match mode: show star correspondences
                    printfn "Output mode: match (correspondence visualization)"
                    printfn ""

                    // Create output directory if needed
                    if not (Directory.Exists(opts.Output)) then
                        Directory.CreateDirectory(opts.Output) |> ignore
                        Log.Information($"Created output directory: {opts.Output}")

                    // Use _mat suffix for match mode
                    let suffix = if opts.Suffix = defaultSuffix then "_mat" else opts.Suffix

                    // Process all files in parallel
                    let tasks = files |> Array.map (fun f ->
                        processMatchFile f opts.Output suffix opts.Overwrite opts.OutputFormat detectionParams refStars refTriangles opts masters)
                    let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                    let successCount = results |> Array.filter id |> Array.length
                    let failCount = results.Length - successCount

                    stopwatch.Stop()
                    let elapsed = stopwatch.Elapsed.TotalSeconds
                    printfn ""
                    printfn $"Processed {files.Length} images, wrote {successCount}, {failCount} failed in {elapsed:F1}s"

                    return if failCount > 0 then 1 else 0

                | Align ->
                    // Align mode: apply transformation
                    printfn "Output mode: align (image transformation)"
                    printfn ""

                    // Create output directory if needed
                    if not (Directory.Exists(opts.Output)) then
                        Directory.CreateDirectory(opts.Output) |> ignore
                        Log.Information($"Created output directory: {opts.Output}")

                    let refFileName = Path.GetFileName(referenceFile)

                    // Process all files in parallel
                    let tasks = files |> Array.map (fun f ->
                        processAlignFile f opts.Output opts.Suffix opts.Overwrite detectionParams refStars refTriangles refFileName width height opts masters)
                    let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                    let successCount = results |> Array.filter id |> Array.length
                    let failCount = results.Length - successCount

                    stopwatch.Stop()
                    let elapsed = stopwatch.Elapsed.TotalSeconds
                    printfn ""
                    printfn $"Aligned {successCount} images, {failCount} failed in {elapsed:F1}s"

                    return if failCount > 0 then 1 else 0

                | Distortion ->
                    // Distortion mode: visualize distortion field
                    printfn "Output mode: distortion (field visualization)"
                    printfn ""

                    // Create output directory if needed
                    if not (Directory.Exists(opts.Output)) then
                        Directory.CreateDirectory(opts.Output) |> ignore
                        Log.Information($"Created output directory: {opts.Output}")

                    // Use _dist suffix for distortion mode
                    let suffix = if opts.Suffix = defaultSuffix then "_dist" else opts.Suffix

                    // Process all files in parallel
                    let tasks = files |> Array.map (fun f ->
                        processDistortionFile f opts.Output suffix opts.Overwrite detectionParams refStars refTriangles width height opts masters)
                    let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                    let successCount = results |> Array.filter id |> Array.length
                    let failCount = results.Length - successCount

                    stopwatch.Stop()
                    let elapsed = stopwatch.Elapsed.TotalSeconds
                    printfn ""
                    printfn $"Processed {files.Length} images, wrote {successCount}, {failCount} failed in {elapsed:F1}s"

                    return if failCount > 0 then 1 else 0

            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep align --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
