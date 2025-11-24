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

// Type aliases for star detection types
type DetectedStar = Algorithms.StarDetection.DetectedStar
type DetectionParams = Algorithms.StarDetection.DetectionParams

// Algorithm selection
type MatchAlgorithm =
    | Triangle   // Original triangle matching (default)
    | Expanding  // Center-seeded expanding match

type InterpolationMethod =
    | Nearest
    | Bilinear
    | Bicubic

type OutputMode =
    | Detect  // Show detected stars only
    | Match   // Show matched star correspondences
    | Align   // Apply transformation (default)

// --- Defaults ---
let private defaultMaxShift = 100
let private defaultInterpolation = Bicubic
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
// ---

type AlignOptions = {
    Input: string
    Output: string
    Reference: string option
    AutoReference: bool
    OutputMode: OutputMode
    MaxShift: int
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
}

// --- Helper Functions ---

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
let processDetectFile (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool) (outputFormat: XisfSampleFormat option) (detectionParams: DetectionParams) (intensity: float) =
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

            // Get pixel data and detect stars
            let pixelFloats = PixelIO.readPixelsAsFloat img
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
    (opts: AlignOptions) =
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

            // Get pixel data and detect stars
            let pixelFloats = PixelIO.readPixelsAsFloat img
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
    printfn "                              detect - Show detected stars only"
    printfn "                              match  - Show matched star correspondences"
    printfn "                              align  - Apply transformation (default)"
    printfn ""
    printfn "Alignment Parameters:"
    printfn $"  --max-shift <pixels>      Maximum pixel shift allowed (default: {defaultMaxShift})"
    printfn "  --interpolation <method>  Resampling method (default: bicubic)"
    printfn "                              nearest  - Nearest neighbor (preserves values)"
    printfn "                              bilinear - Bilinear (smooth, fast)"
    printfn "                              bicubic  - Bicubic (smooth, higher quality)"
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
    printfn "Examples:"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\" --max-shift 50"
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
                       | _ -> failwithf "Unknown output mode: %s (supported: detect, match, align)" value
            parse rest { opts with OutputMode = mode }
        | "--max-shift" :: value :: rest ->
            parse rest { opts with MaxShift = int value }
        | "--interpolation" :: value :: rest ->
            let interp = match value.ToLower() with
                         | "nearest" -> Nearest
                         | "bilinear" -> Bilinear
                         | "bicubic" -> Bicubic
                         | _ -> failwithf "Unknown interpolation method: %s (supported: nearest, bilinear, bicubic)" value
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
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Reference = None
        AutoReference = false
        OutputMode = Align
        MaxShift = defaultMaxShift
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
    }

    let opts = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"

    if opts.Reference.IsSome && opts.AutoReference then
        failwith "--reference and --auto-reference are mutually exclusive"

    if opts.MaxShift < 1 then failwith "Max shift must be at least 1 pixel"
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

                // Get reference pixel data and detect stars
                let refPixels = PixelIO.readPixelsAsFloat refImage

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
                        processDetectFile f opts.Output suffix opts.Overwrite opts.OutputFormat detectionParams opts.Intensity)
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
                        processMatchFile f opts.Output suffix opts.Overwrite opts.OutputFormat detectionParams refStars refTriangles opts)
                    let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                    let successCount = results |> Array.filter id |> Array.length
                    let failCount = results.Length - successCount

                    stopwatch.Stop()
                    let elapsed = stopwatch.Elapsed.TotalSeconds
                    printfn ""
                    printfn $"Processed {files.Length} images, wrote {successCount}, {failCount} failed in {elapsed:F1}s"

                    return if failCount > 0 then 1 else 0

                | Align ->
                    // Align mode: apply transformation (not yet implemented)
                    Log.Error("Align output mode not yet implemented. Use --output-mode match for triangle matching analysis.")
                    return 1

            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep align --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
