module Commands.Align

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms

// Type aliases for star detection types
type DetectedStar = Algorithms.StarDetection.DetectedStar
type DetectionParams = Algorithms.StarDetection.DetectionParams

type InterpolationMethod =
    | Nearest
    | Bilinear
    | Bicubic

// --- Defaults ---
let private defaultMaxShift = 100
let private defaultInterpolation = Bicubic
let private defaultSuffix = "_a"
let private defaultParallel = Environment.ProcessorCount
// ---

type AlignOptions = {
    Input: string
    Output: string
    Reference: string option
    AutoReference: bool
    MaxShift: int
    Interpolation: InterpolationMethod
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
    Diagnostic: bool
}

// --- Triangle Matching Types ---

type Triangle = {
    /// Indices into the star array
    I1: int
    I2: int
    I3: int
    /// Side lengths sorted ascending (scale-invariant descriptor)
    SideA: float  // shortest
    SideB: float  // middle
    SideC: float  // longest
    /// Ratios for matching (rotation and scale invariant)
    RatioAB: float  // SideA / SideB
    RatioBC: float  // SideB / SideC
}

type TriangleMatch = {
    RefTriangle: Triangle
    TargetTriangle: Triangle
    RatioError: float  // Combined error in ratios
}

type AlignmentDiagnostics = {
    FileName: string
    DetectedStars: int
    TrianglesFormed: int
    TrianglesMatched: int
    MatchPercentage: float
    TopVoteCount: int
    EstimatedTransform: (float * float * float * float) option  // dx, dy, rotation, scale
}

// --- Triangle Matching Algorithm ---

/// Form triangles from the brightest N stars
/// Returns triangles with scale/rotation invariant descriptors
let formTriangles (stars: DetectedStar[]) (maxStars: int) : Triangle[] =
    // Use only brightest stars for efficiency
    let topStars =
        stars
        |> Array.sortByDescending (fun s -> s.Flux)
        |> Array.truncate maxStars

    let n = topStars.Length
    if n < 3 then [||]
    else
        let triangles = ResizeArray<Triangle>()

        // Form all possible triangles (n choose 3)
        for i in 0 .. n - 3 do
            for j in i + 1 .. n - 2 do
                for k in j + 1 .. n - 1 do
                    let s1 = topStars.[i]
                    let s2 = topStars.[j]
                    let s3 = topStars.[k]

                    // Calculate side lengths
                    let d12 = sqrt ((s1.X - s2.X) ** 2.0 + (s1.Y - s2.Y) ** 2.0)
                    let d23 = sqrt ((s2.X - s3.X) ** 2.0 + (s2.Y - s3.Y) ** 2.0)
                    let d31 = sqrt ((s3.X - s1.X) ** 2.0 + (s3.Y - s1.Y) ** 2.0)

                    // Sort sides ascending for rotation invariance
                    let sides = [| d12; d23; d31 |] |> Array.sort
                    let sideA, sideB, sideC = sides.[0], sides.[1], sides.[2]

                    // Skip degenerate triangles (collinear or too small)
                    if sideA > 5.0 && sideC > 0.0 then
                        let ratioAB = sideA / sideB
                        let ratioBC = sideB / sideC

                        triangles.Add {
                            I1 = i
                            I2 = j
                            I3 = k
                            SideA = sideA
                            SideB = sideB
                            SideC = sideC
                            RatioAB = ratioAB
                            RatioBC = ratioBC
                        }

        triangles.ToArray()

/// Match triangles between reference and target by ratio similarity
let matchTriangles (refTriangles: Triangle[]) (targetTriangles: Triangle[]) (tolerance: float) : TriangleMatch[] =
    let matches = ResizeArray<TriangleMatch>()

    for refTri in refTriangles do
        for targetTri in targetTriangles do
            // Compare ratios (scale and rotation invariant)
            let errorAB = abs (refTri.RatioAB - targetTri.RatioAB)
            let errorBC = abs (refTri.RatioBC - targetTri.RatioBC)
            let totalError = errorAB + errorBC

            if totalError < tolerance then
                matches.Add {
                    RefTriangle = refTri
                    TargetTriangle = targetTri
                    RatioError = totalError
                }

    matches.ToArray()

/// Vote for star correspondences from triangle matches
/// Returns map of (refStarIdx, targetStarIdx) -> vote count
let voteForCorrespondences (matches: TriangleMatch[]) : Map<(int * int), int> =
    let votes = System.Collections.Generic.Dictionary<(int * int), int>()

    for m in matches do
        // Each triangle match votes for 3 star correspondences
        // We need to figure out which vertices correspond
        // Since sides are sorted, we use the vertex opposite each side

        let refIndices = [| m.RefTriangle.I1; m.RefTriangle.I2; m.RefTriangle.I3 |]
        let targetIndices = [| m.TargetTriangle.I1; m.TargetTriangle.I2; m.TargetTriangle.I3 |]

        // Simple approach: try all 6 possible mappings and use the one consistent with ratios
        // For now, use direct correspondence (this works when triangles are formed consistently)
        for i in 0 .. 2 do
            let pair = (refIndices.[i], targetIndices.[i])
            if votes.ContainsKey pair then
                votes.[pair] <- votes.[pair] + 1
            else
                votes.[pair] <- 1

    votes |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq

/// Estimate similarity transform from matched star pairs using least squares
/// Returns (dx, dy, rotation in degrees, scale)
let estimateTransform
    (refStars: DetectedStar[])
    (targetStars: DetectedStar[])
    (correspondences: ((int * int) * int)[])
    (minVotes: int)
    : (float * float * float * float) option =

    // Filter by vote count and collect point pairs
    let pairs =
        correspondences
        |> Array.filter (fun (_, votes) -> votes >= minVotes)
        |> Array.map (fun ((refIdx, targetIdx), _) ->
            let r = refStars.[refIdx]
            let t = targetStars.[targetIdx]
            (r.X, r.Y, t.X, t.Y))

    if pairs.Length < 2 then None
    else
        // Solve for similarity transform:
        // x' = a*x - b*y + dx
        // y' = b*x + a*y + dy
        // Where scale = sqrt(a² + b²), rotation = atan2(b, a)

        // Using least squares: minimize sum of squared errors
        let n = float pairs.Length

        let sumX = pairs |> Array.sumBy (fun (_, _, tx, _) -> tx)
        let sumY = pairs |> Array.sumBy (fun (_, _, _, ty) -> ty)
        let sumXp = pairs |> Array.sumBy (fun (rx, _, _, _) -> rx)
        let sumYp = pairs |> Array.sumBy (fun (_, ry, _, _) -> ry)

        let sumXXp = pairs |> Array.sumBy (fun (rx, _, tx, _) -> tx * rx)
        let sumXYp = pairs |> Array.sumBy (fun (_, ry, tx, _) -> tx * ry)
        let sumYXp = pairs |> Array.sumBy (fun (rx, _, _, ty) -> ty * rx)
        let sumYYp = pairs |> Array.sumBy (fun (_, ry, _, ty) -> ty * ry)

        let sumXX = pairs |> Array.sumBy (fun (_, _, tx, _) -> tx * tx)
        let sumYY = pairs |> Array.sumBy (fun (_, _, _, ty) -> ty * ty)
        let sumXY = pairs |> Array.sumBy (fun (_, _, tx, ty) -> tx * ty)

        // Solve the normal equations
        let denom = n * (sumXX + sumYY) - sumX * sumX - sumY * sumY

        if abs denom < 1e-10 then None
        else
            let a = (n * (sumXXp + sumYYp) - sumX * sumXp - sumY * sumYp) / denom
            let b = (n * (sumYXp - sumXYp) - sumY * sumXp + sumX * sumYp) / denom

            let dx = (sumXp - a * sumX + b * sumY) / n
            let dy = (sumYp - b * sumX - a * sumY) / n

            let scale = sqrt (a * a + b * b)
            let rotation = atan2 b a * 180.0 / Math.PI

            Some (dx, dy, rotation, scale)

/// Run diagnostic analysis on a target image against reference
let runDiagnostic
    (refStars: DetectedStar[])
    (refTriangles: Triangle[])
    (targetStars: DetectedStar[])
    (fileName: string)
    (ratioTolerance: float)
    (maxStarsForTriangles: int)
    : AlignmentDiagnostics =

    let targetTriangles = formTriangles targetStars maxStarsForTriangles
    let matches = matchTriangles refTriangles targetTriangles ratioTolerance

    let correspondences = voteForCorrespondences matches
    let sortedVotes =
        correspondences
        |> Map.toArray
        |> Array.sortByDescending snd

    let topVoteCount =
        if sortedVotes.Length > 0 then snd sortedVotes.[0]
        else 0

    let transform = estimateTransform refStars targetStars sortedVotes 3

    let matchPercentage =
        if refTriangles.Length > 0 then
            (float matches.Length / float refTriangles.Length) * 100.0
        else 0.0

    {
        FileName = fileName
        DetectedStars = targetStars.Length
        TrianglesFormed = targetTriangles.Length
        TrianglesMatched = matches.Length
        MatchPercentage = matchPercentage
        TopVoteCount = topVoteCount
        EstimatedTransform = transform
    }

/// Calculate MAD for star detection threshold
let calculateMAD (values: float[]) =
    if values.Length = 0 then 0.0
    else
        let sorted = Array.copy values
        Array.sortInPlace sorted
        let mid = sorted.Length / 2
        let median =
            if sorted.Length % 2 = 0 then (sorted.[mid - 1] + sorted.[mid]) / 2.0
            else sorted.[mid]

        let deviations = values |> Array.map (fun v -> abs (v - median))
        Array.sortInPlace deviations
        let madMid = deviations.Length / 2
        if deviations.Length % 2 = 0 then
            (deviations.[madMid - 1] + deviations.[madMid]) / 2.0
        else
            deviations.[madMid]

let showHelp() =
    printfn "align - Register images to reference frame"
    printfn ""
    printfn "Usage: xisfprep align [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for aligned files"
    printfn ""
    printfn "Optional:"
    printfn "  --reference, -r <file>    Reference frame to align to (first file if omitted)"
    printfn "  --auto-reference          Auto-select best reference (highest star count/SNR)"
    printfn $"  --max-shift <pixels>      Maximum pixel shift allowed (default: {defaultMaxShift})"
    printfn "  --interpolation <method>  Resampling method (default: bicubic)"
    printfn "                              nearest  - Nearest neighbor (preserves values)"
    printfn "                              bilinear - Bilinear (smooth, fast)"
    printfn "                              bicubic  - Bicubic (smooth, higher quality)"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn "  --diagnostic              Run triangle matching diagnostics only (no output files)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\" --max-shift 50"

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
        | "--diagnostic" :: rest ->
            parse rest { opts with Diagnostic = true }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Reference = None
        AutoReference = false
        MaxShift = defaultMaxShift
        Interpolation = defaultInterpolation
        Suffix = defaultSuffix
        Overwrite = false
        MaxParallel = defaultParallel
        OutputFormat = None
        Diagnostic = false
    }

    let opts = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"

    if opts.Reference.IsSome && opts.AutoReference then
        failwith "--reference and --auto-reference are mutually exclusive"

    if opts.MaxShift < 1 then failwith "Max shift must be at least 1 pixel"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

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

                let refMAD = calculateMAD refChannel0

                printfn $"Detecting stars in reference (MAD: {refMAD:F2})..."

                let detectionParams = StarDetection.defaultParams
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
                let maxStarsForTriangles = 100  // Use top 100 stars
                let refTriangles = formTriangles refStars maxStarsForTriangles

                printfn $"Reference triangles formed: {refTriangles.Length}"
                printfn ""

                if opts.Diagnostic then
                    // Diagnostic mode: analyze all files and report statistics
                    printfn "Running triangle matching diagnostics..."
                    printfn ""
                    printfn "%-40s %6s %8s %8s %7s %6s %s"
                        "File" "Stars" "Triangles" "Matched" "Match%" "Votes" "Transform"
                    printfn "%s" (String.replicate 110 "-")

                    let ratioTolerance = 0.05  // 5% tolerance on ratios

                    for file in files do
                        let fileName = Path.GetFileName file

                        // Load target image
                        let! targetUnit = reader.ReadAsync(file) |> Async.AwaitTask
                        let targetImage = targetUnit.Images.[0]

                        // Get pixel data
                        let targetPixels = PixelIO.readPixelsAsFloat targetImage
                        let targetChannel0 =
                            if channels = 1 then targetPixels
                            else Array.init (width * height) (fun i -> targetPixels.[i * channels])

                        let targetMAD = calculateMAD targetChannel0

                        // Detect stars
                        let targetStarsResult =
                            StarDetection.detectStarsInChannel
                                targetChannel0 width height targetMAD 0 "Luminance" detectionParams

                        let targetStars = targetStarsResult.Stars

                        // Run diagnostic
                        let diag = runDiagnostic refStars refTriangles targetStars fileName ratioTolerance maxStarsForTriangles

                        // Format transform
                        let transformStr =
                            match diag.EstimatedTransform with
                            | Some (dx, dy, rot, scale) ->
                                sprintf "dx=%.1f dy=%.1f rot=%.2f° s=%.4f" dx dy rot scale
                            | None -> "N/A"

                        printfn "%-40s %6d %8d %8d %6.1f%% %6d %s"
                            (if fileName.Length > 40 then fileName.Substring(0, 37) + "..." else fileName)
                            diag.DetectedStars
                            diag.TrianglesFormed
                            diag.TrianglesMatched
                            diag.MatchPercentage
                            diag.TopVoteCount
                            transformStr

                    printfn ""
                    printfn "Diagnostic complete."
                    return 0
                else
                    // Normal alignment mode (not yet implemented)
                    Log.Error("Full alignment not yet implemented. Use --diagnostic for triangle matching analysis.")
                    return 1

            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep align --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
