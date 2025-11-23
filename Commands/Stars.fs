module Commands.Stars

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms
open Algorithms.Statistics
open Algorithms.Painting
open Algorithms.OutputImage

type MarkerType =
    | Circle
    | Crosshair
    | Gaussian

type ScaleBy =
    | ByFWHM
    | ByFlux
    | Fixed

// --- Defaults ---
let private defaultThreshold = 5.0
let private defaultGridSize = 128
let private defaultMinFWHM = 1.5
let private defaultMaxFWHM = 20.0
let private defaultMaxEccentricity = 0.5
let private defaultMaxStars = 20000
let private defaultMarker = Circle
let private defaultScaleBy = ByFWHM
let private defaultIntensity = 1.0
let private defaultSuffix = "_stars"
let private defaultParallel = System.Environment.ProcessorCount
// ---

// --- Star Detection Functions ---

/// Build detection params from options
let buildDetectionParams (threshold: float) (gridSize: int) (minFwhm: float) (maxFwhm: float) (maxEcc: float) (maxStars: int) : StarDetection.DetectionParams =
    { Threshold = threshold
      GridSize = gridSize
      MinFWHM = minFwhm
      MaxFWHM = maxFwhm
      MaxEccentricity = maxEcc
      MaxStars = Some maxStars }

/// Calculate median of array
let private median (arr: float[]) =
    if arr.Length = 0 then 0.0
    else
        let sorted = Array.sort arr
        let mid = sorted.Length / 2
        if sorted.Length % 2 = 0 then (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else sorted.[mid]

/// Build channel data for star detection
let buildChannelData (pixelFloats: float[]) (width: int) (height: int) (channels: int) =
    let pixelCount = width * height
    Array.init channels (fun ch ->
        let values = Array.init pixelCount (fun pix -> pixelFloats.[pix * channels + ch])
        let med = median values
        let mad = calculateMAD values med
        let channelName =
            match channels, ch with
            | 1, _ -> "Mono"
            | 3, 0 -> "Red"
            | 3, 1 -> "Green"
            | 3, _ -> "Blue"
            | _, n -> $"Channel{n}"
        (values, mad, ch, channelName))

/// Compute aggregate statistics from detection results
let computeStarStatistics (results: StarDetection.StarDetectionResults) =
    let allStars = results.Channels |> Array.collect (fun r -> r.Stars)
    let count = allStars.Length
    if count = 0 then (0, None, None, None, None)
    else
        let fwhms = allStars |> Array.map (fun s -> s.FWHM)
        let hfrs = allStars |> Array.map (fun s -> s.HFR)
        let eccs = allStars |> Array.map (fun s -> s.Eccentricity)
        let snrs = allStars |> Array.map (fun s -> s.SNR)
        (count, Some (median fwhms), Some (median hfrs), Some (median eccs), Some (median snrs))

/// Convert marker type to string
let private markerToString = function
    | Circle -> "CIRCLE"
    | Crosshair -> "CROSSHAIR"
    | Gaussian -> "GAUSSIAN"

/// Convert scale-by to string
let private scaleByToString = function
    | ByFWHM -> "FWHM"
    | ByFlux -> "FLUX"
    | Fixed -> "FIXED"

/// Create FITS headers for star statistics and visualization parameters
let createStarHeaders
    (count: int)
    (medFwhm: float option)
    (medHfr: float option)
    (medEcc: float option)
    (medSnr: float option)
    (threshold: float)
    (gridSize: int)
    (minFwhm: float)
    (maxFwhm: float)
    (maxEcc: float)
    (marker: MarkerType)
    (scaleBy: ScaleBy)
    (intensity: float) =
    [|
        // Core identification (override source)
        XisfFitsKeyword("IMAGETYP", "STARMAP", "Type of exposure") :> XisfCoreElement
        XisfFitsKeyword("SWCREATE", "XisfPrep Stars", "Software that created this file") :> XisfCoreElement

        // Universal star statistics
        XisfFitsKeyword("STARCOUNT", count.ToString(), "Number of detected stars") :> XisfCoreElement
        match medFwhm with
        | Some v -> XisfFitsKeyword("MEDFWHM", sprintf "%.2f" v, "Median FWHM in pixels") :> XisfCoreElement
        | None -> ()
        match medHfr with
        | Some v -> XisfFitsKeyword("MEDHFR", sprintf "%.2f" v, "Median HFR in pixels") :> XisfCoreElement
        | None -> ()
        match medEcc with
        | Some v -> XisfFitsKeyword("MEDECCEN", sprintf "%.3f" v, "Median eccentricity") :> XisfCoreElement
        | None -> ()
        match medSnr with
        | Some v -> XisfFitsKeyword("MEDSNR", sprintf "%.1f" v, "Median SNR") :> XisfCoreElement
        | None -> ()

        // Detection parameters
        XisfFitsKeyword("STARTHRES", sprintf "%.1f" threshold, "Detection threshold (sigma)") :> XisfCoreElement
        XisfFitsKeyword("STARGRID", gridSize.ToString(), "Background grid size") :> XisfCoreElement
        XisfFitsKeyword("STARMINFW", sprintf "%.1f" minFwhm, "Min FWHM filter") :> XisfCoreElement
        XisfFitsKeyword("STARMAXFW", sprintf "%.1f" maxFwhm, "Max FWHM filter") :> XisfCoreElement
        XisfFitsKeyword("STARMAXEC", sprintf "%.2f" maxEcc, "Max eccentricity filter") :> XisfCoreElement

        // Visualization parameters (tool-specific)
        XisfFitsKeyword("XPRPMRKR", markerToString marker, "Marker type") :> XisfCoreElement
        XisfFitsKeyword("XPRPSCAL", scaleByToString scaleBy, "Marker scale method") :> XisfCoreElement
        XisfFitsKeyword("XPRPINTN", sprintf "%.2f" intensity, "Marker intensity") :> XisfCoreElement

        XisfFitsKeyword("HISTORY", "", "Star detection visualization by XisfPrep Stars") :> XisfCoreElement
    |]

// --- Star Painting Functions ---

/// Calculate marker size for a star based on scale-by option
let getMarkerSize (star: StarDetection.DetectedStar) (scaleBy: ScaleBy) (allStars: StarDetection.DetectedStar[]) =
    match scaleBy with
    | ByFWHM -> star.FWHM * 2.0
    | ByFlux ->
        let maxFlux = allStars |> Array.map (fun s -> s.Flux) |> Array.max
        let minSize = 3.0
        let maxSize = 20.0
        minSize + (star.Flux / maxFlux) * (maxSize - minSize)
    | Fixed -> 8.0

/// Paint all stars on the image
let paintAllStars (pixels: byte[]) (width: int) (height: int) (channels: int) (format: XisfSampleFormat) (results: StarDetection.StarDetectionResults) (marker: MarkerType) (scaleBy: ScaleBy) (intensity: float) =
    let allStars = results.Channels |> Array.collect (fun r -> r.Stars)

    for channelResult in results.Channels do
        let ch = channelResult.Channel
        for star in channelResult.Stars do
            let size = getMarkerSize star scaleBy allStars
            match marker with
            | Circle -> Painting.paintCircle pixels width height channels ch star.X star.Y size intensity format
            | Crosshair -> Painting.paintCrosshair pixels width height channels ch star.X star.Y size intensity format
            | Gaussian -> Painting.paintGaussian pixels width height channels ch star.X star.Y star.FWHM intensity format

// ---

type StarsOptions = {
    Input: string
    Output: string option
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
    // Visualization options
    Marker: MarkerType
    ScaleBy: ScaleBy
    Intensity: float
}

let showHelp() =
    printfn "stars - Detect stars and generate visualization overlay"
    printfn ""
    printfn "Usage: xisfprep stars [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn ""
    printfn "Output:"
    printfn "  --output, -o <directory>  Output directory (default: same as input)"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Detection Parameters:"
    printfn $"  --threshold <sigma>       Detection threshold in sigma (default: {defaultThreshold})"
    printfn $"  --grid-size <px>          Background grid size in pixels (default: {defaultGridSize})"
    printfn $"  --min-fwhm <px>           Minimum FWHM filter in pixels (default: {defaultMinFWHM})"
    printfn $"  --max-fwhm <px>           Maximum FWHM filter in pixels (default: {defaultMaxFWHM})"
    printfn $"  --max-eccentricity <val>  Maximum eccentricity filter (default: {defaultMaxEccentricity})"
    printfn $"  --max-stars <n>           Maximum stars to detect (default: {defaultMaxStars})"
    printfn ""
    printfn "Visualization Options:"
    printfn "  --marker <type>           Marker type (default: circle)"
    printfn "                              circle    - Draw circles around stars"
    printfn "                              crosshair - Draw crosshair markers"
    printfn "                              gaussian  - Paint Gaussian profiles matching FWHM"
    printfn "  --scale-by <property>     Scale marker size by property (default: fwhm)"
    printfn "                              fwhm  - Scale by star FWHM"
    printfn "                              flux  - Scale by star brightness"
    printfn "                              fixed - Fixed marker size"
    printfn $"  --intensity <0-1>         Marker brightness (default: {defaultIntensity})"
    printfn ""
    printfn "Output Headers:"
    printfn "  STARCOUNT - Number of detected stars"
    printfn "  MEDFWHM   - Median FWHM in pixels"
    printfn "  MEDHFR    - Median HFR in pixels"
    printfn "  MEDECCEN  - Median eccentricity"
    printfn "  MEDSNR    - Median signal-to-noise"
    printfn "  STARTHRES - Detection threshold (sigma)"
    printfn "  STARGRID  - Background grid size"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep stars -i \"lights/*.xisf\" -o \"validation/\""
    printfn "  xisfprep stars -i \"Ha*.xisf\" -o \"validation/\" --threshold 3.0 --max-fwhm 25.0"
    printfn "  xisfprep stars -i \"image.xisf\" -o \"check/\" --marker crosshair --scale-by flux"
    printfn "  xisfprep stars -i \"focus_test.xisf\" --marker gaussian --scale-by fwhm"

let parseArgs (args: string array) : StarsOptions =
    let rec parse (args: string list) (opts: StarsOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = Some value }
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
        // Visualization options
        | "--marker" :: value :: rest ->
            let marker =
                match value.ToLower() with
                | "circle" -> Circle
                | "crosshair" -> Crosshair
                | "gaussian" -> Gaussian
                | _ -> failwithf "Unknown marker type: %s (supported: circle, crosshair, gaussian)" value
            parse rest { opts with Marker = marker }
        | "--scale-by" :: value :: rest ->
            let scaleBy =
                match value.ToLower() with
                | "fwhm" -> ByFWHM
                | "flux" -> ByFlux
                | "fixed" -> Fixed
                | _ -> failwithf "Unknown scale-by value: %s (supported: fwhm, flux, fixed)" value
            parse rest { opts with ScaleBy = scaleBy }
        | "--intensity" :: value :: rest ->
            parse rest { opts with Intensity = float value }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = None
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
        Marker = defaultMarker
        ScaleBy = defaultScaleBy
        Intensity = defaultIntensity
    }

    let opts = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if opts.Threshold <= 0.0 then failwith "Threshold must be positive"
    if opts.GridSize < 16 then failwith "Grid size must be at least 16"
    if opts.MinFWHM <= 0.0 then failwith "Min FWHM must be positive"
    if opts.MaxFWHM <= opts.MinFWHM then failwith "Max FWHM must be greater than min FWHM"
    if opts.MaxEccentricity < 0.0 || opts.MaxEccentricity > 1.0 then failwith "Max eccentricity must be between 0 and 1"
    if opts.MaxStars < 1 then failwith "Max stars must be at least 1"
    if opts.Intensity < 0.0 || opts.Intensity > 1.0 then failwith "Intensity must be between 0 and 1"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

    opts

/// Process a single image: read, detect stars, create visualization
let processImage (inputPath: string) (outputDir: string) (opts: StarsOptions) : Async<bool> =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{opts.Suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            // Skip if output exists and not overwriting
            if File.Exists(outPath) && not opts.Overwrite then
                Log.Warning($"Output file '{outFileName}' already exists, skipping (use --overwrite to replace)")
                return true
            else

            printfn $"Processing: {fileName}"

            // Read input image
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Read pixel data and build channel data for detection
            let pixelFloats = PixelIO.readPixelsAsFloat img
            let channelData = buildChannelData pixelFloats width height channels

            // Build detection parameters and run detection
            let detectionParams = buildDetectionParams opts.Threshold opts.GridSize opts.MinFWHM opts.MaxFWHM opts.MaxEccentricity opts.MaxStars
            let results = StarDetection.detectStars channelData width height detectionParams

            // Compute statistics and create headers
            let (count, medFwhm, medHfr, medEcc, medSnr) = computeStarStatistics results
            let starHeaders = createStarHeaders count medFwhm medHfr medEcc medSnr
                                opts.Threshold opts.GridSize opts.MinFWHM opts.MaxFWHM opts.MaxEccentricity
                                opts.Marker opts.ScaleBy opts.Intensity

            // Determine output format
            let outputFormat = opts.OutputFormat |> Option.defaultValue img.SampleFormat

            // Create black pixels and paint stars
            let pixels = createBlackPixels width height channels outputFormat
            paintAllStars pixels width height channels outputFormat results opts.Marker opts.ScaleBy opts.Intensity

            // Create output image with headers
            let inPlaceKeys = Set.ofList ["IMAGETYP"; "SWCREATE"]
            let outputImage = createOutputImage img pixels outputFormat starHeaders inPlaceKeys

            // Write output
            do! writeOutputFile outPath outputImage "XisfPrep Stars v1.0"

            let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
            printfn $"  {count} stars -> {outFileName} ({width}x{height}, {channels}ch, {sizeMB}MB)"

            return true
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(inputPath)}: {ex.Message}")
            return false
    }

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let opts = parseArgs args

                Log.Information($"Star detection with threshold {opts.Threshold}σ, marker: {opts.Marker}")

                // Determine output directory
                let inputDir = Path.GetDirectoryName(opts.Input)
                let pattern = Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                let outputDir =
                    match opts.Output with
                    | Some dir -> dir
                    | None -> actualDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else

                // Create output directory if needed
                if not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                    Log.Information($"Created output directory: {outputDir}")

                let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                if files.Length = 0 then
                    Log.Error($"No files found matching pattern: {opts.Input}")
                    return 1
                else

                printfn $"Found {files.Length} files to process"
                printfn $"Detection: threshold={opts.Threshold}σ, FWHM=[{opts.MinFWHM}-{opts.MaxFWHM}], maxEcc={opts.MaxEccentricity}"
                printfn $"Visualization: marker={opts.Marker}, scaleBy={opts.ScaleBy}, intensity={opts.Intensity}"
                printfn ""

                // Process all files in parallel with max parallelism limit
                let tasks = files |> Array.map (fun f -> processImage f outputDir opts)
                let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                let successCount = results |> Array.filter id |> Array.length
                let failCount = results.Length - successCount

                printfn $"Completed: {successCount} succeeded, {failCount} failed"

                return if failCount > 0 then 1 else 0
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep stars --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
