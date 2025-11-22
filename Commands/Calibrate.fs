module Commands.Calibrate

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Statistics

// --- Defaults ---
let private defaultPedestal = 0
let private defaultSuffix = "_cal"
let private defaultParallel = Environment.ProcessorCount
// ---

type CalibrationConfig = {
    InputPattern: string
    OutputDir: string
    BiasFrame: string option
    BiasLevel: float option
    DarkFrame: string option
    FlatFrame: string option
    UncalibratedDark: bool
    UncalibratedFlat: bool
    OptimizeDark: bool
    OutputPedestal: int
    Suffix: string
    Overwrite: bool
    DryRun: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
}

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
    formats
    |> List.maxBy formatPrecision

let showHelp() =
    printfn "calibrate - Apply calibration frames to light frames"
    printfn ""
    printfn "Usage: xisfprep calibrate [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input light frames (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for calibrated frames"
    printfn ""
    printfn "Calibration Frames:"
    printfn "  --bias, -b <file>         Master bias frame"
    printfn "  --bias-level <value>      Constant bias value (alternative to --bias)"
    printfn "  --dark, -d <file>         Master dark frame"
    printfn "  --flat, -f <file>         Master flat frame"
    printfn ""
    printfn "Master Frame State:"
    printfn "  --uncalibrated-dark       Dark is raw (not bias-subtracted)"
    printfn "  --uncalibrated-flat       Flat is raw (not bias/dark-subtracted)"
    printfn "                            Default: masters are pre-calibrated"
    printfn ""
    printfn "Dark Optimization:"
    printfn "  --optimize-dark           Optimize dark scaling for temperature/exposure differences"
    printfn "                            Requires: --uncalibrated-dark, --bias or --bias-level"
    printfn ""
    printfn "Optional:"
    printfn "  --pedestal <value>        Output pedestal [0-65535] (default: 0)"
    printfn "  --suffix <text>           Output filename suffix (default: _cal)"
    printfn "  --overwrite               Overwrite existing output files"
    printfn "  --dry-run                 Show what would be done without processing"
    printfn "  --parallel <n>            Number of parallel operations (default: CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: max precision of inputs)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Algorithm:"
    printfn "  Output = ((Light - Bias - Dark) / Flat) + Pedestal"
    printfn ""
    printfn "Examples:"
    printfn "  # Full calibration (standard workflow)"
    printfn "  xisfprep calibrate -i \"lights/*.xisf\" -o \"cal/\" -b bias.xisf -d dark.xisf -f flat.xisf"
    printfn ""
    printfn "  # Create master darks with constant bias"
    printfn "  xisfprep calibrate -i \"darks/*.xisf\" -o \"darks_cal/\" --bias-level 500"
    printfn ""
    printfn "  # With output pedestal to prevent clipping"
    printfn "  xisfprep calibrate -i \"lights/*.xisf\" -o \"cal/\" -b bias.xisf -d dark.xisf -f flat.xisf --pedestal 100"
    printfn ""
    printfn "  # Pedestal only (shift histogram)"
    printfn "  xisfprep calibrate -i \"lights/*.xisf\" -o \"adjusted/\" --pedestal 100"

let parseArgs (args: string array) : CalibrationConfig =
    let rec parse (args: string list) (cfg: CalibrationConfig) =
        match args with
        | [] -> cfg
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { cfg with InputPattern = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { cfg with OutputDir = value }
        | "--bias" :: value :: rest | "-b" :: value :: rest ->
            parse rest { cfg with BiasFrame = Some value }
        | "--bias-level" :: value :: rest ->
            parse rest { cfg with BiasLevel = Some (float value) }
        | "--dark" :: value :: rest | "-d" :: value :: rest ->
            parse rest { cfg with DarkFrame = Some value }
        | "--flat" :: value :: rest | "-f" :: value :: rest ->
            parse rest { cfg with FlatFrame = Some value }
        | "--uncalibrated-dark" :: rest ->
            parse rest { cfg with UncalibratedDark = true }
        | "--uncalibrated-flat" :: rest ->
            parse rest { cfg with UncalibratedFlat = true }
        | "--optimize-dark" :: rest ->
            parse rest { cfg with OptimizeDark = true }
        | "--pedestal" :: value :: rest ->
            parse rest { cfg with OutputPedestal = int value }
        | "--suffix" :: value :: rest ->
            parse rest { cfg with Suffix = value }
        | "--overwrite" :: rest ->
            parse rest { cfg with Overwrite = true }
        | "--dry-run" :: rest ->
            parse rest { cfg with DryRun = true }
        | "--parallel" :: value :: rest ->
            parse rest { cfg with MaxParallel = int value }
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> parse rest { cfg with OutputFormat = Some fmt }
            | None -> failwithf "Unknown output format: %s (supported: uint8, uint16, uint32, float32, float64)" value
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        InputPattern = ""
        OutputDir = ""
        BiasFrame = None
        BiasLevel = None
        DarkFrame = None
        FlatFrame = None
        UncalibratedDark = false
        UncalibratedFlat = false
        OptimizeDark = false
        OutputPedestal = defaultPedestal
        Suffix = defaultSuffix
        Overwrite = false
        DryRun = false
        MaxParallel = defaultParallel
        OutputFormat = None
    }

    let cfg = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty cfg.InputPattern then failwith "Required argument: --input"
    if String.IsNullOrEmpty cfg.OutputDir then failwith "Required argument: --output"

    if cfg.BiasFrame.IsSome && cfg.BiasLevel.IsSome then
        failwith "--bias and --bias-level are mutually exclusive"

    if cfg.BiasFrame.IsNone && cfg.BiasLevel.IsNone && cfg.DarkFrame.IsNone && cfg.FlatFrame.IsNone && cfg.OutputPedestal = 0 then
        failwith "At least one of --bias, --bias-level, --dark, --flat, or --pedestal required"

    if cfg.UncalibratedDark && cfg.BiasFrame.IsNone && cfg.BiasLevel.IsNone then
        failwith "--uncalibrated-dark requires --bias or --bias-level"

    if cfg.UncalibratedFlat && cfg.BiasFrame.IsNone && cfg.BiasLevel.IsNone then
        failwith "--uncalibrated-flat requires --bias or --bias-level"

    if cfg.UncalibratedFlat && cfg.DarkFrame.IsNone then
        failwith "--uncalibrated-flat requires --dark"

    if cfg.OptimizeDark then
        if cfg.DarkFrame.IsNone then
            failwith "--optimize-dark requires --dark"
        if not cfg.UncalibratedDark then
            failwith "--optimize-dark requires --uncalibrated-dark (dark frame must be raw/uncalibrated)"
        if cfg.BiasFrame.IsNone && cfg.BiasLevel.IsNone then
            failwith "--optimize-dark requires --bias or --bias-level"

    if cfg.OutputPedestal < 0 || cfg.OutputPedestal > 65535 then
        failwith "Pedestal must be in range [0, 65535]"

    if cfg.MaxParallel < 1 then
        failwith "Parallel count must be at least 1"

    cfg

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

        // Use PixelIO to read pixels in any format (UInt8, UInt16, UInt32, Float32, Float64)
        let floatData = PixelIO.readPixelsAsFloat img

        return (floatData, width, height, channels, sampleFormat)
    }

/// Golden section search to find optimal dark scaling factor
let findOptimalDarkScale (lightPixels: float[]) (darkPixels: float[]) (biasPixels: float[] option) (biasLevel: float option)
                         (flatPixels: float[] option) (flatMedian: float option) (pedestal: int) : float =
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

        // Load bias frame or create constant bias
        match config.BiasFrame, config.BiasLevel with
        | Some path, _ ->
            Log.Information("Loading master bias: {Path}", path)
            let! (data, w, h, c, fmt) = loadFrameAsFloat path
            width <- w
            height <- h
            channels <- c
            biasData <- Some data
            formats <- fmt :: formats
        | None, Some level ->
            Log.Information("Using constant bias level: {Level}", level)
            biasData <- None // Will be applied as constant during calibration
        | None, None ->
            () // No bias

        // Load dark frame
        match config.DarkFrame with
        | Some path ->
            Log.Information("Loading master dark: {Path}", path)
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
            Log.Information("Loading master flat: {Path}", path)
            let! (data, w, h, c, fmt) = loadFrameAsFloat path
            if width > 0 && (w <> width || h <> height || c <> channels) then
                failwithf "Flat dimension mismatch: Expected %dx%dx%d, got %dx%dx%d" width height channels w h c
            width <- w
            height <- h
            channels <- c

            // Calculate median for normalization
            let median = calculateMedian data
            Log.Information("Flat median: {Median:F2}", median)
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

let calibratePixel (lightValue: float) (biasValue: float) (darkValue: float option)
                   (flatValue: float option) (flatMedian: float option) (pedestal: int)
                   (uncalibratedDark: bool) (darkScale: float) : uint16 =
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
        | Some f, Some m when f > 0.0 -> afterDark * (m / f)
        | _ -> afterDark

    let final = afterFlat + float pedestal
    uint16 (max 0.0 (min 65535.0 final))

let buildCalibrationHistory (config: CalibrationConfig) =
    [
        yield "Calibrated with xisfprep calibrate"

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

let getBiasValue (masters: MasterFrames) (config: CalibrationConfig) (index: int) =
    match masters.BiasData, config.BiasLevel with
    | Some data, _ -> data.[index]
    | None, Some level -> level
    | None, None -> 0.0

let processPixels (lightImg: XisfImage) (masters: MasterFrames) (config: CalibrationConfig) (pixelCount: int) (outputFormat: XisfSampleFormat) (normalize: bool) (darkScale: float) =
    // Read light frame pixels using PixelIO (handles all sample formats)
    let lightPixels = PixelIO.readPixelsAsFloat lightImg

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

        let calibrated = calibratePixel lightValue biasValue darkValue flatValue masters.FlatMedian config.OutputPedestal config.UncalibratedDark darkScale

        if calibrated = 0us then
            zeroCount <- zeroCount + 1

        calibratedFloats.[i] <- float calibrated

    // Write calibrated pixels in requested format (preserves input format by default)
    let calibratedData = PixelIO.writePixelsFromFloat calibratedFloats outputFormat normalize

    (calibratedData, zeroCount, flatZeroCount)

let logClippingWarnings (zeroCount: int) (flatZeroCount: int) (pixelCount: int) =
    let totalPixels = float pixelCount

    if flatZeroCount > 0 then
        let pct = (float flatZeroCount / totalPixels) * 100.0
        if pct > 0.1 then
            Log.Warning("Flat has {Count} zero pixels ({Pct:F2}%) - division skipped for these pixels", flatZeroCount, pct)

    if zeroCount > 0 then
        let pct = (float zeroCount / totalPixels) * 100.0
        if pct > 1.0 then
            Log.Warning("Output has {Count} zero pixels ({Pct:F2}%) - consider increasing --pedestal", zeroCount, pct)

let createCalibratedImage (lightImg: XisfImage) (calibratedData: byte[]) (config: CalibrationConfig) (outputFormat: XisfSampleFormat) (darkScale: float) =
    let dataBlock = InlineDataBlock(ReadOnlyMemory(calibratedData), XisfEncoding.Base64)

    let historyEntries =
        let baseHistory = buildCalibrationHistory config
        if config.OptimizeDark then
            baseHistory @ [sprintf "Dark scale factor: %.6f" darkScale]
        else
            baseHistory
    let historyFits = historyEntries |> List.map (fun text -> XisfFitsKeyword("HISTORY", "", text) :> XisfCoreElement)

    let existingElements =
        if isNull lightImg.AssociatedElements then [||]
        else lightImg.AssociatedElements |> Seq.toArray

    let allElements = Array.append existingElements (Array.ofList historyFits)

    let calibrationProps = buildCalibrationProperties config darkScale
    let existingProps =
        if isNull lightImg.Properties then [||]
        else lightImg.Properties |> Seq.toArray

    let allProps = Array.append existingProps (Array.ofList calibrationProps)

    // Get bounds per XISF spec: Some for Float32/Float64, None for integer formats
    let bounds =
        match PixelIO.getBoundsForFormat outputFormat with
        | Some b -> b
        | None -> Unchecked.defaultof<XisfImageBounds>  // null for integer formats

    XisfImage(
        lightImg.Geometry,
        outputFormat,
        lightImg.ColorSpace,
        dataBlock,
        bounds,
        lightImg.PixelStorage,
        lightImg.ImageType,
        lightImg.Offset,
        lightImg.Orientation,
        lightImg.ImageId,
        lightImg.Uuid,
        allProps,
        allElements
    )

let calibrateImage (lightPath: string) (masters: MasterFrames) (config: CalibrationConfig) : Async<XisfImage * float> =
    async {
        let reader = new XisfReader()
        let! lightUnit = reader.ReadAsync(lightPath) |> Async.AwaitTask

        if lightUnit.Images.Count = 0 then
            failwithf "No images in light frame: %s" lightPath

        let lightImg = lightUnit.Images.[0]
        let width = int lightImg.Geometry.Width
        let height = int lightImg.Geometry.Height
        let channels = int lightImg.Geometry.ChannelCount

        // Validate dimensions
        if masters.Width > 0 && (width <> masters.Width || height <> masters.Height || channels <> masters.Channels) then
            failwithf "Light frame dimension mismatch: Expected %dx%dx%d, got %dx%dx%d in %s"
                masters.Width masters.Height masters.Channels width height channels lightPath

        let pixelCount = width * height * channels

        // Determine output format (use override, or max precision of all inputs)
        let (outputFormat, normalize) =
            match config.OutputFormat with
            | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
            | None ->
                // Select most precise format from light frame and all masters
                let bestFormat = maxPrecisionFormat [lightImg.SampleFormat; masters.MaxPrecisionFormat]
                PixelIO.getRecommendedOutputFormat bestFormat

        // Compute optimal dark scale if enabled
        let darkScale =
            if config.OptimizeDark && masters.DarkData.IsSome then
                let lightPixels = PixelIO.readPixelsAsFloat lightImg
                let darkPixels = masters.DarkData.Value
                let darkScale = findOptimalDarkScale lightPixels darkPixels masters.BiasData config.BiasLevel masters.FlatData masters.FlatMedian config.OutputPedestal
                Log.Information("Optimized dark scale for {File}: {Scale:F6}", Path.GetFileName(lightPath), darkScale)
                darkScale
            else
                1.0

        // Process pixels using PixelIO (handles all sample formats)
        let (calibratedData, zeroCount, flatZeroCount) = processPixels lightImg masters config pixelCount outputFormat normalize darkScale

        logClippingWarnings zeroCount flatZeroCount pixelCount

        let calibratedImg = createCalibratedImage lightImg calibratedData config outputFormat darkScale
        return (calibratedImg, darkScale)
    }

let processFile (filePath: string) (masters: MasterFrames) (config: CalibrationConfig) : Async<Result<string, string>> =
    async {
        try
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            let ext = Path.GetExtension(filePath)
            let outputFileName = fileName + config.Suffix + ext
            let outputPath = Path.Combine(config.OutputDir, outputFileName)

            if not config.Overwrite && File.Exists(outputPath) then
                Log.Warning("Output file exists, skipping (use --overwrite to replace): {Path}", outputPath)
                return Ok filePath
            else
                if config.DryRun then
                    if config.Overwrite && File.Exists(outputPath) then
                        Log.Information("[DRY RUN] Would overwrite: {Path}", outputPath)
                    else
                        Log.Information("[DRY RUN] Would create: {Path}", outputPath)
                    return Ok filePath
                else
                    Log.Information("Calibrating: {File}", Path.GetFileName(filePath))

                    let! (calibratedImage, _darkScale) = calibrateImage filePath masters config

                    let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Calibrate v1.0")
                    let outUnit = XisfFactory.CreateMonolithic(metadata, calibratedImage)

                    let writer = new XisfWriter()
                    do! writer.WriteAsync(outUnit, outputPath) |> Async.AwaitTask

                    return Ok filePath
        with ex ->
            Log.Error("Failed to process {File}: {Error}", Path.GetFileName(filePath), ex.Message)
            return Error ex.Message
    }

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let config = parseArgs args

                let inputDir = Path.GetDirectoryName(config.InputPattern)
                let pattern = Path.GetFileName(config.InputPattern)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error("Input directory not found: {Dir}", actualDir)
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                    if files.Length = 0 then
                        Log.Error("No files found matching pattern: {Pattern}", config.InputPattern)
                        return 1
                    else
                        let plural = if files.Length = 1 then "" else "s"
                        let mode = if config.DryRun then " [DRY RUN]" else ""
                        printfn "Found %d file%s to calibrate%s" files.Length plural mode

                        // Load master frames
                        let! masters = loadMasterFrames config

                        // Create output directory
                        if not config.DryRun && not (Directory.Exists(config.OutputDir)) then
                            Directory.CreateDirectory(config.OutputDir) |> ignore
                            Log.Information("Created output directory: {Dir}", config.OutputDir)

                        // Process files in parallel
                        let! results =
                            files
                            |> Array.map (fun file -> processFile file masters config)
                            |> Async.Parallel

                        let successes = results |> Array.filter (function Ok _ -> true | Error _ -> false)
                        let failures = results |> Array.filter (function Error _ -> true | Ok _ -> false)

                        printfn ""
                        if config.DryRun then
                            printfn "Dry run complete - no files were modified"
                        else
                            printfn "Successfully calibrated %d of %d file%s" successes.Length files.Length plural

                        if failures.Length > 0 then
                            printfn "Failed to process %d file%s" failures.Length (if failures.Length = 1 then "" else "s")
                            return 1
                        else
                            return 0
            with ex ->
                Log.Error("Error: {Message}", ex.Message)
                printfn ""
                printfn "Run 'xisfprep calibrate --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
