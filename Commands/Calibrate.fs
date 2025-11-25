module Commands.Calibrate

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Calibration
open Algorithms.Statistics

// --- Defaults ---
let private defaultPedestal = 0
let private defaultSuffix = "_cal"
let private defaultParallel = Environment.ProcessorCount
// ---

type CalibrateCommandConfig = {
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

let parseArgs (args: string array) : CalibrateCommandConfig =
    let rec parse (args: string list) (cfg: CalibrateCommandConfig) =
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

/// Convert command config to calibration config
let toCalibrationConfig (config: CalibrateCommandConfig) : CalibrationConfig =
    {
        BiasFrame = config.BiasFrame
        BiasLevel = config.BiasLevel
        DarkFrame = config.DarkFrame
        FlatFrame = config.FlatFrame
        UncalibratedDark = config.UncalibratedDark
        UncalibratedFlat = config.UncalibratedFlat
        OptimizeDark = config.OptimizeDark
        OutputPedestal = config.OutputPedestal
    }

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

let createCalibratedImage (lightImg: XisfImage) (calibratedData: byte[]) (calConfig: CalibrationConfig) (outputFormat: XisfSampleFormat) (darkScale: float) =
    let dataBlock = InlineDataBlock(ReadOnlyMemory(calibratedData), XisfEncoding.Base64)

    let historyEntries =
        let baseHistory = buildCalibrationHistory calConfig
        if calConfig.OptimizeDark then
            baseHistory @ [sprintf "Dark scale factor: %.6f" darkScale]
        else
            baseHistory
    let historyFits = historyEntries |> List.map (fun text -> XisfFitsKeyword("HISTORY", "", text) :> XisfCoreElement)

    let existingElements =
        if isNull lightImg.AssociatedElements then [||]
        else lightImg.AssociatedElements |> Seq.toArray

    let allElements = Array.append existingElements (Array.ofList historyFits)

    let calibrationProps = buildCalibrationProperties calConfig darkScale
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

let calibrateImage (lightPath: string) (masters: MasterFrames) (calConfig: CalibrationConfig) (outputFormat: XisfSampleFormat option) : Async<XisfImage * float> =
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
        match validateDimensions masters width height channels with
        | Error msg -> failwith msg
        | Ok () -> ()

        let pixelCount = width * height * channels

        // Determine output format (use override, or max precision of all inputs)
        let (outputFmt, normalize) =
            match outputFormat with
            | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
            | None ->
                // Select most precise format from light frame and all masters
                let bestFormat = maxPrecisionFormat [lightImg.SampleFormat; masters.MaxPrecisionFormat]
                PixelIO.getRecommendedOutputFormat bestFormat

        // Read light pixels
        let lightPixels = PixelIO.readPixelsAsFloat lightImg

        // Calibrate pixels (dark optimization happens inside if enabled)
        let result = calibratePixels lightPixels masters calConfig

        // Log dark scale if optimization was used
        if calConfig.OptimizeDark && result.DarkScale <> 1.0 then
            Log.Information("Optimized dark scale for {File}: {Scale:F6}", Path.GetFileName(lightPath), result.DarkScale)

        logClippingWarnings result.ClippedPixels result.FlatZeroPixels pixelCount

        // Convert to output format
        let calibratedData = PixelIO.writePixelsFromFloat result.CalibratedPixels outputFmt normalize

        let calibratedImg = createCalibratedImage lightImg calibratedData calConfig outputFmt result.DarkScale
        return (calibratedImg, result.DarkScale)
    }

let processFile (filePath: string) (masters: MasterFrames) (cmdConfig: CalibrateCommandConfig) : Async<Result<string, string>> =
    async {
        try
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            let ext = Path.GetExtension(filePath)
            let outputFileName = fileName + cmdConfig.Suffix + ext
            let outputPath = Path.Combine(cmdConfig.OutputDir, outputFileName)

            if not cmdConfig.Overwrite && File.Exists(outputPath) then
                Log.Warning("Output file exists, skipping (use --overwrite to replace): {Path}", outputPath)
                return Ok filePath
            else
                if cmdConfig.DryRun then
                    if cmdConfig.Overwrite && File.Exists(outputPath) then
                        Log.Information("[DRY RUN] Would overwrite: {Path}", outputPath)
                    else
                        Log.Information("[DRY RUN] Would create: {Path}", outputPath)
                    return Ok filePath
                else
                    Log.Information("Calibrating: {File}", Path.GetFileName(filePath))

                    let calConfig = toCalibrationConfig cmdConfig
                    let! (calibratedImage, _darkScale) = calibrateImage filePath masters calConfig cmdConfig.OutputFormat

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

                        // Load master frames using Algorithms.Calibration
                        let calConfig = toCalibrationConfig config
                        let! masters = loadMasterFrames calConfig

                        // Log what was loaded
                        match calConfig.BiasFrame with
                        | Some path -> Log.Information("Master bias: {Path}", path)
                        | None -> ()
                        match calConfig.BiasLevel with
                        | Some level -> Log.Information("Bias level: {Level}", level)
                        | None -> ()
                        match calConfig.DarkFrame with
                        | Some path -> Log.Information("Master dark: {Path}", path)
                        | None -> ()
                        match calConfig.FlatFrame with
                        | Some path ->
                            Log.Information("Master flat: {Path}", path)
                            match masters.FlatMedian with
                            | Some median -> Log.Information("Flat median: {Median:F2}", median)
                            | None -> ()
                        | None -> ()

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
