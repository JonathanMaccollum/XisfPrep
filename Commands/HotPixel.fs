module Commands.HotPixel

open System
open System.IO
open Serilog
open FsToolkit.ErrorHandling
open XisfLib.Core
open Algorithms.Debayering
open Algorithms.HotPixelCorrection

let private defaultKHot = 5.0
let private defaultKCold = 5.0
let private defaultOverlap = 24
let private defaultSuffix = "_corrected"
let private defaultStarProtectionRatio = 1.5

type HotPixelOptions = {
    Input: string
    Output: string
    KHot: float
    KCold: float
    Overlap: int
    BayerPattern: BayerPattern option
    StarProtection: StarProtectionMethod
    StarProtectionRatio: float
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfLib.Core.XisfSampleFormat option
}

let showHelp() =
    printfn "hotpixel - Detect and correct hot/cold pixels using tiled processing"
    printfn ""
    printfn "Usage: xisfprep hotpixel [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input XISF files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for corrected files"
    printfn ""
    printfn "Detection Parameters:"
    printfn $"  --k-hot <value>           Hot pixel threshold in sigma (default: {defaultKHot})"
    printfn $"  --k-cold <value>          Cold pixel threshold in sigma (default: {defaultKCold})"
    printfn "  --k <value>               Set both k-hot and k-cold to same value"
    printfn ""
    printfn "Bayer Pattern:"
    printfn "  --bayer <pattern>         Bayer pattern: RGGB, BGGR, GRBG, GBRG"
    printfn "                              (omit for debayered RGB or mono images)"
    printfn ""
    printfn "Tiling:"
    printfn $"  --overlap <pixels>        Tile overlap: 8, 12, 16, 24, 32 (default: {defaultOverlap})"
    printfn ""
    printfn "Star Protection:"
    printfn "  --star-protection <mode>  Protect stars from being flagged as hot pixels"
    printfn "                              none: No protection (for testing/comparison)"
    printfn "                              isolation: Only correct isolated defects (≤1 bright neighbor)"
    printfn "                              ratio: Only correct if pixel >>  neighbors (recommended)"
    printfn $"  --star-ratio <value>      Minimum ratio for 'ratio' mode (default: {defaultStarProtectionRatio})"
    printfn ""
    printfn "Common Options:"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --parallel <n>            Number of parallel file operations (default: CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Algorithm:"
    printfn "  - Divides image into 128×128 tiles with configurable overlap"
    printfn "  - Computes global and local statistics (median, MAD) per color plane"
    printfn "  - Detects outliers using sigma-based thresholds"
    printfn "  - Corrects using VNG-style directional interpolation"
    printfn "  - Merges tiles by averaging corrections in overlap zones"
    printfn ""
    printfn "Best Practices:"
    printfn "  - Run on calibrated data (after dark/flat correction if available)"
    printfn "  - For Bayer data: specify pattern and run before debayering"
    printfn "  - Start with default k=5.0, increase if too aggressive"
    printfn "  - Use --verbose to see per-file correction statistics"
    printfn ""
    printfn "Examples:"
    printfn "  # Mono/RGB with isolation-based star protection"
    printfn "  xisfprep hotpixel -i \"lights/*.xisf\" -o \"corrected/\" --star-protection isolation"
    printfn ""
    printfn "  # Ratio-based star protection with custom threshold"
    printfn "  xisfprep hotpixel -i \"*.xisf\" -o \"out/\" --star-protection ratio --star-ratio 5.0"
    printfn ""
    printfn "  # Bayer data before debayering"
    printfn "  xisfprep hotpixel -i \"*.xisf\" -o \"fixed/\" --bayer RGGB --star-protection isolation"
    printfn ""
    printfn "  # Test without star protection (original behavior)"
    printfn "  xisfprep hotpixel -i \"*.xisf\" -o \"out/\" --star-protection none"

let parseArgs (args: string array) : HotPixelOptions =
    let argsList = List.ofArray args

    let (remaining, common) =
        SharedInfra.ArgumentParsing.parseCommonFlags
            argsList
            SharedInfra.ArgumentParsing.defaultCommonOptions

    let rec parseSpecific
        (args: string list)
        (kHot: float)
        (kCold: float)
        (overlap: int)
        (bayerPattern: BayerPattern option)
        (starProtection: StarProtectionMethod)
        (starRatio: float)
        (customSuffix: string option) =
        match args with
        | [] ->
            let finalSuffix = customSuffix |> Option.defaultValue defaultSuffix
            (kHot, kCold, overlap, bayerPattern, starProtection, starRatio, finalSuffix)
        | "--k-hot" :: value :: rest ->
            parseSpecific rest (float value) kCold overlap bayerPattern starProtection starRatio customSuffix
        | "--k-cold" :: value :: rest ->
            parseSpecific rest kHot (float value) overlap bayerPattern starProtection starRatio customSuffix
        | "--k" :: value :: rest ->
            let k = float value
            parseSpecific rest k k overlap bayerPattern starProtection starRatio customSuffix
        | "--overlap" :: value :: rest ->
            parseSpecific rest kHot kCold (int value) bayerPattern starProtection starRatio customSuffix
        | "--bayer" :: value :: rest ->
            let pattern = match value.ToUpper() with
                          | "RGGB" -> RGGB
                          | "BGGR" -> BGGR
                          | "GRBG" -> GRBG
                          | "GBRG" -> GBRG
                          | _ -> failwithf "Unknown Bayer pattern: %s (supported: RGGB, BGGR, GRBG, GBRG)" value
            parseSpecific rest kHot kCold overlap (Some pattern) starProtection starRatio customSuffix
        | "--star-protection" :: value :: rest ->
            let protection =
                match value.ToLower() with
                | "none" -> NoProtection
                | "isolation" -> Isolation
                | "ratio" -> Ratio
                | _ -> failwithf "Unknown star protection mode: %s (supported: none, isolation, ratio)" value
            parseSpecific rest kHot kCold overlap bayerPattern protection starRatio customSuffix
        | "--star-ratio" :: value :: rest ->
            parseSpecific rest kHot kCold overlap bayerPattern starProtection (float value) customSuffix
        | "--suffix" :: value :: rest ->
            parseSpecific rest kHot kCold overlap bayerPattern starProtection starRatio (Some value)
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let (kHot, kCold, overlap, bayerPattern, starProtection, starRatio, suffix) =
        parseSpecific remaining defaultKHot defaultKCold defaultOverlap None NoProtection defaultStarProtectionRatio None

    if String.IsNullOrEmpty common.Input then
        failwith "Required argument: --input"
    if String.IsNullOrEmpty common.Output then
        failwith "Required argument: --output"
    if kHot <= 0.0 then
        failwithf "k-hot must be positive, got: %f" kHot
    if kCold <= 0.0 then
        failwithf "k-cold must be positive, got: %f" kCold
    if not (List.contains overlap [8; 12; 16; 24; 32]) then
        failwithf "overlap must be one of: 8, 12, 16, 24, 32 (got: %d)" overlap

    {
        Input = common.Input
        Output = common.Output
        KHot = kHot
        KCold = kCold
        Overlap = overlap
        BayerPattern = bayerPattern
        StarProtection = starProtection
        StarProtectionRatio = starRatio
        Suffix = suffix
        Overwrite = common.Overwrite
        MaxParallel = common.MaxParallel
        OutputFormat = common.OutputFormat
    }

type HotPixelError =
    | LoadFailed of XisfIO.XisfError
    | ProcessingFailed of Algorithms.HotPixelCorrection.HotPixelError
    | SaveFailed of XisfIO.XisfError

    override this.ToString() =
        match this with
        | LoadFailed err -> err.ToString()
        | ProcessingFailed err -> err.ToString()
        | SaveFailed err -> err.ToString()

let processFile (inputPath: string) (outputPaths: string list) (opts: HotPixelOptions) : Async<Result<unit, HotPixelError>> =
    asyncResult {
        let outputPath = List.head outputPaths
        let fileName = Path.GetFileName inputPath

        Log.Information("Processing: {File}", fileName)

        let! (originalImg, metadata, pixels) =
            XisfIO.loadImageWithPixels inputPath
            |> AsyncResult.mapError LoadFailed

        Log.Verbose("  Loaded: {Width}×{Height}, {Channels} channel(s), {Format}",
                    metadata.Width, metadata.Height, metadata.Channels, metadata.Format)

        match opts.BayerPattern with
        | Some pattern ->
            Log.Verbose("  Bayer mode: {Pattern}", pattern)
            if metadata.Channels <> 1 then
                Log.Warning("  Bayer pattern specified but image has {Channels} channels (expected 1)", metadata.Channels)
        | None ->
            Log.Verbose("  RGB/Mono mode: {Channels} channel(s)", metadata.Channels)

        let config: Algorithms.HotPixelCorrection.DetectionConfig = {
            KHot = opts.KHot
            KCold = opts.KCold
            TileSize = 128
            Overlap = opts.Overlap
            BayerPattern = opts.BayerPattern
            StarProtection = opts.StarProtection
            StarProtectionRatio = opts.StarProtectionRatio
        }

        Log.Verbose("  Detection config: K-hot={KHot}, K-cold={KCold}, overlap={Overlap}px, star-protection={StarProtection}",
                    config.KHot, config.KCold, config.Overlap, config.StarProtection)

        let! (correctedPixels, result) =
            Algorithms.HotPixelCorrection.processImage pixels metadata.Width metadata.Height config
            |> Result.mapError ProcessingFailed
            |> AsyncResult.ofResult

        let percentage =
            if metadata.Width * metadata.Height > 0 then
                float result.TotalCorrected / float (metadata.Width * metadata.Height) * 100.0
            else 0.0

        Log.Information("  Corrected: {Total} pixels ({Hot} hot, {Cold} cold) - {Percentage:F3}%",
                        result.TotalCorrected, result.HotPixelCount, result.ColdPixelCount, percentage)

        let (outputFormat, normalize) =
            match opts.OutputFormat with
            | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
            | None -> PixelIO.getRecommendedOutputFormat metadata.Format

        let correctedBytes = PixelIO.writePixelsFromFloat correctedPixels outputFormat normalize

        // Build detailed history entry
        let bayerStr =
            match opts.BayerPattern with
            | Some pat -> $", bayer={pat}"
            | None -> ""
        let starProtStr =
            match opts.StarProtection with
            | Ratio -> $", star-protection=ratio({opts.StarProtectionRatio})"
            | Isolation -> ", star-protection=isolation"
            | NoProtection -> ""
        let historyEntry =
            $"HotPixel: k-hot={opts.KHot}, k-cold={opts.KCold}, corrected={result.TotalCorrected} ({result.HotPixelCount} hot, {result.ColdPixelCount} cold){bayerStr}{starProtStr}"

        // Build XISF properties for structured metadata
        let xisfProps = [|
            XisfStringProperty("XisfPrep:HotPixel:Version", "0.1") :> XisfProperty
            XisfScalarProperty<float>("XisfPrep:HotPixel:KHot", opts.KHot) :> XisfProperty
            XisfScalarProperty<float>("XisfPrep:HotPixel:KCold", opts.KCold) :> XisfProperty
            XisfScalarProperty<int>("XisfPrep:HotPixel:TileSize", 128) :> XisfProperty
            XisfScalarProperty<int>("XisfPrep:HotPixel:Overlap", opts.Overlap) :> XisfProperty
            XisfStringProperty("XisfPrep:HotPixel:StarProtection", opts.StarProtection.ToString()) :> XisfProperty
            XisfScalarProperty<int>("XisfPrep:HotPixel:CorrectedTotal", result.TotalCorrected) :> XisfProperty
            XisfScalarProperty<int>("XisfPrep:HotPixel:CorrectedHot", result.HotPixelCount) :> XisfProperty
            XisfScalarProperty<int>("XisfPrep:HotPixel:CorrectedCold", result.ColdPixelCount) :> XisfProperty
            XisfStringProperty("XisfPrep:HotPixel:ProcessedUTC", DateTime.UtcNow.ToString("O")) :> XisfProperty
        |]

        // Add optional properties
        let xisfPropsWithOptional =
            match opts.StarProtection with
            | Ratio ->
                Array.append xisfProps [|
                    XisfScalarProperty<float>("XisfPrep:HotPixel:StarRatio", opts.StarProtectionRatio) :> XisfProperty
                |]
            | _ -> xisfProps

        let xisfPropsComplete =
            match opts.BayerPattern with
            | Some pat ->
                Array.append xisfPropsWithOptional [|
                    XisfStringProperty("XisfPrep:HotPixel:BayerPattern", pat.ToString()) :> XisfProperty
                |]
            | None -> xisfPropsWithOptional

        let! outputImg =
            XisfIO.createOutputImage originalImg correctedBytes {
                XisfIO.defaultOutputImageConfig with
                    Dimensions = Some (metadata.Width, metadata.Height, metadata.Channels)
                    Format = Some outputFormat
                    HistoryEntries = [historyEntry]
                    AdditionalProps = xisfPropsComplete
            }
            |> Result.mapError LoadFailed
            |> AsyncResult.ofResult

        do! XisfIO.writeImage outputPath "XisfPrep HotPixel v0.1" outputImg
            |> AsyncResult.mapError SaveFailed

        let sizeMB = (FileInfo outputPath).Length / 1024L / 1024L
        Log.Information("  -> {Output} ({Size} MB)", Path.GetFileName outputPath, sizeMB)

        return ()
    }

let run (args: string array) : int =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let opts = parseArgs args

                Log.Information("Hot/Cold Pixel Correction")
                Log.Information("  K-hot: {KHot}, K-cold: {KCold}", opts.KHot, opts.KCold)
                Log.Information("  Overlap: {Overlap}px", opts.Overlap)
                Log.Information("  Star protection: {StarProtection}", opts.StarProtection)
                if opts.StarProtection = Ratio then
                    Log.Information("  Star ratio threshold: {Ratio}", opts.StarProtectionRatio)
                match opts.BayerPattern with
                | Some pattern -> Log.Information("  Bayer: {Pattern}", pattern)
                | None -> Log.Information("  Mode: RGB/Mono")

                let inputDir = System.IO.Path.GetDirectoryName(opts.Input)
                let pattern = System.IO.Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (System.IO.Directory.Exists(actualDir)) then
                    Log.Error("Input directory not found: {Dir}", actualDir)
                    return 1
                else
                    let files = System.IO.Directory.GetFiles(actualDir, pattern)

                    if files.Length = 0 then
                        Log.Error("No files found matching pattern: {Pattern}", opts.Input)
                        return 1
                    else
                        printfn "Found %d file%s to process" files.Length (if files.Length = 1 then "" else "s")
                        printfn ""

                        let batchConfig: SharedInfra.BatchProcessing.BatchConfig<HotPixelOptions> = {
                            Files = files
                            OutputDir = opts.Output
                            Suffix = opts.Suffix
                            Overwrite = opts.Overwrite
                            MaxParallel = opts.MaxParallel
                            Config = opts
                        }

                        let buildOutputPaths baseName suffix outputDir =
                            let outFileName = $"{baseName}{suffix}.xisf"
                            let outPath = System.IO.Path.Combine(outputDir, outFileName)
                            Some [outPath]

                        let! exitCode = SharedInfra.BatchProcessing.processBatch batchConfig buildOutputPaths processFile

                        return exitCode
            with ex ->
                Log.Error("Error: {Message}", ex.Message)
                printfn ""
                printfn "Run 'xisfprep hotpixel --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
