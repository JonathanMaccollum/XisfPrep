module Commands.Bin

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Binning
open FsToolkit.ErrorHandling

// --- Defaults ---
let private defaultFactor = 2
let private defaultMethod = Average
let private defaultSuffix factor = $"_bin{factor}x"
// ---

type BinOptions = {
    Input: string
    Output: string
    Factor: int
    Method: BinningMethod
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
}

type BinError =
    | LoadFailed of XisfIO.XisfError
    | ValidationFailed of Algorithms.Binning.BinningError
    | ImageCreationFailed of XisfIO.XisfError
    | SaveFailed of XisfIO.XisfError

    override this.ToString() =
        match this with
        | LoadFailed err -> err.ToString()
        | ValidationFailed err -> err.ToString()
        | ImageCreationFailed err -> err.ToString()
        | SaveFailed err -> err.ToString()

let showHelp() =
    printfn "bin - Downsample images by binning pixels together"
    printfn ""
    printfn "Usage: xisfprep bin [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for binned files"
    printfn ""
    printfn "Optional:"
    printfn $"  --factor <n>              Binning factor: 2, 3, 4, 5, or 6 (default: {defaultFactor})"
    printfn $"  --method <method>         Binning method (default: {defaultMethod.ToString().ToLower()})"
    printfn "                              average - Average binning (preserves flux)"
    printfn "                              median  - Median binning (robust to outliers)"
    printfn "                              sum     - Sum binning (adds pixel values)"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix defaultFactor} where N is factor)"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Use Cases:"
    printfn "  - Quick preview of large images"
    printfn "  - Reduce file size for testing workflows"
    printfn "  - Match image scales between different instruments"
    printfn "  - Speed up processing for alignment tests"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep bin -i \"lights/*.xisf\" -o \"binned/\" --factor 2"
    printfn "  xisfprep bin -i \"lights/*.xisf\" -o \"preview/\" --factor 4 --method median"

let parseArgs (args: string array) : BinOptions =
    let argsList = List.ofArray args

    // Parse common flags first
    let (remaining, common) = SharedInfra.ArgumentParsing.parseCommonFlags argsList SharedInfra.ArgumentParsing.defaultCommonOptions

    // Track suffix separately since it depends on factor
    let rec parseSpecific (args: string list) (factor: int) (method: BinningMethod) (customSuffix: string option) =
        match args with
        | [] ->
            let finalSuffix = customSuffix |> Option.defaultValue (defaultSuffix factor)
            (factor, method, finalSuffix)
        | "--factor" :: value :: rest ->
            parseSpecific rest (int value) method customSuffix
        | "--method" :: value :: rest ->
            let m = match value.ToLower() with
                    | "average" -> Average
                    | "median" -> Median
                    | "sum" -> Sum
                    | _ -> failwithf "Unknown binning method: %s" value
            parseSpecific rest factor m customSuffix
        | "--suffix" :: value :: rest ->
            parseSpecific rest factor method (Some value)
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let (factor, method, suffix) = parseSpecific remaining defaultFactor defaultMethod None

    // Build options
    let opts = {
        Input = common.Input
        Output = common.Output
        Factor = factor
        Method = method
        Suffix = suffix
        Overwrite = common.Overwrite
        MaxParallel = common.MaxParallel
        OutputFormat = common.OutputFormat
    }

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

    let binConfig = { Factor = opts.Factor; Method = opts.Method }
    match validateConfig binConfig with
    | Error err -> failwith (err.ToString())
    | Ok () -> ()

    opts

let processFile (inputPath: string) (outputPaths: string list) (opts: BinOptions) : Async<Result<unit, BinError>> =
    asyncResult {
        let outputPath = List.head outputPaths  // Bin always produces single output

        printfn "Processing: %s" (Path.GetFileName inputPath)

        // Load image
        let! (originalImg, metadata, pixels) = XisfIO.loadImageWithPixels inputPath
                                                |> AsyncResult.mapError LoadFailed

        // Validate dimensions
        let binConfig = { Factor = opts.Factor; Method = opts.Method }
        let! _ = validateDimensions metadata.Width metadata.Height metadata.Channels binConfig.Factor
                 |> Result.mapError ValidationFailed
                 |> AsyncResult.ofResult

        // Apply binning
        let binResult = binPixels pixels metadata.Width metadata.Height metadata.Channels binConfig

        // Determine output format
        let (outputFormat, normalize) =
            match opts.OutputFormat with
            | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
            | None -> PixelIO.getRecommendedOutputFormat metadata.Format

        // Convert binned floats to bytes
        let binnedBytes = PixelIO.writePixelsFromFloat binResult.BinnedPixels outputFormat normalize

        // Create output image
        let historyEntry = $"Binned {opts.Factor}x{opts.Factor} using {opts.Method} method"
        let! outputImg = XisfIO.createOutputImage originalImg binnedBytes {
                             XisfIO.defaultOutputImageConfig with
                                 Dimensions = Some (binResult.NewWidth, binResult.NewHeight, binResult.NewChannels)
                                 Format = Some outputFormat
                                 HistoryEntries = [historyEntry]
                         }
                         |> Result.mapError ImageCreationFailed
                         |> AsyncResult.ofResult

        // Write output
        do! XisfIO.writeImage outputPath "XisfPrep Bin v1.0" outputImg
            |> AsyncResult.mapError SaveFailed

        let sizeMB = (FileInfo outputPath).Length / 1024L / 1024L
        printfn "  -> %s (%d MB, %dx%d)" (Path.GetFileName outputPath) sizeMB binResult.NewWidth binResult.NewHeight
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

                Log.Information("Binning images with factor {Factor} using {Method} method", opts.Factor, opts.Method)

                // Resolve input files
                let inputDir = Path.GetDirectoryName(opts.Input)
                let pattern = Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error("Input directory not found: {Dir}", actualDir)
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern)

                    if files.Length = 0 then
                        Log.Error("No files found matching pattern: {Pattern}", opts.Input)
                        return 1
                    else
                        printfn "Found %d files to process" files.Length
                        printfn ""

                        // Build batch config
                        let batchConfig: SharedInfra.BatchProcessing.BatchConfig<BinOptions> = {
                            Files = files
                            OutputDir = opts.Output
                            Suffix = opts.Suffix
                            Overwrite = opts.Overwrite
                            MaxParallel = opts.MaxParallel
                            Config = opts
                        }

                        // Build output paths function (single output file per input)
                        let buildOutputPaths baseName suffix outputDir =
                            let outFileName = $"{baseName}{suffix}.xisf"
                            let outPath = Path.Combine(outputDir, outFileName)
                            Some [outPath]

                        // Process batch
                        return! SharedInfra.BatchProcessing.processBatch batchConfig buildOutputPaths processFile

            with ex ->
                Log.Error("Error: {Message}", ex.Message)
                printfn ""
                printfn "Run 'xisfprep bin --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
