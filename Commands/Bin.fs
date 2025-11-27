module Commands.Bin

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Binning

// --- Defaults ---
let private defaultFactor = 2
let private defaultMethod = Average
let private defaultSuffix factor = $"_bin{factor}x"
let private defaultParallel = System.Environment.ProcessorCount
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

/// Convert command options to binning config
let toBinningConfig (opts: BinOptions) : BinningConfig =
    {
        Factor = opts.Factor
        Method = opts.Method
    }

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
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
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
    // Track suffix separately since it depends on factor
    let rec parse (args: string list) (opts: BinOptions) (customSuffix: string option) =
        match args with
        | [] ->
            // Apply suffix default based on final factor
            let finalSuffix = customSuffix |> Option.defaultValue (defaultSuffix opts.Factor)
            { opts with Suffix = finalSuffix }
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value } customSuffix
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = value } customSuffix
        | "--factor" :: value :: rest ->
            parse rest { opts with Factor = int value } customSuffix
        | "--method" :: value :: rest ->
            let m = match value.ToLower() with
                    | "average" -> Average
                    | "median" -> Median
                    | "sum" -> Sum
                    | _ -> failwithf "Unknown binning method: %s" value
            parse rest { opts with Method = m } customSuffix
        | "--suffix" :: value :: rest ->
            parse rest opts (Some value)
        | "--overwrite" :: rest ->
            parse rest { opts with Overwrite = true } customSuffix
        | "--parallel" :: value :: rest ->
            parse rest { opts with MaxParallel = int value } customSuffix
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> parse rest { opts with OutputFormat = Some fmt } customSuffix
            | None -> failwithf "Unknown output format: %s (supported: uint8, uint16, uint32, float32, float64)" value
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Factor = defaultFactor
        Method = defaultMethod
        Suffix = ""  // Will be set based on factor
        Overwrite = false
        MaxParallel = defaultParallel
        OutputFormat = None
    }

    let opts = parse (List.ofArray args) defaults None

    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

    // Validate binning configuration using shared validation
    let binConfig = toBinningConfig opts
    match validateConfig binConfig with
    | Error err -> failwith (err.ToString())
    | Ok () -> ()

    opts

let createOutputImage (img: XisfImage) (binnedData: byte[]) newWidth newHeight channels factor (method: BinningMethod) (outputFormat: XisfSampleFormat) =
    let newGeometry = XisfImageGeometry([| uint32 newWidth; uint32 newHeight |], uint32 channels)
    let dataBlock = InlineDataBlock(ReadOnlyMemory(binnedData), XisfEncoding.Base64)

    let historyEntry = $"Binned {factor}x{factor} using {method} method"
    let updatedFits =
        if isNull img.AssociatedElements then
            [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]
        else
            let existingFits = img.AssociatedElements :> seq<_> |> Seq.toArray
            Array.append existingFits [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]

    // Get bounds per XISF spec: Some for Float32/Float64, None for integer formats
    let bounds =
        match PixelIO.getBoundsForFormat outputFormat with
        | Some b -> b
        | None -> Unchecked.defaultof<XisfImageBounds>  // null for integer formats

    XisfImage(
        newGeometry,
        outputFormat,
        img.ColorSpace,
        dataBlock,
        bounds,
        img.PixelStorage,
        img.ImageType,
        img.Offset,
        img.Orientation,
        img.ImageId,
        img.Uuid,
        img.Properties,
        updatedFits
    )

let binImage (inputPath: string) (outputDir: string) (binConfig: BinningConfig) (suffix: string) (overwrite: bool) (outputFormatOverride: XisfSampleFormat option) : Async<bool> =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)
            let outFileName = $"{baseName}{suffix}.xisf"
            let outPath = Path.Combine(outputDir, outFileName)

            // Check if output file exists and skip if not overwriting
            if File.Exists(outPath) && not overwrite then
                Log.Warning($"Output file '{outFileName}' already exists, skipping (use --overwrite to replace)")
                return true  // Return true to not count as failure
            else

            printfn $"Processing: {fileName}"

            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount

            // Validate dimensions using shared validation
            match validateDimensions width height channels binConfig.Factor with
            | Error err ->
                Log.Error(err.ToString())
                return false
            | Ok _truncated ->
                // Warning already logged by validateDimensions if needed

                // Read pixels as float using PixelIO (handles all sample formats)
                let pixelFloats = PixelIO.readPixelsAsFloat img

                // Apply binning using algorithm module
                let binResult = binPixels pixelFloats width height channels binConfig

                // Determine output format (use override or preserve input format)
                let (outputFormat, normalize) =
                    match outputFormatOverride with
                    | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
                    | None -> PixelIO.getRecommendedOutputFormat img.SampleFormat

                // Convert binned floats to output format
                let binnedData = PixelIO.writePixelsFromFloat binResult.BinnedPixels outputFormat normalize

                let binnedImage = createOutputImage img binnedData binResult.NewWidth binResult.NewHeight binResult.NewChannels binConfig.Factor binConfig.Method outputFormat

                let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Bin v1.0")
                let outUnit = XisfFactory.CreateMonolithic(metadata, binnedImage)

                let writer = new XisfWriter()
                do! writer.WriteAsync(outUnit, outPath) |> Async.AwaitTask

                let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
                printfn $"  -> {outFileName} ({sizeMB} MB, {binResult.NewWidth}x{binResult.NewHeight})"

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

                Log.Information($"Binning images with factor {opts.Factor} using {opts.Method} method")

                if not (Directory.Exists(opts.Output)) then
                    Directory.CreateDirectory(opts.Output) |> ignore
                    Log.Information($"Created output directory: {opts.Output}")

                let inputDir = Path.GetDirectoryName(opts.Input)
                let pattern = Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern)

                    if files.Length = 0 then
                        Log.Error($"No files found matching pattern: {opts.Input}")
                        return 1
                    else
                        printfn $"Found {files.Length} files to process"
                        printfn ""

                        // Process all files in parallel with max parallelism limit
                        let binConfig = toBinningConfig opts
                        let tasks = files |> Array.map (fun f ->
                            binImage f opts.Output binConfig opts.Suffix opts.Overwrite opts.OutputFormat)
                        let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                        let successCount = results |> Array.filter id |> Array.length
                        let failCount = results.Length - successCount

                        printfn ""
                        printfn $"Completed: {successCount} succeeded, {failCount} failed"

                        return if failCount > 0 then 1 else 0
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep bin --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
