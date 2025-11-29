module Commands.Debayer

open System
open System.IO
open Serilog
open XisfLib.Core
open Algorithms.Debayering

// --- Defaults ---
let private defaultPattern = "RGGB"
let private defaultSuffix = "_d"
let private defaultParallel = System.Environment.ProcessorCount
// ---

type DebayerOptions = {
    Input: string
    Output: string
    Pattern: string option
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
    Split: bool
}

let showHelp() =
    printfn "debayer - Convert Bayer mosaic to RGB using VNG interpolation"
    printfn ""
    printfn "Usage: xisfprep debayer [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input Bayer mosaic files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for RGB files"
    printfn ""
    printfn "Optional:"
    printfn $"  --pattern, -p <pattern>   Bayer pattern override (default: auto-detect from FITS, fallback {defaultPattern})"
    printfn "                              Supported: RGGB, BGGR, GRBG, GBRG"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --split                   Output as separate R, G, B monochrome files"
    printfn "                              Creates _R.xisf, _G.xisf, _B.xisf instead of RGB"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Algorithm:"
    printfn "  VNG (Variable Number of Gradients) - High quality gradient-based interpolation"
    printfn "  Uses 5x5 kernel for interior pixels, bilinear for borders"
    printfn ""
    printfn "Process:"
    printfn "  1. Validates input is single-channel monochrome"
    printfn "  2. Reads Bayer pattern from FITS BAYERPAT keyword or uses override"
    printfn "  3. Applies VNG interpolation"
    printfn "  4. Outputs 3-channel RGB image (or separate R/G/B with --split)"
    printfn "  5. Removes ColorFilterArray and Bayer FITS keywords from output"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep debayer -i \"lights/*.xisf\" -o \"rgb/\""
    printfn "  xisfprep debayer -i \"lights/*.xisf\" -o \"rgb/\" --pattern RGGB --overwrite"
    printfn "  xisfprep debayer -i \"lights/*.xisf\" -o \"mono/\" --split"

let parseArgs (args: string array) : DebayerOptions =
    let rec parse (args: string list) (opts: DebayerOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = value }
        | "--pattern" :: value :: rest | "-p" :: value :: rest ->
            let p = value.ToUpper()
            if not (p = "RGGB" || p = "BGGR" || p = "GRBG" || p = "GBRG") then
                failwithf "Unknown Bayer pattern: %s (supported: RGGB, BGGR, GRBG, GBRG)" value
            parse rest { opts with Pattern = Some p }
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
        | "--split" :: rest ->
            parse rest { opts with Split = true }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Pattern = None
        Suffix = defaultSuffix
        Overwrite = false
        MaxParallel = defaultParallel
        OutputFormat = None
        Split = false
    }

    let opts = parse (List.ofArray args) defaults

    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

    opts

// Extract bayer pattern string from FITS keyword and parse to BayerPattern type
let parseBayerPattern (img: XisfImage) (patternOverride: string option) : BayerPattern =
    let patternStr =
        match patternOverride with
        | Some p -> p
        | None ->
            if isNull img.AssociatedElements then defaultPattern
            else
                img.AssociatedElements :> seq<_>
                |> Seq.toArray
                |> Array.tryPick (fun e ->
                    if e :? XisfFitsKeyword then
                        let fits = e :?> XisfFitsKeyword
                        if fits.Name = "BAYERPAT" then Some (fits.Value.Trim([|'\''|])) else None
                    else None)
                |> Option.defaultValue defaultPattern

    match patternStr.ToUpper() with
    | "RGGB" -> BayerPattern.RGGB
    | "BGGR" -> BayerPattern.BGGR
    | "GRBG" -> BayerPattern.GRBG
    | "GBRG" -> BayerPattern.GBRG
    | _ -> failwithf "Unknown Bayer pattern: %s" patternStr

let debayerImage (inputPath: string) (outputDir: string) (patternOverride: string option) (suffix: string) (overwrite: bool) (outputFormatOverride: XisfSampleFormat option) (split: bool) : Async<bool> =
    async {
        try
            let fileName = Path.GetFileName(inputPath)
            let baseName = Path.GetFileNameWithoutExtension(inputPath)

            // Check output files based on split mode
            let shouldSkip =
                if split then
                    let outPaths = [| "_R"; "_G"; "_B" |] |> Array.map (fun ch ->
                        let outFileName = $"{baseName}{suffix}{ch}.xisf"
                        Path.Combine(outputDir, outFileName))
                    let existingFiles = outPaths |> Array.filter File.Exists
                    if existingFiles.Length > 0 && not overwrite then
                        for f in existingFiles do
                            Log.Warning($"Output file '{Path.GetFileName(f)}' already exists, skipping (use --overwrite to replace)")
                        true
                    else false
                else
                    let outFileName = $"{baseName}{suffix}.xisf"
                    let outPath = Path.Combine(outputDir, outFileName)
                    if File.Exists(outPath) && not overwrite then
                        Log.Warning($"Output file '{outFileName}' already exists, skipping (use --overwrite to replace)")
                        true
                    else false

            if shouldSkip then
                return true
            else

            printfn $"Processing: {fileName}"

            match! XisfIO.loadImageWithPixels inputPath with
            | Error err ->
                Log.Error($"Failed to load {fileName}: {err}")
                return false
            | Ok (img, metadata, pixelFloats) ->

            // Validate that image is monochrome (single channel) before debayering
            match validateMonochrome metadata.Channels with
            | Error err ->
                Log.Error($"Cannot debayer '{fileName}': {err}")
                return false
            | Ok () ->

            if img.ColorSpace = XisfColorSpace.RGB then
                // Additional safety check: reject if already RGB color space
                Log.Error($"Cannot debayer '{fileName}': image is already in RGB color space")
                return false
            else

            let bayerPattern = parseBayerPattern img patternOverride

            let width = metadata.Width
            let height = metadata.Height

            let debayerConfig = { Pattern = bayerPattern }

            // Call the debayering algorithm
            match debayerPixels pixelFloats width height debayerConfig with
            | Error err ->
                Log.Error($"Debayering failed for '{fileName}': {err}")
                return false
            | Ok result ->

            // Determine output format (use override or preserve input format)
            let (outputFormat, normalize) =
                match outputFormatOverride with
                | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
                | None -> PixelIO.getRecommendedOutputFormat img.SampleFormat

            let historyEntry = $"Debayered using VNG algorithm (pattern: {bayerPattern})"

            // Exclude ColorFilterArray and Bayer-related FITS keywords
            let bayerKeywords = Set.ofList ["BAYERPAT"; "XBAYROFF"; "YBAYROFF"]

            if split then
                // Split into three separate monochrome files
                let channels = [|
                    ("R", "Red", result.RedChannel)
                    ("G", "Green", result.GreenChannel)
                    ("B", "Blue", result.BlueChannel)
                |]

                let! writeResults =
                    channels
                    |> Array.map (fun (chSuffix, filterName, channelFloats) ->
                        async {
                            let outFileName = $"{baseName}{suffix}_{chSuffix}.xisf"
                            let outPath = Path.Combine(outputDir, outFileName)

                            let channelBytes = PixelIO.writePixelsFromFloat channelFloats outputFormat normalize

                            // Create output config with FILTER keyword and excluding Bayer keywords
                            let outputConfig = {
                                XisfIO.defaultOutputImageConfig with
                                    Dimensions = Some (width, height, 1)
                                    Format = Some outputFormat
                                    HistoryEntries = [historyEntry]
                                    ExcludeFitsKeys = bayerKeywords
                                    AdditionalFits = [|
                                        XisfFitsKeyword("FILTER", "", filterName) :> XisfCoreElement
                                    |]
                            }

                            match XisfIO.createOutputImage img channelBytes outputConfig with
                            | Error err ->
                                Log.Error($"Failed to create output image for {chSuffix}: {err}")
                                return false
                            | Ok monoImage ->

                            match! XisfIO.writeImage outPath "XisfPrep Debayer v1.0" monoImage with
                            | Error err ->
                                Log.Error($"Failed to write {outFileName}: {err}")
                                return false
                            | Ok () ->
                                let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
                                printfn $"  -> {outFileName} ({sizeMB} MB, {width}x{height} Mono)"
                                return true
                        })
                    |> Async.Sequential

                return writeResults |> Array.forall id

            else
                // Standard RGB output - interleave the three channels
                let outFileName = $"{baseName}{suffix}.xisf"
                let outPath = Path.Combine(outputDir, outFileName)

                // Interleave R, G, B channels into single RGB array
                let pixelCount = width * height
                let rgbFloats = Array.zeroCreate (pixelCount * 3)
                for i = 0 to pixelCount - 1 do
                    rgbFloats.[i * 3 + 0] <- result.RedChannel.[i]
                    rgbFloats.[i * 3 + 1] <- result.GreenChannel.[i]
                    rgbFloats.[i * 3 + 2] <- result.BlueChannel.[i]

                let debayered = PixelIO.writePixelsFromFloat rgbFloats outputFormat normalize

                // Create output config excluding Bayer keywords
                let outputConfig = {
                    XisfIO.defaultOutputImageConfig with
                        Dimensions = Some (width, height, 3)
                        Format = Some outputFormat
                        HistoryEntries = [historyEntry]
                        ExcludeFitsKeys = bayerKeywords
                }

                match XisfIO.createOutputImage img debayered outputConfig with
                | Error err ->
                    Log.Error($"Failed to create output image: {err}")
                    return false
                | Ok rgbImage ->

                match! XisfIO.writeImage outPath "XisfPrep Debayer v1.0" rgbImage with
                | Error err ->
                    Log.Error($"Failed to write {outFileName}: {err}")
                    return false
                | Ok () ->

                    let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
                    printfn $"  -> {outFileName} ({sizeMB} MB, {width}x{height} RGB)"

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

                Log.Information($"Debayering images using VNG algorithm")

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
                        let tasks = files |> Array.map (fun f ->
                            debayerImage f opts.Output opts.Pattern opts.Suffix opts.Overwrite opts.OutputFormat opts.Split)
                        let! results = Async.Parallel(tasks, maxDegreeOfParallelism = opts.MaxParallel)

                        let successCount = results |> Array.filter id |> Array.length
                        let failCount = results.Length - successCount

                        printfn ""
                        printfn $"Completed: {successCount} succeeded, {failCount} failed"

                        return if failCount > 0 then 1 else 0
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep debayer --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
