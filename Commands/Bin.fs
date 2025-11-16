module Commands.Bin

open System
open System.IO
open Serilog
open XisfLib.Core

type BinningMethod =
    | Average
    | Median
    | Sum

    member this.Description =
        match this with
        | Average -> "Average binning (preserves flux)"
        | Median -> "Median binning (robust to outliers)"
        | Sum -> "Sum binning (adds pixel values)"

// --- Defaults ---
let private defaultFactor = 2
let private defaultMethod = Average
let private defaultSuffix factor = $"_bin{factor}x"
let private defaultParallel = System.Environment.ProcessorCount
// ---

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

let parseArgs (args: string array) =
    let rec parse (args: string list) input output factor method suffix overwrite maxParallel =
        match args with
        | [] -> (input, output, factor, method, suffix, overwrite, maxParallel)
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest (Some value) output factor method suffix overwrite maxParallel
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest input (Some value) factor method suffix overwrite maxParallel
        | "--factor" :: value :: rest ->
            parse rest input output (Some (int value)) method suffix overwrite maxParallel
        | "--method" :: value :: rest ->
            let m = match value.ToLower() with
                    | "average" -> Average
                    | "median" -> Median
                    | "sum" -> Sum
                    | _ -> failwithf "Unknown binning method: %s" value
            parse rest input output factor (Some m) suffix overwrite maxParallel
        | "--suffix" :: value :: rest ->
            parse rest input output factor method (Some value) overwrite maxParallel
        | "--overwrite" :: rest ->
            parse rest input output factor method suffix true maxParallel
        | "--parallel" :: value :: rest ->
            parse rest input output factor method suffix overwrite (Some (int value))
        | arg :: rest ->
            failwithf "Unknown argument: %s" arg

    let (input, output, factor, method, suffix, overwrite, maxParallel) = parse (List.ofArray args) None None None None None false None

    let input = match input with Some v -> v | None -> failwith "Required argument: --input"
    let output = match output with Some v -> v | None -> failwith "Required argument: --output"
    let factor = factor |> Option.defaultValue defaultFactor
    let method = method |> Option.defaultValue defaultMethod
    let suffix = suffix |> Option.defaultValue (defaultSuffix factor)
    let maxParallel = maxParallel |> Option.defaultValue defaultParallel

    if factor < 2 || factor > 6 then
        failwith "Binning factor must be between 2 and 6"

    if maxParallel < 1 then
        failwith "Parallel count must be at least 1"

    (input, output, factor, method, suffix, overwrite, maxParallel)

let getPixelData (img: XisfImage) =
    let data = img.PixelData
    if data :? InlineDataBlock then
        (data :?> InlineDataBlock).Data.ToArray()
    else
        failwith "Expected inline data"

let applyBinningMethod (pixels: uint16 array) (method: BinningMethod) =
    if Array.isEmpty pixels then 0us
    else
        match method with
        | Average ->
            let sum = pixels |> Array.sumBy uint32
            uint16 (sum / uint32 pixels.Length)
        | Median ->
            let sorted = pixels |> Array.sort
            let mid = sorted.Length / 2
            if sorted.Length % 2 = 0 then
                (uint32 sorted.[mid - 1] + uint32 sorted.[mid]) / 2u |> uint16
            else
                sorted.[mid]
        | Sum ->
            pixels |> Array.sumBy uint32 |> min 65535u |> uint16

let createBinnedData (pixelData: byte[]) width height channels factor (method: BinningMethod) =
    let bytesPerPixel = channels * 2
    let newWidth = width / factor
    let newHeight = height / factor

    let getPixel x y channel =
        let offset = (y * width + x) * bytesPerPixel + channel * 2
        uint16 pixelData.[offset] ||| (uint16 pixelData.[offset + 1] <<< 8)

    let binnedData = Array.zeroCreate (newWidth * newHeight * bytesPerPixel)

    // Pre-allocate one array for the binning block and reuse it
    let blockPixels = Array.zeroCreate (factor * factor)

    for newY = 0 to newHeight - 1 do
        for newX = 0 to newWidth - 1 do
            for ch = 0 to channels - 1 do
                let mutable pixelCount = 0
                for dy in 0 .. factor - 1 do
                    for dx in 0 .. factor - 1 do
                        let srcX = newX * factor + dx
                        let srcY = newY * factor + dy
                        if srcX < width && srcY < height then
                            blockPixels.[pixelCount] <- getPixel srcX srcY ch
                            pixelCount <- pixelCount + 1

                let binnedValue =
                    if pixelCount = 0 then 0us
                    else
                        // Pass a slice of the array that contains valid pixels
                        applyBinningMethod blockPixels.[0 .. pixelCount - 1] method

                let offset = (newY * newWidth + newX) * bytesPerPixel + ch * 2
                binnedData.[offset] <- byte binnedValue
                binnedData.[offset + 1] <- byte (binnedValue >>> 8)

    binnedData

let validateDimensions width height factor =
    if width % factor <> 0 || height % factor <> 0 then
        Log.Warning($"Image dimensions ({width}x{height}) not divisible by factor {factor} - will truncate")

let createOutputImage (img: XisfImage) (binnedData: byte[]) newWidth newHeight channels factor (method: BinningMethod) =
    let newGeometry = XisfImageGeometry([| uint32 newWidth; uint32 newHeight |], uint32 channels)
    let dataBlock = InlineDataBlock(ReadOnlyMemory(binnedData), XisfEncoding.Base64)

    let historyEntry = $"Binned {factor}x{factor} using {method} method"
    let updatedFits =
        if isNull img.AssociatedElements then
            [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]
        else
            let existingFits = img.AssociatedElements :> seq<_> |> Seq.toArray
            Array.append existingFits [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]

    XisfImage(
        newGeometry,
        img.SampleFormat,
        img.ColorSpace,
        dataBlock,
        img.Bounds,
        img.PixelStorage,
        img.ImageType,
        img.Offset,
        img.Orientation,
        img.ImageId,
        img.Uuid,
        img.Properties,
        updatedFits
    )

let binImage (inputPath: string) (outputDir: string) (factor: int) (method: BinningMethod) (suffix: string) (overwrite: bool) : Async<bool> =
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

            validateDimensions width height factor

            let newWidth = width / factor
            let newHeight = height / factor

            let pixelData = getPixelData img
            let binnedData = createBinnedData pixelData width height channels factor method

            let binnedImage = createOutputImage img binnedData newWidth newHeight channels factor method

            let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Bin v1.0")
            let outUnit = XisfFactory.CreateMonolithic(metadata, binnedImage)

            let writer = new XisfWriter()
            do! writer.WriteAsync(outUnit, outPath) |> Async.AwaitTask

            let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
            printfn $"  -> {outFileName} ({sizeMB} MB, {newWidth}x{newHeight})"

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
                let (inputPattern, outputDir, factor, method, suffix, overwrite, maxParallel) = parseArgs args

                Log.Information($"Binning images with factor {factor} using {method} method")

                if not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                    Log.Information($"Created output directory: {outputDir}")

                let inputDir = Path.GetDirectoryName(inputPattern)
                let pattern = Path.GetFileName(inputPattern)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern)

                    if files.Length = 0 then
                        Log.Error($"No files found matching pattern: {inputPattern}")
                        return 1
                    else
                        printfn $"Found {files.Length} files to process"
                        printfn ""

                        // Process files in chunks based on parallel count
                        let chunks = files |> Array.chunkBySize maxParallel
                        let mutable allResults = []

                        for chunk in chunks do
                            let tasks = chunk |> Array.map (fun f -> binImage f outputDir factor method suffix overwrite)
                            let! results = Async.Parallel tasks
                            allResults <- allResults @ (Array.toList results)

                        let successCount = allResults |> List.filter id |> List.length
                        let failCount = allResults.Length - successCount

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
