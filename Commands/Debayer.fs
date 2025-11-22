module Commands.Debayer

open System
open System.IO
open Serilog
open XisfLib.Core

// Constants
let VNG_KERNEL_BORDER = 2  // 5x5 kernel requires 2-pixel margin from image edges

// Color channel indices
type Channel = R = 0 | G = 1 | B = 2

// Eight gradient directions for VNG
type Direction = N | NE | E | SE | S | SW | W | NW

let allDirections = [| N; NE; E; SE; S; SW; W; NW |]

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

// Extract bayer pattern from image
let getBayerPattern (img: XisfImage) (patternOverride: string option) =
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

// Bayer pattern positions: RGGB = R at (0,0), G at (0,1) and (1,0), B at (1,1)
let getColor x y pattern : Channel =
    let evenRow = y % 2 = 0
    let evenCol = x % 2 = 0
    match pattern with
    | "RGGB" ->
        if evenRow && evenCol then Channel.R
        elif evenRow && not evenCol then Channel.G
        elif not evenRow && evenCol then Channel.G
        else Channel.B
    | "BGGR" ->
        if evenRow && evenCol then Channel.B
        elif evenRow && not evenCol then Channel.G
        elif not evenRow && evenCol then Channel.G
        else Channel.R
    | "GRBG" ->
        if evenRow && evenCol then Channel.G
        elif evenRow && not evenCol then Channel.R
        elif not evenRow && evenCol then Channel.B
        else Channel.G
    | "GBRG" ->
        if evenRow && evenCol then Channel.G
        elif evenRow && not evenCol then Channel.B
        elif not evenRow && evenCol then Channel.R
        else Channel.G
    | _ -> failwith "Unknown bayer pattern"

// Compute gradient magnitude in a direction using 5x5 neighborhood
let computeGradient (getPixel: int -> int -> float) x y dir =
    let inline diff a b = abs (a - b)
    match dir with
    | N  -> diff (getPixel x (y-1)) (getPixel x (y+1)) + diff (getPixel x (y-2)) (getPixel x y)
    | NE -> diff (getPixel (x+1) (y-1)) (getPixel (x-1) (y+1)) + diff (getPixel (x+2) (y-2)) (getPixel x y)
    | E  -> diff (getPixel (x+1) y) (getPixel (x-1) y) + diff (getPixel (x+2) y) (getPixel x y)
    | SE -> diff (getPixel (x+1) (y+1)) (getPixel (x-1) (y-1)) + diff (getPixel (x+2) (y+2)) (getPixel x y)
    | S  -> diff (getPixel x (y+1)) (getPixel x (y-1)) + diff (getPixel x (y+2)) (getPixel x y)
    | SW -> diff (getPixel (x-1) (y+1)) (getPixel (x+1) (y-1)) + diff (getPixel (x-2) (y+2)) (getPixel x y)
    | W  -> diff (getPixel (x-1) y) (getPixel (x+1) y) + diff (getPixel (x-2) y) (getPixel x y)
    | NW -> diff (getPixel (x-1) (y-1)) (getPixel (x+1) (y+1)) + diff (getPixel (x-2) (y-2)) (getPixel x y)

// Helper to average values, with a fallback to a simple average if no valid directions are found
let average (values: float array) =
    if values.Length > 0 then Array.average values
    else 0.0

let getFallback (fallbackValues: float array) =
    if fallbackValues.Length > 0 then Array.average fallbackValues
    else 0.0


// VNG (Variable Number of Gradients) interpolation for interior pixels
let interpolateVNG (getPixel: int -> int -> float) x y (color: Channel) (value: float) bayerPattern =
    let gradients = allDirections |> Array.map (fun dir -> (dir, computeGradient getPixel x y dir))
    let minGrad = gradients |> Array.map snd |> Array.min
    let maxGrad = gradients |> Array.map snd |> Array.max
    let threshold = minGrad + (maxGrad - minGrad) / 2.0
    let validDirs = gradients |> Array.filter (fun (_, g) -> g <= threshold) |> Array.map fst
    let validDirsSet = Set.ofArray validDirs

    match color with
    | Channel.R ->
        // Red pixel - interpolate G and B
        // G-neighbors are at N, S, E, W
        let gDirs = [| N; E; S; W |]
        let gFallbacks = gDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let gValues = gDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        // B-neighbors are at NE, SE, SW, NW
        let bDirs = [| NE; SE; SW; NW |]
        let bFallbacks = bDirs |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))
        let bValues = bDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))

        let g = if gValues.Length > 0 then average gValues else getFallback gFallbacks
        let b = if bValues.Length > 0 then average bValues else getFallback bFallbacks
        (value, g, b)

    | Channel.B ->
        // Blue pixel - interpolate G and R
        // G-neighbors are at N, S, E, W
        let gDirs = [| N; E; S; W |]
        let gFallbacks = gDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let gValues = gDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        // R-neighbors are at NE, SE, SW, NW
        let rDirs = [| NE; SE; SW; NW |]
        let rFallbacks = rDirs |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))
        let rValues = rDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))

        let g = if gValues.Length > 0 then average gValues else getFallback gFallbacks
        let r = if rValues.Length > 0 then average rValues else getFallback rFallbacks
        (r, g, value)

    | Channel.G ->
        // Green pixel - interpolate R and B
        // Determine which neighbors are R and which are B based on Bayer pattern
        let (rDirs, bDirs) =
            match bayerPattern with
            // For RGGB, G on R-row (y%2=0) has R neighbors E/W, B neighbors N/S
            | "RGGB" -> if y % 2 = 0 then ([| E; W |], [| N; S |]) else ([| N; S |], [| E; W |])
            // For BGGR, G on B-row (y%2=0) has B neighbors E/W, R neighbors N/S
            | "BGGR" -> if y % 2 = 0 then ([| N; S |], [| E; W |]) else ([| E; W |], [| N; S |])
            // For GRBG, G on G-row (y%2=0) has R neighbors N/S, B neighbors E/W
            | "GRBG" -> if y % 2 = 0 then ([| N; S |], [| E; W |]) else ([| E; W |], [| N; S |])
            // For GBRG, G on G-row (y%2=0) has B neighbors N/S, R neighbors E/W
            | "GBRG" -> if y % 2 = 0 then ([| E; W |], [| N; S |]) else ([| N; S |], [| E; W |])
            | _ -> failwith "Unknown bayer pattern"

        let rFallbacks = rDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let rValues = rDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        let bFallbacks = bDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let bValues = bDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        let r = if rValues.Length > 0 then average rValues else getFallback rFallbacks
        let b = if bValues.Length > 0 then average bValues else getFallback bFallbacks
        (r, value, b)

    | _ -> failwith "Invalid channel"

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

            let reader = new XisfReader()
            let! unit = reader.ReadAsync(inputPath) |> Async.AwaitTask
            let img = unit.Images.[0]

            // Validate that image is monochrome (single channel) before debayering
            if img.Geometry.ChannelCount <> 1u then
                Log.Error($"Cannot debayer '{fileName}': image has {img.Geometry.ChannelCount} channels (expected 1 for monochrome Bayer data)")
                return false
            elif img.ColorSpace = XisfColorSpace.RGB then
                // Additional safety check: reject if already RGB color space
                Log.Error($"Cannot debayer '{fileName}': image is already in RGB color space")
                return false
            else

            let bayerPattern = getBayerPattern img patternOverride

            // Read pixels using PixelIO (handles all sample formats)
            let pixelFloats = PixelIO.readPixelsAsFloat img

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height

            let getPixel x y =
                // Clamp coordinates to image bounds (replicate edge pixels)
                let x' = max 0 (min (width - 1) x)
                let y' = max 0 (min (height - 1) y)
                pixelFloats.[y' * width + x']

            // Output is 3 channels (RGB)
            let debayeredFloats = Array.zeroCreate (width * height * 3)

            // Pass 1: VNG interpolation for interior pixels
            for y = VNG_KERNEL_BORDER to height - VNG_KERNEL_BORDER - 1 do
                for x = VNG_KERNEL_BORDER to width - VNG_KERNEL_BORDER - 1 do
                    let outIdx = (y * width + x) * 3
                    let color = getColor x y bayerPattern
                    let value = getPixel x y
                    let r, g, b = interpolateVNG getPixel x y color value bayerPattern
                    debayeredFloats.[outIdx + 0] <- r
                    debayeredFloats.[outIdx + 1] <- g
                    debayeredFloats.[outIdx + 2] <- b

            // Pass 2: Edge replication for borders (copy nearest interior VNG result)
            for y = 0 to height - 1 do
                for x = 0 to width - 1 do
                    let isBorder = x < VNG_KERNEL_BORDER || x >= width - VNG_KERNEL_BORDER ||
                                   y < VNG_KERNEL_BORDER || y >= height - VNG_KERNEL_BORDER
                    if isBorder then
                        let outIdx = (y * width + x) * 3
                        let srcX = max VNG_KERNEL_BORDER (min (width - VNG_KERNEL_BORDER - 1) x)
                        let srcY = max VNG_KERNEL_BORDER (min (height - VNG_KERNEL_BORDER - 1) y)
                        let srcIdx = (srcY * width + srcX) * 3
                        debayeredFloats.[outIdx + 0] <- debayeredFloats.[srcIdx + 0]
                        debayeredFloats.[outIdx + 1] <- debayeredFloats.[srcIdx + 1]
                        debayeredFloats.[outIdx + 2] <- debayeredFloats.[srcIdx + 2]

            // Determine output format (use override or preserve input format)
            let (outputFormat, normalize) =
                match outputFormatOverride with
                | Some fmt -> PixelIO.getRecommendedOutputFormat fmt
                | None -> PixelIO.getRecommendedOutputFormat img.SampleFormat

            // Filter out ColorFilterArray and Bayer-related FITS keywords - debayered images are no longer mosaiced
            let bayerKeywords = Set.ofList ["BAYERPAT"; "XBAYROFF"; "YBAYROFF"]

            let historyEntry = $"Debayered using VNG algorithm (pattern: {bayerPattern})"

            let filterElements (additionalKeywords: XisfFitsKeyword array) =
                if isNull img.AssociatedElements then
                    additionalKeywords |> Array.map (fun k -> k :> XisfCoreElement)
                else
                    let filtered =
                        img.AssociatedElements
                        |> Seq.toArray
                        |> Array.filter (fun e ->
                            match e with
                            | :? XisfColorFilterArray -> false
                            | :? XisfFitsKeyword as fits -> not (bayerKeywords.Contains fits.Name)
                            | _ -> true)
                    Array.append filtered (additionalKeywords |> Array.map (fun k -> k :> XisfCoreElement))

            // Get bounds per XISF spec: Some for Float32/Float64, None for integer formats
            let bounds =
                match PixelIO.getBoundsForFormat outputFormat with
                | Some b -> b
                | None -> Unchecked.defaultof<XisfImageBounds>  // null for integer formats

            if split then
                // Split into three separate monochrome files
                let channelNames = [| ("R", "Red"); ("G", "Green"); ("B", "Blue") |]
                let pixelCount = width * height

                for i in 0 .. 2 do
                    let (chSuffix, filterName) = channelNames.[i]
                    let outFileName = $"{baseName}{suffix}_{chSuffix}.xisf"
                    let outPath = Path.Combine(outputDir, outFileName)

                    // Extract single channel from interleaved RGB data
                    let channelFloats = Array.zeroCreate pixelCount
                    for p in 0 .. pixelCount - 1 do
                        channelFloats.[p] <- debayeredFloats.[p * 3 + i]

                    let channelBytes = PixelIO.writePixelsFromFloat channelFloats outputFormat normalize

                    let monoGeometry = XisfImageGeometry([| uint32 width; uint32 height |], 1u)
                    let dataBlock = InlineDataBlock(ReadOnlyMemory(channelBytes), XisfEncoding.Base64)

                    // Add FILTER keyword and HISTORY
                    let additionalKeywords = [|
                        XisfFitsKeyword("FILTER", "", filterName)
                        XisfFitsKeyword("HISTORY", "", historyEntry)
                    |]
                    let elements = filterElements additionalKeywords
                    let elementsList =
                        if elements.Length = 0 then null
                        else elements :> System.Collections.Generic.IReadOnlyList<_>

                    let monoImage = XisfImage(
                        monoGeometry,
                        outputFormat,
                        XisfColorSpace.Gray,
                        dataBlock,
                        bounds,
                        XisfPixelStorage.Normal,
                        img.ImageType,
                        img.Offset,
                        img.Orientation,
                        img.ImageId,
                        img.Uuid,
                        img.Properties,
                        elementsList
                    )

                    let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Debayer v1.0")
                    let outUnit = XisfFactory.CreateMonolithic(metadata, monoImage)

                    let writer = new XisfWriter()
                    do! writer.WriteAsync(outUnit, outPath) |> Async.AwaitTask

                    let sizeMB = (FileInfo outPath).Length / 1024L / 1024L
                    printfn $"  -> {outFileName} ({sizeMB} MB, {width}x{height} Mono)"

            else
                // Standard RGB output
                let outFileName = $"{baseName}{suffix}.xisf"
                let outPath = Path.Combine(outputDir, outFileName)

                let debayered = PixelIO.writePixelsFromFloat debayeredFloats outputFormat normalize

                let rgbGeometry = XisfImageGeometry([| uint32 width; uint32 height |], 3u)
                let dataBlock = InlineDataBlock(ReadOnlyMemory(debayered), XisfEncoding.Base64)

                let additionalKeywords = [| XisfFitsKeyword("HISTORY", "", historyEntry) |]
                let elements = filterElements additionalKeywords
                let filteredElements =
                    if elements.Length = 0 then null
                    else elements :> System.Collections.Generic.IReadOnlyList<_>

                let rgbImage = XisfImage(
                    rgbGeometry,
                    outputFormat,
                    XisfColorSpace.RGB,
                    dataBlock,
                    bounds,
                    XisfPixelStorage.Normal,
                    img.ImageType,
                    img.Offset,
                    img.Orientation,
                    img.ImageId,
                    img.Uuid,
                    img.Properties,
                    filteredElements
                )

                let metadata = XisfFactory.CreateMinimalMetadata("XisfPrep Debayer v1.0")
                let outUnit = XisfFactory.CreateMonolithic(metadata, rgbImage)

                let writer = new XisfWriter()
                do! writer.WriteAsync(outUnit, outPath) |> Async.AwaitTask

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
