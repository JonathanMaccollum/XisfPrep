module Commands.Debayer

open System
open System.IO
open Serilog
open XisfLib.Core

// Constants
let VNG_KERNEL_BORDER = 2  // 5x5 kernel requires 2-pixel margin from image edges
let BYTES_PER_UINT16 = 2   // uint16 values are stored as 2 bytes

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
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn ""
    printfn "Algorithm:"
    printfn "  VNG (Variable Number of Gradients) - High quality gradient-based interpolation"
    printfn "  Uses 5x5 kernel for interior pixels, bilinear for borders"
    printfn ""
    printfn "Process:"
    printfn "  1. Validates input is single-channel monochrome"
    printfn "  2. Reads Bayer pattern from FITS BAYERPAT keyword or uses override"
    printfn "  3. Applies VNG interpolation"
    printfn "  4. Outputs 3-channel RGB image"
    printfn "  5. Removes ColorFilterArray and Bayer FITS keywords from output"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep debayer -i \"lights/*.xisf\" -o \"rgb/\""
    printfn "  xisfprep debayer -i \"lights/*.xisf\" -o \"rgb/\" --pattern RGGB --overwrite"

let parseArgs (args: string array) =
    let rec parse (args: string list) input output pattern suffix overwrite maxParallel =
        match args with
        | [] -> (input, output, pattern, suffix, overwrite, maxParallel)
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest (Some value) output pattern suffix overwrite maxParallel
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest input (Some value) pattern suffix overwrite maxParallel
        | "--pattern" :: value :: rest | "-p" :: value :: rest ->
            let p = value.ToUpper()
            if not (p = "RGGB" || p = "BGGR" || p = "GRBG" || p = "GBRG") then
                failwithf "Unknown Bayer pattern: %s (supported: RGGB, BGGR, GRBG, GBRG)" value
            parse rest input output (Some p) suffix overwrite maxParallel
        | "--suffix" :: value :: rest ->
            parse rest input output pattern (Some value) overwrite maxParallel
        | "--overwrite" :: rest ->
            parse rest input output pattern suffix true maxParallel
        | "--parallel" :: value :: rest ->
            parse rest input output pattern suffix overwrite (Some (int value))
        | arg :: rest ->
            failwithf "Unknown argument: %s" arg

    let (input, output, pattern, suffix, overwrite, maxParallel) = parse (List.ofArray args) None None None None false None

    let input = match input with Some v -> v | None -> failwith "Required argument: --input"
    let output = match output with Some v -> v | None -> failwith "Required argument: --output"
    let pattern = pattern  // Option<string> - None means auto-detect
    let suffix = suffix |> Option.defaultValue defaultSuffix
    let maxParallel = maxParallel |> Option.defaultValue defaultParallel

    if maxParallel < 1 then
        failwith "Parallel count must be at least 1"

    (input, output, pattern, suffix, overwrite, maxParallel)

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
let computeGradient (getPixel: int -> int -> uint16) x y dir =
    let inline diff a b = abs (int a - int b)
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
let average (values: uint16 array) =
    if values.Length > 0 then (values |> Array.map uint32 |> Array.sum) / (uint32 values.Length) |> uint16
    else 0us

let getFallback (fallbackValues: uint16 array) =
    if fallbackValues.Length > 0 then (fallbackValues |> Array.map uint32 |> Array.sum) / (uint32 fallbackValues.Length) |> uint16
    else 0us

// Simple bilinear interpolation for border pixels (where 5x5 VNG kernel doesn't fit)
let interpolateBorder (getPixel: int -> int -> uint16) (color: Channel) (value: uint16) =
    match color with
    | Channel.R ->
        let g = (getPixel -1 0 + getPixel 1 0 + getPixel 0 -1 + getPixel 0 1) / 4us
        let b = (getPixel -1 -1 + getPixel 1 -1 + getPixel -1 1 + getPixel 1 1) / 4us
        (value, g, b)
    | Channel.G ->
        let r = (getPixel -1 0 + getPixel 1 0 + getPixel 0 -1 + getPixel 0 1) / 4us
        let b = (getPixel -1 0 + getPixel 1 0 + getPixel 0 -1 + getPixel 0 1) / 4us
        (r, value, b)
    | Channel.B ->
        let g = (getPixel -1 0 + getPixel 1 0 + getPixel 0 -1 + getPixel 0 1) / 4us
        let r = (getPixel -1 -1 + getPixel 1 -1 + getPixel -1 1 + getPixel 1 1) / 4us
        (r, g, value)
    | _ -> failwith "Invalid channel"

// VNG (Variable Number of Gradients) interpolation for interior pixels
let interpolateVNG (getPixel: int -> int -> uint16) x y (color: Channel) (value: uint16) bayerPattern =
    let gradients = allDirections |> Array.map (fun dir -> (dir, computeGradient getPixel x y dir))
    let minGrad = gradients |> Array.map snd |> Array.min
    let maxGrad = gradients |> Array.map snd |> Array.max
    let threshold = minGrad + (maxGrad - minGrad) / 2
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

let debayerImage (inputPath: string) (outputDir: string) (patternOverride: string option) (suffix: string) (overwrite: bool) : Async<bool> =
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
            let pixelData =
                if img.PixelData :? InlineDataBlock then
                    (img.PixelData :?> InlineDataBlock).Data.ToArray()
                else
                    failwith "Expected inline data"

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height

            let getPixel x y =
                // Clamp coordinates to image bounds (replicate edge pixels)
                let x' = max 0 (min (width - 1) x)
                let y' = max 0 (min (height - 1) y)
                let offset = (y' * width + x') * BYTES_PER_UINT16
                uint16 pixelData.[offset] ||| (uint16 pixelData.[offset + 1] <<< 8)

            let debayered = Array.zeroCreate (width * height * 3 * 2)

            for y = 0 to height - 1 do
                for x = 0 to width - 1 do
                    let outIdx = (y * width + x) * 3 * BYTES_PER_UINT16
                    let color = getColor x y bayerPattern
                    let value = getPixel x y

                    let r, g, b =
                        // Check if we're in the border region where 5x5 VNG kernel doesn't fit
                        if x < VNG_KERNEL_BORDER || x >= width - VNG_KERNEL_BORDER ||
                           y < VNG_KERNEL_BORDER || y >= height - VNG_KERNEL_BORDER then
                            // Border: use simple bilinear interpolation
                            let getPixelRelative dx dy = getPixel (x + dx) (y + dy)
                            interpolateBorder getPixelRelative color value
                        else
                            // Interior: use VNG with gradient-based interpolation
                            interpolateVNG getPixel x y color value bayerPattern

                    debayered.[outIdx + 0] <- byte r
                    debayered.[outIdx + 1] <- byte (r >>> 8)
                    debayered.[outIdx + 2] <- byte g
                    debayered.[outIdx + 3] <- byte (g >>> 8)
                    debayered.[outIdx + 4] <- byte b
                    debayered.[outIdx + 5] <- byte (b >>> 8)

            let rgbGeometry = XisfImageGeometry([| uint32 width; uint32 height |], 3u)
            let dataBlock = InlineDataBlock(ReadOnlyMemory(debayered), XisfEncoding.Base64)

            // Filter out ColorFilterArray and Bayer-related FITS keywords - debayered images are no longer mosaiced
            let bayerKeywords = Set.ofList ["BAYERPAT"; "XBAYROFF"; "YBAYROFF"]

            let historyEntry = $"Debayered using VNG algorithm (pattern: {bayerPattern})"
            let filteredAndUpdatedElements =
                if isNull img.AssociatedElements then
                    [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]
                else
                    let filtered =
                        img.AssociatedElements
                        |> Seq.toArray
                        |> Array.filter (fun e ->
                            match e with
                            | :? XisfColorFilterArray -> false
                            | :? XisfFitsKeyword as fits -> not (bayerKeywords.Contains fits.Name)
                            | _ -> true)
                    Array.append filtered [| XisfFitsKeyword("HISTORY", "", historyEntry) :> XisfCoreElement |]

            let filteredElements =
                if filteredAndUpdatedElements.Length = 0 then null
                else filteredAndUpdatedElements :> System.Collections.Generic.IReadOnlyList<_>

            let rgbImage = XisfImage(
                rgbGeometry,
                img.SampleFormat,
                XisfColorSpace.RGB,
                dataBlock,
                img.Bounds,
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
                let (inputPattern, outputDir, patternOverride, suffix, overwrite, maxParallel) = parseArgs args

                Log.Information($"Debayering images using VNG algorithm")

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

                        // Process all files in parallel with max parallelism limit
                        let tasks = files |> Array.map (fun f -> debayerImage f outputDir patternOverride suffix overwrite)
                        let! results = Async.Parallel(tasks, maxDegreeOfParallelism = maxParallel)

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
