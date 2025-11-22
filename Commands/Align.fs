module Commands.Align

open System
open System.IO
open Serilog
open XisfLib.Core

type InterpolationMethod =
    | Nearest
    | Bilinear
    | Bicubic

// --- Defaults ---
let private defaultMaxShift = 100
let private defaultInterpolation = Bicubic
let private defaultSuffix = "_a"
let private defaultParallel = Environment.ProcessorCount
// ---

type AlignOptions = {
    Input: string
    Output: string
    Reference: string option
    AutoReference: bool
    MaxShift: int
    Interpolation: InterpolationMethod
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
}

let showHelp() =
    printfn "align - Register images to reference frame"
    printfn ""
    printfn "Usage: xisfprep align [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for aligned files"
    printfn ""
    printfn "Optional:"
    printfn "  --reference, -r <file>    Reference frame to align to (first file if omitted)"
    printfn "  --auto-reference          Auto-select best reference (highest star count/SNR)"
    printfn $"  --max-shift <pixels>      Maximum pixel shift allowed (default: {defaultMaxShift})"
    printfn "  --interpolation <method>  Resampling method (default: bicubic)"
    printfn "                              nearest  - Nearest neighbor (preserves values)"
    printfn "                              bilinear - Bilinear (smooth, fast)"
    printfn "                              bicubic  - Bicubic (smooth, higher quality)"
    printfn $"  --suffix <text>           Output filename suffix (default: {defaultSuffix})"
    printfn "  --overwrite               Overwrite existing output files (default: skip existing)"
    printfn $"  --parallel <n>            Number of parallel operations (default: {defaultParallel} CPU cores)"
    printfn "  --output-format <format>  Output sample format (default: preserve input format)"
    printfn "                              uint8, uint16, uint32, float32, float64"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" --auto-reference"
    printfn "  xisfprep align -i \"images/*.xisf\" -o \"aligned/\" -r \"best.xisf\" --max-shift 50"

let parseArgs (args: string array) : AlignOptions =
    let rec parse (args: string list) (opts: AlignOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = value }
        | "--reference" :: value :: rest | "-r" :: value :: rest ->
            parse rest { opts with Reference = Some value }
        | "--auto-reference" :: rest ->
            parse rest { opts with AutoReference = true }
        | "--max-shift" :: value :: rest ->
            parse rest { opts with MaxShift = int value }
        | "--interpolation" :: value :: rest ->
            let interp = match value.ToLower() with
                         | "nearest" -> Nearest
                         | "bilinear" -> Bilinear
                         | "bicubic" -> Bicubic
                         | _ -> failwithf "Unknown interpolation method: %s (supported: nearest, bilinear, bicubic)" value
            parse rest { opts with Interpolation = interp }
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
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = ""
        Reference = None
        AutoReference = false
        MaxShift = defaultMaxShift
        Interpolation = defaultInterpolation
        Suffix = defaultSuffix
        Overwrite = false
        MaxParallel = defaultParallel
        OutputFormat = None
    }

    let opts = parse (List.ofArray args) defaults

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if String.IsNullOrEmpty opts.Output then failwith "Required argument: --output"

    if opts.Reference.IsSome && opts.AutoReference then
        failwith "--reference and --auto-reference are mutually exclusive"

    if opts.MaxShift < 1 then failwith "Max shift must be at least 1 pixel"
    if opts.MaxParallel < 1 then failwith "Parallel count must be at least 1"

    opts

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("align command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep align --help' for usage information"
        1
