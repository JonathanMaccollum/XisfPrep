open Serilog
open Serilog.Events
open System

let configureLogging (verbose: bool) (quiet: bool) =
    let level =
        if quiet then LogEventLevel.Error
        elif verbose then LogEventLevel.Verbose
        else LogEventLevel.Information

    Log.Logger <-
        LoggerConfiguration()
            .MinimumLevel.Is(level)
            .WriteTo.Console()
            .CreateLogger()

let showUsage() =
    printfn "xisfprep - XISF Preprocessing CLI"
    printfn ""
    printfn "Usage: xisfprep <command> [options]"
    printfn ""
    printfn "Commands:"
    printfn "  calibrate   Apply bias/dark/flat calibration frames"
    printfn "  debayer     Convert Bayer mosaic to RGB"
    printfn "  headers     Extract FITS keywords and HISTORY values"
    printfn "  align       Register images to reference"
    printfn "  integrate   Stack/combine multiple images"
    printfn "  stats       Calculate image statistics"
    printfn "  stars       Detect stars and generate visualization"
    printfn "  convert     Format conversion and export"
    printfn "  bin         Downsample images by binning pixels"
    printfn "  inspect     Diagnostic inspection of XISF file structure"
    printfn ""
    printfn "Global options:"
    printfn "  --verbose, -v    Enable verbose diagnostic output"
    printfn "  --quiet, -q      Suppress all output except errors"
    printfn "  --help, -h       Show help information"
    printfn ""
    printfn "Run 'xisfprep <command> --help' for command-specific help"

let commandHandlers =
    Map.ofList [
        "calibrate", Commands.Calibrate.run
        "debayer", Commands.Debayer.run
        "headers", Commands.Headers.run
        "align", Commands.Align.run
        "integrate", Commands.Integrate.run
        "stats", Commands.Stats.run
        "stars", Commands.Stars.run
        "convert", Commands.Convert.run
        "bin", Commands.Bin.run
        "inspect", Commands.Inspect.run
    ]

[<EntryPoint>]
let main argv =
    // Parse for global flags
    let verbose = argv |> Array.contains "--verbose" || argv |> Array.contains "-v"
    let quiet = argv |> Array.contains "--quiet" || argv |> Array.contains "-q"
    let help = argv |> Array.contains "--help" || argv |> Array.contains "-h"

    configureLogging verbose quiet

    Log.Verbose("Verbose logging enabled")
    Log.Verbose("Command line args: {Args}", argv)

    // Get command (first non-flag argument)
    let command =
        argv
        |> Array.tryFind (fun arg -> not (arg.StartsWith("-")))

    // Filter out command name and global flags to get command-specific args
    let getCommandArgs cmdName =
        let globalFlags = Set.ofList ["--verbose"; "-v"; "--quiet"; "-q"]
        argv
        |> Array.filter (fun arg -> arg <> cmdName && not (globalFlags.Contains arg))

    match command with
    | None ->
        showUsage()
        0
    | Some cmd ->
        match Map.tryFind cmd commandHandlers with
        | Some handler ->
            handler (getCommandArgs cmd)
        | None ->
            Log.Error("Unknown command: {Command}", cmd)
            showUsage()
            1
