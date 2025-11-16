open Serilog
open Serilog.Events
open System

let configureLogging (verbose: bool) =
    let level = if verbose then LogEventLevel.Verbose else LogEventLevel.Information

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
    printfn "  align       Register images to reference"
    printfn "  integrate   Stack/combine multiple images"
    printfn "  stats       Calculate image statistics"
    printfn "  convert     Format conversion and export"
    printfn ""
    printfn "Global options:"
    printfn "  --verbose, -v    Enable verbose diagnostic output"
    printfn "  --help, -h       Show help information"
    printfn ""
    printfn "Run 'xisfprep <command> --help' for command-specific help"

[<EntryPoint>]
let main argv =
    // Parse for verbose flag
    let verbose = argv |> Array.contains "--verbose" || argv |> Array.contains "-v"
    let help = argv |> Array.contains "--help" || argv |> Array.contains "-h"

    configureLogging verbose

    Log.Verbose("Verbose logging enabled")
    Log.Verbose("Command line args: {Args}", argv)

    // Get command (first non-flag argument)
    let command =
        argv
        |> Array.tryFind (fun arg -> not (arg.StartsWith("-")))

    match command with
    | None ->
        showUsage()
        0
    | Some "calibrate" ->
        Commands.Calibrate.run argv
    | Some "debayer" ->
        Commands.Debayer.run argv
    | Some "align" ->
        Commands.Align.run argv
    | Some "integrate" ->
        Commands.Integrate.run argv
    | Some "stats" ->
        Commands.Stats.run argv
    | Some "convert" ->
        Commands.Convert.run argv
    | Some cmd ->
        Log.Error("Unknown command: {Command}", cmd)
        showUsage()
        1
