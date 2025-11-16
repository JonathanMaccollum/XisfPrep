module Commands.Calibrate

open Serilog

let showHelp() =
    printfn "calibrate - Apply bias/dark/flat calibration frames"
    printfn ""
    printfn "Usage: xisfprep calibrate [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input light frames (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for calibrated frames"
    printfn ""
    printfn "Optional:"
    printfn "  --bias, -b <file>         Master bias frame"
    printfn "  --dark, -d <file>         Master dark frame"
    printfn "  --flat, -f <file>         Master flat frame"
    printfn "  --optimize-darks          Scale dark frame by exposure time"
    printfn "  --pedestal <value>        Output pedestal value [0-65535] (default: 0)"
    printfn "  --suffix <text>           Output filename suffix (default: _cal)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep calibrate -i \"lights/*.xisf\" -o \"calibrated/\" -b bias.xisf -d dark.xisf -f flat.xisf"
    printfn "  xisfprep calibrate -i \"lights/*.xisf\" -o \"calibrated/\" --pedestal 100"

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("calibrate command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep calibrate --help' for usage information"
        1
