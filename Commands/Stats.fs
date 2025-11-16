module Commands.Stats

open Serilog

let showHelp() =
    printfn "stats - Calculate and display image statistics"
    printfn ""
    printfn "Usage: xisfprep stats [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files (wildcards supported)"
    printfn ""
    printfn "Optional:"
    printfn "  --output, -o <file>       Output CSV file path"
    printfn "  --metrics <level>         Statistics to calculate (default: basic)"
    printfn "                              basic     - Mean, median, stddev, min, max"
    printfn "                              all       - Basic + MAD, SNR, star count estimates"
    printfn "                              histogram - Include histogram data"
    printfn ""
    printfn "Output:"
    printfn "  Console table with statistics per file"
    printfn "  Optional CSV export for analysis"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep stats -i \"images/*.xisf\""
    printfn "  xisfprep stats -i \"images/*.xisf\" -o stats.csv --metrics all"

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("stats command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep stats --help' for usage information"
        1
