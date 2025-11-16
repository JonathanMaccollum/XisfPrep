module Commands.Debayer

open Serilog

let showHelp() =
    printfn "debayer - Convert Bayer mosaic to RGB"
    printfn ""
    printfn "Usage: xisfprep debayer [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input Bayer mosaic files (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for RGB files"
    printfn ""
    printfn "Optional:"
    printfn "  --pattern, -p <pattern>   Bayer pattern override (RGGB, BGGR, GRBG, GBRG)"
    printfn "  --algorithm <name>        Interpolation algorithm (default: vng)"
    printfn "                              vng      - Variable Number of Gradients (high quality)"
    printfn "                              bilinear - Simple bilinear (fast, lower quality)"
    printfn "  --skip-existing           Skip files already debayered in output directory"
    printfn "  --suffix <text>           Output filename suffix (default: _d)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep debayer -i \"mono/*.xisf\" -o \"rgb/\""
    printfn "  xisfprep debayer -i \"mono/*.xisf\" -o \"rgb/\" --pattern RGGB --algorithm vng"

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("debayer command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep debayer --help' for usage information"
        1
