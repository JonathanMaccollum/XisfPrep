module Commands.Convert

open Serilog

let showHelp() =
    printfn "convert - Format conversion and export"
    printfn ""
    printfn "Usage: xisfprep convert [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <file>        Input XISF file"
    printfn "  --output, -o <file>       Output file path"
    printfn ""
    printfn "Optional:"
    printfn "  --format <format>         Output format (default: inferred from extension)"
    printfn "                              xisf - XISF format"
    printfn "                              fits - FITS format"
    printfn "                              tiff - 16-bit TIFF"
    printfn "  --compression <codec>     Compression codec (default: lz4hcsh for XISF)"
    printfn "                              none     - Uncompressed"
    printfn "                              lz4      - LZ4 fast compression"
    printfn "                              lz4hc    - LZ4 high compression"
    printfn "                              lz4sh    - LZ4 with byte shuffling"
    printfn "                              lz4hcsh  - LZ4HC with byte shuffling"
    printfn "                              zlib     - Zlib compression"
    printfn "                              zlibsh   - Zlib with byte shuffling"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep convert -i result.xisf -o result.fits"
    printfn "  xisfprep convert -i result.xisf -o result_compressed.xisf --compression lz4hcsh"

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("convert command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep convert --help' for usage information"
        1
