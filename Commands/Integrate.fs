module Commands.Integrate

open Serilog

let showHelp() =
    printfn "integrate - Stack/combine multiple images"
    printfn ""
    printfn "Usage: xisfprep integrate [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input image files to stack (wildcards supported)"
    printfn "  --output, -o <file>       Output integrated file path"
    printfn ""
    printfn "Optional:"
    printfn "  --combination <method>    Pixel combination method (default: average)"
    printfn "                              average - Mean (default for lights)"
    printfn "                              median  - Median (robust but slower)"
    printfn "  --normalization <method>  Image normalization (default: multiplicative)"
    printfn "                              none                - No normalization"
    printfn "                              additive            - Additive: P' = P + (K - m_i)"
    printfn "                              multiplicative      - Multiplicative: P' = P * (K / m_i)"
    printfn "                              additive-scaling    - Additive with scale"
    printfn "                              multiplicative-scaling - Multiplicative with scale"
    printfn "  --rejection <algorithm>   Pixel rejection algorithm (default: none)"
    printfn "                              none      - No rejection"
    printfn "                              minmax    - Min/max clipping"
    printfn "                              sigma     - Iterative sigma clipping"
    printfn "                              linearfit - Linear fit clipping"
    printfn "  --low-sigma <value>       Low rejection threshold (default: 2.5)"
    printfn "  --high-sigma <value>      High rejection threshold (default: 2.0)"
    printfn "  --low-count <n>           Drop N lowest pixels (for minmax)"
    printfn "  --high-count <n>          Drop N highest pixels (for minmax)"
    printfn "  --iterations <n>          Rejection iterations (default: 3)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep integrate -i \"subs/*.xisf\" -o \"master.xisf\""
    printfn "  xisfprep integrate -i \"subs/*.xisf\" -o \"master.xisf\" --rejection linearfit --low-sigma 2.5"

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        Log.Error("integrate command is not yet implemented")
        printfn ""
        printfn "Arguments received:"
        args |> Array.iter (fun arg -> printfn "  %s" arg)
        printfn ""
        printfn "Run 'xisfprep integrate --help' for usage information"
        1
