module Commands.Headers

open System
open System.IO
open Serilog
open XisfLib.Core

type HeadersOptions = {
    Input: string
    Keys: string list
    History: string list
    Output: string option
    Overwrite: bool
}

let showHelp() =
    printfn "headers - Extract FITS keywords and HISTORY values from files"
    printfn ""
    printfn "Usage: xisfprep headers [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input files (wildcards supported)"
    printfn ""
    printfn "Optional:"
    printfn "  --keys <list>             Comma-separated FITS keywords to extract"
    printfn "                              e.g., FILTER,CCD-TEMP,EXPOSURE"
    printfn "  --history <list>          Comma-separated HISTORY comment patterns"
    printfn "                              e.g., masterBias.fileName,masterDark.fileName"
    printfn "  --output, -o <file>       Write results to CSV file"
    printfn "  --overwrite               Overwrite existing output file"
    printfn ""
    printfn "Output:"
    printfn "  Table with filename and requested values (or CSV if --output specified)"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep headers -i \"calibrated/*.xisf\" --keys FILTER,CCD-TEMP"
    printfn "  xisfprep headers -i \"calibrated/*.xisf\" --history \"masterBias.fileName,masterFlat.fileName\""
    printfn "  xisfprep headers -i \"calibrated/*.xisf\" --keys FILTER --history \"masterFlat.fileName\""
    printfn "  xisfprep headers -i \"calibrated/*.xisf\" --keys FILTER -o headers.csv"

let parseArgs (args: string array) : HeadersOptions =
    let rec parse (args: string list) (opts: HeadersOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--keys" :: value :: rest ->
            let keys = value.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            parse rest { opts with Keys = keys }
        | "--history" :: value :: rest ->
            let patterns = value.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            parse rest { opts with History = patterns }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = Some value }
        | "--overwrite" :: rest ->
            parse rest { opts with Overwrite = true }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Keys = []
        History = []
        Output = None
        Overwrite = false
    }

    let opts = parse (List.ofArray args) defaults

    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"
    if opts.Keys.IsEmpty && opts.History.IsEmpty then failwith "At least one of --keys or --history is required"

    opts

let extractHeaders (filePath: string) (keys: string list) (historyPatterns: string list) : Map<string, string> =
    let reader = new XisfReader()
    let metadata = reader.ReadMetadataFromFile(filePath)
    let img = metadata.Images.[0]

    let mutable results = Map.empty

    if isNull img.AssociatedElements then
        results
    else
        let fitsKeywords =
            img.AssociatedElements
            |> Seq.choose (fun e ->
                match e with
                | :? XisfFitsKeyword as fits -> Some fits
                | _ -> None)
            |> Seq.toArray

        // Extract requested FITS keywords
        for key in keys do
            let value =
                fitsKeywords
                |> Array.tryFind (fun k -> k.Name = key)
                |> Option.map (fun k -> k.Value.Trim([|'\''|]))
                |> Option.defaultValue ""
            results <- results.Add(key, value)

        // Extract HISTORY comment patterns
        let historyComments =
            fitsKeywords
            |> Array.filter (fun k -> k.Name = "HISTORY")
            |> Array.map (fun k -> k.Comment)

        for pattern in historyPatterns do
            let prefix = $"ImageCalibration.{pattern}: "
            let value =
                historyComments
                |> Array.tryFind (fun c -> c.StartsWith(prefix))
                |> Option.map (fun c -> c.Substring(prefix.Length))
                |> Option.defaultValue ""
            results <- results.Add(pattern, value)

        results

let escapeCsv (s: string) =
    if s.Contains(",") || s.Contains("\"") || s.Contains("\n") then
        "\"" + s.Replace("\"", "\"\"") + "\""
    else
        s

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        try
            let opts = parseArgs args

            let inputDir = Path.GetDirectoryName(opts.Input)
            let pattern = Path.GetFileName(opts.Input)
            let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

            if not (Directory.Exists(actualDir)) then
                Log.Error($"Input directory not found: {actualDir}")
                1
            else
                let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                if files.Length = 0 then
                    Log.Error($"No files found matching pattern: {opts.Input}")
                    1
                else
                    // Check output file exists
                    match opts.Output with
                    | Some outPath when File.Exists(outPath) && not opts.Overwrite ->
                        Log.Warning($"Output file '{outPath}' already exists, skipping (use --overwrite to replace)")
                        1
                    | _ ->
                        // Collect all results
                        let allResults =
                            files
                            |> Array.map (fun f ->
                                let fileName = Path.GetFileName(f)
                                let headers = extractHeaders f opts.Keys opts.History
                                (fileName, headers))

                        // Determine column headers
                        let columns = List.append opts.Keys opts.History

                        match opts.Output with
                        | Some outPath ->
                            // Write to CSV
                            let headerLine = "File," + (columns |> List.map escapeCsv |> String.concat ",")
                            let dataLines =
                                allResults
                                |> Array.map (fun (fileName, headers) ->
                                    let values = columns |> List.map (fun col ->
                                        headers.TryFind(col) |> Option.defaultValue "" |> escapeCsv)
                                    escapeCsv fileName + "," + (values |> String.concat ","))

                            File.WriteAllLines(outPath, Array.append [| headerLine |] dataLines)
                            printfn $"Wrote {files.Length} rows to {outPath}"
                            0

                        | None ->
                            // Print table to console
                            let fileColWidth =
                                allResults
                                |> Array.map (fun (name, _) -> name.Length)
                                |> Array.max
                                |> max 4

                            let colWidths =
                                columns
                                |> List.map (fun col ->
                                    let maxValue =
                                        allResults
                                        |> Array.map (fun (_, headers) ->
                                            headers.TryFind(col) |> Option.defaultValue "" |> String.length)
                                        |> Array.max
                                    max col.Length maxValue)

                            // Print header
                            let headerLine =
                                sprintf "%-*s" fileColWidth "File" +
                                (List.zip columns colWidths
                                 |> List.map (fun (col, width) -> sprintf " | %-*s" width col)
                                 |> String.concat "")
                            printfn "%s" headerLine

                            // Print separator
                            let sepLine =
                                String.replicate fileColWidth "-" +
                                (colWidths |> List.map (fun w -> "-+-" + String.replicate w "-") |> String.concat "")
                            printfn "%s" sepLine

                            // Print rows
                            for (fileName, headers) in allResults do
                                let row =
                                    sprintf "%-*s" fileColWidth fileName +
                                    (List.zip columns colWidths
                                     |> List.map (fun (col, width) ->
                                         let value = headers.TryFind(col) |> Option.defaultValue ""
                                         sprintf " | %-*s" width value)
                                     |> String.concat "")
                                printfn "%s" row

                            printfn ""
                            printfn $"Total: {files.Length} files"

                            0
        with ex ->
            Log.Error($"Error: {ex.Message}")
            printfn ""
            printfn "Run 'xisfprep headers --help' for usage information"
            1
