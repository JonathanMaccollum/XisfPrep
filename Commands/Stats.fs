module Commands.Stats

open System
open System.IO
open Serilog
open XisfLib.Core

type MetricsLevel =
    | Basic
    | Histogram
    | All

type ChannelStats = {
    Channel: int
    Min: float
    Max: float
    Mean: float
    Median: float
    StdDev: float
    MAD: float option
    SNR: float option
    Histogram: int[] option
}

type ImageStats = {
    FileName: string
    Width: int
    Height: int
    Channels: int
    ChannelStats: ChannelStats[]
}

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

let parseArgs (args: string array) =
    let rec parse (args: string list) input output metrics detectStars =
        match args with
        | [] -> (input, output, metrics, detectStars)
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest (Some value) output metrics detectStars
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest input (Some value) metrics detectStars
        | "--metrics" :: value :: rest ->
            let level = match value.ToLower() with
                        | "basic" -> Basic
                        | "histogram" -> Histogram
                        | "all" -> All
                        | _ -> failwithf "Unknown metrics level: %s" value
            parse rest input output (Some level) detectStars
        | "--detect-stars" :: rest ->
            parse rest input output metrics true
        | arg :: rest ->
            failwithf "Unknown argument: %s" arg

    let (input, output, metrics, detectStars) = parse (List.ofArray args) None None None false
    let input = match input with Some v -> v | None -> failwith "Required argument: --input"
    let metrics = metrics |> Option.defaultValue Basic
    (input, output, metrics, detectStars)

let calculateStats (values: float[]) =
    if Array.isEmpty values then
        (0.0, 0.0, 0.0, 0.0, 0.0)
    else
        let min = Array.min values
        let max = Array.max values
        let mean = Array.average values

        let sorted = Array.copy values
        Array.sortInPlace sorted
        let median =
            let mid = sorted.Length / 2
            if sorted.Length % 2 = 0 then
                (sorted.[mid - 1] + sorted.[mid]) / 2.0
            else
                sorted.[mid]

        let variance = values |> Array.sumBy (fun x -> (x - mean) * (x - mean))
        let stdDev = sqrt (variance / float values.Length)

        (min, max, mean, median, stdDev)

let calculateHistogram (values: float[]) (bins: int) =
    if Array.isEmpty values then
        Array.zeroCreate bins
    else
        let minVal = Array.min values
        let maxVal = Array.max values
        let range = maxVal - minVal
        let histogram = Array.zeroCreate bins

        if range > 0.0 then
            for value in values do
                let binIndex = int ((value - minVal) / range * float (bins - 1))
                let clampedIndex = min (bins - 1) (max 0 binIndex)
                histogram.[clampedIndex] <- histogram.[clampedIndex] + 1

        histogram

let calculateMAD (values: float[]) (median: float) =
    if Array.isEmpty values then
        0.0
    else
        let deviations = values |> Array.map (fun v -> abs (v - median))
        let sorted = Array.sort deviations
        let mid = sorted.Length / 2
        if sorted.Length % 2 = 0 then
            (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else
            sorted.[mid]

let calculateSNR (mean: float) (stdDev: float) =
    if stdDev > 0.0 then mean / stdDev
    else 0.0

let analyzeImage (filePath: string) (metricsLevel: MetricsLevel) (detectStars: bool) : Async<ImageStats option> =
    async {
        try
            let fileName = Path.GetFileName(filePath)
            let reader = new XisfReader()
            let! unit = reader.ReadAsync(filePath) |> Async.AwaitTask
            let img = unit.Images.[0]

            let pixelData =
                if img.PixelData :? InlineDataBlock then
                    (img.PixelData :?> InlineDataBlock).Data.ToArray()
                else
                    failwith "Expected inline data"

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount
            let pixelCount = width * height

            let channelStats =
                Array.init channels (fun ch ->
                    let values = Array.init pixelCount (fun pix ->
                        let offset = (pix * channels * 2) + (ch * 2)
                        float (uint16 pixelData.[offset] ||| (uint16 pixelData.[offset + 1] <<< 8))
                    )

                    let (min, max, mean, median, stdDev) = calculateStats values
                    let madValue = calculateMAD values median

                    let mad =
                        match metricsLevel with
                        | All -> Some madValue
                        | _ -> None

                    let snr =
                        match metricsLevel with
                        | All -> Some (calculateSNR mean stdDev)
                        | _ -> None

                    let histogram =
                        match metricsLevel with
                        | Histogram | All -> Some (calculateHistogram values 256)
                        | Basic -> None

                    { Channel = ch
                      Min = min
                      Max = max
                      Mean = mean
                      Median = median
                      StdDev = stdDev
                      MAD = mad
                      SNR = snr
                      Histogram = histogram }
                )

            if detectStars then
                let channelData =
                    Array.init channels (fun ch ->
                        let values = Array.init pixelCount (fun pix ->
                            let offset = (pix * channels * 2) + (ch * 2)
                            float (uint16 pixelData.[offset] ||| (uint16 pixelData.[offset + 1] <<< 8))
                        )
                        let stats = channelStats.[ch]
                        let mad = stats.MAD |> Option.defaultValue (calculateMAD values stats.Median)
                        let channelName =
                            if channels = 1 then "Mono"
                            elif channels = 3 then
                                match ch with
                                | 0 -> "Red"
                                | 1 -> "Green"
                                | 2 -> "Blue"
                                | _ -> sprintf "Ch%d" ch
                            else sprintf "Ch%d" ch
                        (values, mad, ch, channelName)
                    )

                let results = StarDetection.detectStars channelData width height StarDetection.defaultParams
                StarDetection.printStarDetectionResults results

            return Some {
                FileName = fileName
                Width = width
                Height = height
                Channels = channels
                ChannelStats = channelStats
            }
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(filePath)}: {ex.Message}")
            return None
    }

let printStats (stats: ImageStats[]) =
    printfn ""
    printfn "Image Statistics:"
    printfn "================================================================================"

    let hasAdvanced =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].MAD.IsSome

    for imgStats in stats do
        printfn ""
        printfn "File: %s (%dx%d, %d channel%s)"
            imgStats.FileName
            imgStats.Width
            imgStats.Height
            imgStats.Channels
            (if imgStats.Channels = 1 then "" else "s")
        printfn "--------------------------------------------------------------------------------"
        if hasAdvanced then
            printfn "%-8s %8s %8s %10s %10s %10s %10s %10s" "Channel" "Min" "Max" "Mean" "Median" "StdDev" "MAD" "SNR"
        else
            printfn "%-8s %8s %8s %10s %10s %10s" "Channel" "Min" "Max" "Mean" "Median" "StdDev"
        printfn "--------------------------------------------------------------------------------"

        for ch in imgStats.ChannelStats do
            let channelName =
                if imgStats.Channels = 1 then "Mono"
                elif imgStats.Channels = 3 then
                    match ch.Channel with
                    | 0 -> "Red"
                    | 1 -> "Green"
                    | 2 -> "Blue"
                    | _ -> sprintf "Ch%d" ch.Channel
                else sprintf "Ch%d" ch.Channel

            if hasAdvanced then
                let madVal = ch.MAD |> Option.defaultValue 0.0
                let snrVal = ch.SNR |> Option.defaultValue 0.0
                printfn "%-8s %8.0f %8.0f %10.2f %10.2f %10.2f %10.2f %10.2f"
                    channelName ch.Min ch.Max ch.Mean ch.Median ch.StdDev madVal snrVal
            else
                printfn "%-8s %8.0f %8.0f %10.2f %10.2f %10.2f"
                    channelName ch.Min ch.Max ch.Mean ch.Median ch.StdDev

let writeCsv (outputPath: string) (stats: ImageStats[]) =
    use writer = new StreamWriter(outputPath)

    let hasAdvanced =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].MAD.IsSome

    let hasHistogram =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].Histogram.IsSome

    let baseHeader =
        if hasAdvanced then
            "FileName,Width,Height,Channels,Channel,Min,Max,Mean,Median,StdDev,MAD,SNR"
        else
            "FileName,Width,Height,Channels,Channel,Min,Max,Mean,Median,StdDev"

    if hasHistogram then
        let binCount = stats.[0].ChannelStats.[0].Histogram.Value.Length
        let histHeaders = [0 .. binCount - 1] |> List.map (sprintf "Bin%d") |> String.concat ","
        writer.WriteLine($"{baseHeader},{histHeaders}")
    else
        writer.WriteLine(baseHeader)

    for imgStats in stats do
        for ch in imgStats.ChannelStats do
            let channelName =
                if imgStats.Channels = 1 then "Mono"
                elif imgStats.Channels = 3 then
                    match ch.Channel with
                    | 0 -> "Red"
                    | 1 -> "Green"
                    | 2 -> "Blue"
                    | _ -> sprintf "Ch%d" ch.Channel
                else sprintf "Ch%d" ch.Channel

            let basicStats =
                if hasAdvanced then
                    let madVal = ch.MAD |> Option.defaultValue 0.0
                    let snrVal = ch.SNR |> Option.defaultValue 0.0
                    sprintf "%s,%d,%d,%d,%s,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f"
                        imgStats.FileName
                        imgStats.Width
                        imgStats.Height
                        imgStats.Channels
                        channelName
                        ch.Min
                        ch.Max
                        ch.Mean
                        ch.Median
                        ch.StdDev
                        madVal
                        snrVal
                else
                    sprintf "%s,%d,%d,%d,%s,%.0f,%.0f,%.2f,%.2f,%.2f"
                        imgStats.FileName
                        imgStats.Width
                        imgStats.Height
                        imgStats.Channels
                        channelName
                        ch.Min
                        ch.Max
                        ch.Mean
                        ch.Median
                        ch.StdDev

            match ch.Histogram with
            | Some hist ->
                let histData = hist |> Array.map string |> String.concat ","
                writer.WriteLine($"{basicStats},{histData}")
            | None ->
                writer.WriteLine(basicStats)

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        let computation = async {
            try
                let (inputPattern, outputPath, metricsLevel, detectStars) = parseArgs args

                let inputDir = Path.GetDirectoryName(inputPattern)
                let pattern = Path.GetFileName(inputPattern)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                    if files.Length = 0 then
                        Log.Error($"No files found matching pattern: {inputPattern}")
                        return 1
                    else
                        let plural = if files.Length = 1 then "" else "s"
                        printfn $"Found {files.Length} file{plural} to analyze"

                        let! results =
                            files
                            |> Array.map (fun f -> analyzeImage f metricsLevel detectStars)
                            |> Async.Parallel

                        let stats = results |> Array.choose id

                        if stats.Length = 0 then
                            Log.Error("No images could be analyzed")
                            return 1
                        else
                            printStats stats

                            match outputPath with
                            | Some path ->
                                writeCsv path stats
                                printfn ""
                                printfn $"CSV exported to: {path}"
                            | None -> ()

                            printfn ""
                            let plural = if files.Length = 1 then "" else "s"
                            printfn $"Analyzed {stats.Length} of {files.Length} file{plural} successfully"

                            return if stats.Length = files.Length then 0 else 1
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep stats --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
