module Commands.Stats

open System
open System.IO
open Serilog
open XisfLib.Core

type MetricsLevel =
    | Basic
    | Histogram
    | All

type GroupBy =
    | NoGrouping
    | ByTarget
    | ByFilter
    | ByTargetAndFilter
    | ByImageType

type SortBy =
    | ByName
    | ByMedian
    | BySNR
    | ByMAD
    | ByFWHM
    | ByStars

type SortOrder =
    | Ascending
    | Descending

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
    FileSize: int64
    Compression: string
    // FITS metadata
    Object: string option
    Filter: string option
    ImageType: string option
    // Star detection results
    StarCount: int option
    MedianFWHM: float option
}

let showHelp() =
    printfn "stats - Calculate and display image statistics with grouping and sorting"
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
    printfn "                              all       - Basic + MAD, SNR, FITS metadata"
    printfn "                              histogram - Include histogram data"
    printfn "  --detect-stars            Run star detection (auto-skipped for calibration frames)"
    printfn "  --group-by <strategy>     Group output (default: none)"
    printfn "                              target         - Group by Object FITS keyword"
    printfn "                              filter         - Group by Filter FITS keyword"
    printfn "                              target,filter  - Group by target, then filter"
    printfn "                              imagetype      - Group by frame type"
    printfn "  --sort-by <metric>        Sort within groups (default: name)"
    printfn "                              name, median, snr, mad, fwhm, stars"
    printfn "  --sort-order <dir>        Sort direction: asc or desc (auto default)"
    printfn "  --min-median <value>      Filter out frames below median threshold"
    printfn "  --max-mad <value>         Filter out frames above MAD threshold"
    printfn "  --min-snr <value>         Filter out frames below SNR threshold"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep stats -i \"images/*.xisf\" --metrics all --group-by target,filter"
    printfn "  xisfprep stats -i \"Ha*.xisf\" --metrics all --sort-by snr --sort-order desc"
    printfn "  xisfprep stats -i \"*.xisf\" --detect-stars --sort-by fwhm --min-median 500"

let parseArgs (args: string array) =
    let rec parse (args: string list) input output metrics detectStars groupBy sortBy sortOrder minMedian maxMad minSnr =
        match args with
        | [] -> (input, output, metrics, detectStars, groupBy, sortBy, sortOrder, minMedian, maxMad, minSnr)
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest (Some value) output metrics detectStars groupBy sortBy sortOrder minMedian maxMad minSnr
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest input (Some value) metrics detectStars groupBy sortBy sortOrder minMedian maxMad minSnr
        | "--metrics" :: value :: rest ->
            let level = match value.ToLower() with
                        | "basic" -> Basic
                        | "histogram" -> Histogram
                        | "all" -> All
                        | _ -> failwithf "Unknown metrics level: %s" value
            parse rest input output (Some level) detectStars groupBy sortBy sortOrder minMedian maxMad minSnr
        | "--detect-stars" :: rest ->
            parse rest input output metrics true groupBy sortBy sortOrder minMedian maxMad minSnr
        | "--group-by" :: value :: rest ->
            let group = match value.ToLower() with
                        | "target" -> ByTarget
                        | "filter" -> ByFilter
                        | "target,filter" | "targetfilter" -> ByTargetAndFilter
                        | "imagetype" -> ByImageType
                        | _ -> failwithf "Unknown grouping: %s" value
            parse rest input output metrics detectStars (Some group) sortBy sortOrder minMedian maxMad minSnr
        | "--sort-by" :: value :: rest ->
            let sort = match value.ToLower() with
                       | "name" -> ByName
                       | "median" -> ByMedian
                       | "snr" -> BySNR
                       | "mad" -> ByMAD
                       | "fwhm" -> ByFWHM
                       | "stars" -> ByStars
                       | _ -> failwithf "Unknown sort metric: %s" value
            parse rest input output metrics detectStars groupBy (Some sort) sortOrder minMedian maxMad minSnr
        | "--sort-order" :: value :: rest ->
            let order = match value.ToLower() with
                        | "asc" | "ascending" -> Ascending
                        | "desc" | "descending" -> Descending
                        | _ -> failwithf "Unknown sort order: %s" value
            parse rest input output metrics detectStars groupBy sortBy (Some order) minMedian maxMad minSnr
        | "--min-median" :: value :: rest ->
            let threshold = Double.Parse(value)
            parse rest input output metrics detectStars groupBy sortBy sortOrder (Some threshold) maxMad minSnr
        | "--max-mad" :: value :: rest ->
            let threshold = Double.Parse(value)
            parse rest input output metrics detectStars groupBy sortBy sortOrder minMedian (Some threshold) minSnr
        | "--min-snr" :: value :: rest ->
            let threshold = Double.Parse(value)
            parse rest input output metrics detectStars groupBy sortBy sortOrder minMedian maxMad (Some threshold)
        | arg :: rest ->
            failwithf "Unknown argument: %s" arg

    let (input, output, metrics, detectStars, groupBy, sortBy, sortOrder, minMedian, maxMad, minSnr) =
        parse (List.ofArray args) None None None false None None None None None None

    let input = match input with Some v -> v | None -> failwith "Required argument: --input"
    let metrics = metrics |> Option.defaultValue Basic
    let groupBy = groupBy |> Option.defaultValue NoGrouping
    let sortBy = sortBy |> Option.defaultValue ByName

    // Auto-determine sort order based on metric if not specified
    let sortOrder =
        match sortOrder with
        | Some order -> order
        | None ->
            match sortBy with
            | ByMedian | BySNR | ByStars -> Descending  // Higher is better
            | ByMAD | ByFWHM | ByName -> Ascending      // Lower is better

    (input, output, metrics, detectStars, groupBy, sortBy, sortOrder, minMedian, maxMad, minSnr)

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

// Step 1: Check if an element is the FITS keyword we want
let tryGetFitsKeywordValue (keywordName: string) (elem: XisfCoreElement) : string option =
    match elem with
    | :? XisfFitsKeyword as fits when fits.Name = keywordName -> Some fits.Value
    | _ -> None

// Step 2: Find the keyword in a collection
let findFitsKeyword (keywordName: string) (elements: seq<XisfCoreElement>) : string option =
    elements |> Seq.tryPick (tryGetFitsKeywordValue keywordName)

// Step 3: Extract from image
let extractFitsKeyword (img: XisfImage) (keywordName: string) : string option =
    if isNull img.AssociatedElements then
        None
    else
        findFitsKeyword keywordName img.AssociatedElements

let shouldSkipStarDetection (imageType: string option) : bool =
    match imageType with
    | Some typ ->
        let typeLower = typ.ToLower().Trim()
        typeLower = "bias" || typeLower = "dark" || typeLower = "flat" ||
        typeLower = "master bias" || typeLower = "master dark" || typeLower = "master flat" ||
        typeLower = "masterbias" || typeLower = "masterdark" || typeLower = "masterflat"
    | None -> false

let formatFileSize (bytes: int64) =
    if bytes < 1024L then
        sprintf "%d B" bytes
    elif bytes < 1024L * 1024L then
        sprintf "%.1f KB" (float bytes / 1024.0)
    elif bytes < 1024L * 1024L * 1024L then
        sprintf "%.2f MB" (float bytes / (1024.0 * 1024.0))
    else
        sprintf "%.2f GB" (float bytes / (1024.0 * 1024.0 * 1024.0))

let analyzeImage (filePath: string) (metricsLevel: MetricsLevel) (detectStars: bool) : Async<ImageStats option> =
    async {
        try
            let fileName = Path.GetFileName(filePath)
            let fileInfo = FileInfo(filePath)
            let fileSize = fileInfo.Length

            let reader = new XisfReader()
            let! unit = reader.ReadAsync(filePath) |> Async.AwaitTask
            let img = unit.Images.[0]

            // Extract FITS metadata
            let object = extractFitsKeyword img "OBJECT"
            let filter = extractFitsKeyword img "FILTER"
            let imageType = extractFitsKeyword img "IMAGETYP"

            let compression =
                if img.PixelData :? InlineDataBlock then
                    let block = img.PixelData :?> InlineDataBlock
                    match block.Compression with
                    | null -> "none"
                    | comp -> comp.ToString()
                else
                    "embedded"

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

            // Star detection with smart skip for calibration frames
            let (starCount, medianFWHM) =
                if detectStars then
                    if shouldSkipStarDetection imageType then
                        let typeDesc = imageType |> Option.defaultValue "unknown"
                        Log.Information($"Skipping star detection for {typeDesc} frame: {fileName}")
                        (None, None)
                    else
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

                        let results: StarDetection.StarDetectionResults =
                            StarDetection.detectStars channelData width height StarDetection.defaultParams
                        StarDetection.printStarDetectionResults results

                        // Extract star count and median FWHM
                        let count = results.Channels |> Array.sumBy (fun r -> Seq.length r.Stars)
                        let allFwhm =
                            results.Channels
                            |> Array.collect (fun r -> r.Stars |> Seq.map (fun s -> s.FWHM) |> Seq.toArray)
                            |> Array.sort
                        let fwhm =
                            if allFwhm.Length > 0 then
                                let mid = allFwhm.Length / 2
                                if allFwhm.Length % 2 = 0 then
                                    Some ((allFwhm.[mid - 1] + allFwhm.[mid]) / 2.0)
                                else
                                    Some allFwhm.[mid]
                            else None
                        (Some count, fwhm)
                else
                    (None, None)

            return Some {
                FileName = fileName
                Width = width
                Height = height
                Channels = channels
                ChannelStats = channelStats
                FileSize = fileSize
                Compression = compression
                Object = object
                Filter = filter
                ImageType = imageType
                StarCount = starCount
                MedianFWHM = medianFWHM
            }
        with ex ->
            Log.Error($"Error processing {Path.GetFileName(filePath)}: {ex.Message}")
            return None
    }

let applyFilters (stats: ImageStats[]) (minMedian: float option) (maxMad: float option) (minSnr: float option) : ImageStats[] =
    stats
    |> Array.filter (fun s ->
        let passMedian =
            match minMedian with
            | None -> true
            | Some threshold ->
                // Use first channel median for filtering
                s.ChannelStats.[0].Median >= threshold

        let passMad =
            match maxMad with
            | None -> true
            | Some threshold ->
                match s.ChannelStats.[0].MAD with
                | Some mad -> mad <= threshold
                | None -> true

        let passSnr =
            match minSnr with
            | None -> true
            | Some threshold ->
                match s.ChannelStats.[0].SNR with
                | Some snr -> snr >= threshold
                | None -> true

        passMedian && passMad && passSnr
    )

let sortStats (stats: ImageStats[]) (sortBy: SortBy) (sortOrder: SortOrder) : ImageStats[] =
    let compareFunc =
        match sortBy with
        | ByName -> fun (a: ImageStats) (b: ImageStats) -> String.Compare(a.FileName, b.FileName)
        | ByMedian -> fun a b -> compare a.ChannelStats.[0].Median b.ChannelStats.[0].Median
        | BySNR -> fun a b ->
            let snrA = a.ChannelStats.[0].SNR |> Option.defaultValue 0.0
            let snrB = b.ChannelStats.[0].SNR |> Option.defaultValue 0.0
            compare snrA snrB
        | ByMAD -> fun a b ->
            let madA = a.ChannelStats.[0].MAD |> Option.defaultValue System.Double.MaxValue
            let madB = b.ChannelStats.[0].MAD |> Option.defaultValue System.Double.MaxValue
            compare madA madB
        | ByFWHM -> fun a b ->
            let fwhmA = a.MedianFWHM |> Option.defaultValue System.Double.MaxValue
            let fwhmB = b.MedianFWHM |> Option.defaultValue System.Double.MaxValue
            compare fwhmA fwhmB
        | ByStars -> fun a b ->
            let starsA = a.StarCount |> Option.defaultValue 0
            let starsB = b.StarCount |> Option.defaultValue 0
            compare starsA starsB

    let sorted = stats |> Array.sortWith compareFunc
    match sortOrder with
    | Ascending -> sorted
    | Descending -> Array.rev sorted

let groupStats (stats: ImageStats[]) (groupBy: GroupBy) : (string * ImageStats[])[] =
    match groupBy with
    | NoGrouping -> [|("All Files", stats)|]
    | ByTarget ->
        stats
        |> Array.groupBy (fun s -> s.Object |> Option.defaultValue "Unknown Target")
        |> Array.sortBy fst
    | ByFilter ->
        stats
        |> Array.groupBy (fun s -> s.Filter |> Option.defaultValue "No Filter")
        |> Array.sortBy fst
    | ByTargetAndFilter ->
        stats
        |> Array.groupBy (fun s ->
            let target = s.Object |> Option.defaultValue "Unknown Target"
            let filter = s.Filter |> Option.defaultValue "No Filter"
            (target, filter))
        |> Array.sortBy fst
        |> Array.map (fun ((target, filter), items) -> (sprintf "%s - %s" target filter, items))
    | ByImageType ->
        stats
        |> Array.groupBy (fun s -> s.ImageType |> Option.defaultValue "Unknown Type")
        |> Array.sortBy fst

let printStats (stats: ImageStats[]) (groupBy: GroupBy) (sortBy: SortBy) (sortOrder: SortOrder) =
    printfn ""
    printfn "Image Statistics:"
    printfn "================================================================================"

    let hasAdvanced =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].MAD.IsSome

    let hasFitsMetadata =
        hasAdvanced &&
        stats.Length > 0 &&
        (stats.[0].Object.IsSome || stats.[0].Filter.IsSome || stats.[0].ImageType.IsSome)

    let hasStarData =
        stats.Length > 0 &&
        (stats.[0].StarCount.IsSome || stats.[0].MedianFWHM.IsSome)

    let groups = groupStats stats groupBy
    let sortedGroups =
        groups |> Array.map (fun (groupName, groupStats) ->
            (groupName, sortStats groupStats sortBy sortOrder))

    for (groupName, groupStats) in sortedGroups do
        if groupBy <> NoGrouping then
            printfn ""
            printfn "================================================================================"
            printfn "GROUP: %s (%d file%s)" groupName groupStats.Length (if groupStats.Length = 1 then "" else "s")
            printfn "================================================================================"

        for imgStats in groupStats do
            printfn ""
            if hasAdvanced then
                let metaInfo =
                    if hasFitsMetadata then
                        let obj = imgStats.Object |> Option.defaultValue "-"
                        let flt = imgStats.Filter |> Option.defaultValue "-"
                        let typ = imgStats.ImageType |> Option.defaultValue "-"
                        sprintf " [%s | %s | %s]" obj flt typ
                    else ""
                let starInfo =
                    if hasStarData then
                        match (imgStats.StarCount, imgStats.MedianFWHM) with
                        | (Some count, Some fwhm) -> sprintf " Stars: %d, FWHM: %.2f" count fwhm
                        | (Some count, None) -> sprintf " Stars: %d" count
                        | (None, Some fwhm) -> sprintf " FWHM: %.2f" fwhm
                        | _ -> ""
                    else ""
                printfn "File: %s (%dx%d, %d channel%s, %s, %s)%s%s"
                    imgStats.FileName
                    imgStats.Width
                    imgStats.Height
                    imgStats.Channels
                    (if imgStats.Channels = 1 then "" else "s")
                    (formatFileSize imgStats.FileSize)
                    imgStats.Compression
                    metaInfo
                    starInfo
            else
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

    let hasFitsMetadata =
        hasAdvanced &&
        stats.Length > 0 &&
        (stats.[0].Object.IsSome || stats.[0].Filter.IsSome || stats.[0].ImageType.IsSome)

    let hasStarData =
        stats.Length > 0 &&
        (stats.[0].StarCount.IsSome || stats.[0].MedianFWHM.IsSome)

    let baseHeader =
        if hasAdvanced then
            let metadata = if hasFitsMetadata then ",Object,Filter,ImageType" else ""
            let starData = if hasStarData then ",StarCount,MedianFWHM" else ""
            sprintf "FileName,Width,Height,Channels,FileSize,Compression%s%s,Channel,Min,Max,Mean,Median,StdDev,MAD,SNR" metadata starData
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
                    let metadataStr =
                        if hasFitsMetadata then
                            let obj = imgStats.Object |> Option.defaultValue ""
                            let flt = imgStats.Filter |> Option.defaultValue ""
                            let typ = imgStats.ImageType |> Option.defaultValue ""
                            sprintf ",%s,%s,%s" obj flt typ
                        else ""
                    let starDataStr =
                        if hasStarData then
                            let count = imgStats.StarCount |> Option.map string |> Option.defaultValue ""
                            let fwhm = imgStats.MedianFWHM |> Option.map (sprintf "%.2f") |> Option.defaultValue ""
                            sprintf ",%s,%s" count fwhm
                        else ""
                    sprintf "%s,%d,%d,%d,%d,%s%s%s,%s,%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f"
                        imgStats.FileName
                        imgStats.Width
                        imgStats.Height
                        imgStats.Channels
                        imgStats.FileSize
                        imgStats.Compression
                        metadataStr
                        starDataStr
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
                let (inputPattern, outputPath, metricsLevel, detectStars, groupBy, sortBy, sortOrder, minMedian, maxMad, minSnr) = parseArgs args

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

                        let allStats = results |> Array.choose id

                        if allStats.Length = 0 then
                            Log.Error("No images could be analyzed")
                            return 1
                        else
                            // Apply filters
                            let filteredStats = applyFilters allStats minMedian maxMad minSnr

                            if filteredStats.Length = 0 then
                                Log.Warning("No images passed filtering criteria")
                                return 1
                            else
                                if filteredStats.Length < allStats.Length then
                                    let filtered = allStats.Length - filteredStats.Length
                                    let filteredPlural = if filtered = 1 then "" else "s"
                                    printfn $"Filtered out {filtered} file{filteredPlural} based on quality thresholds"

                                printStats filteredStats groupBy sortBy sortOrder

                                match outputPath with
                                | Some path ->
                                    writeCsv path filteredStats
                                    printfn ""
                                    printfn $"CSV exported to: {path}"
                                | None -> ()

                                printfn ""
                                let plural = if files.Length = 1 then "" else "s"
                                printfn $"Analyzed {allStats.Length} of {files.Length} file{plural} successfully"
                                if filteredStats.Length < allStats.Length then
                                    let displayedPlural = if filteredStats.Length = 1 then "" else "s"
                                    printfn $"Displayed {filteredStats.Length} file{displayedPlural} after filtering"

                                return if allStats.Length = files.Length then 0 else 1
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep stats --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
