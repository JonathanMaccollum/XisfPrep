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
    | ByExposure
    | ByDate

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
    ExposureTime: float option
    ObservationDate: DateTime option
    SideOfPier: string option
    // Star detection results
    StarCount: int option
    MedianFWHM: float option
}

type AnalysisResult = {
    FilePath: string
    Stats: ImageStats option
    Error: string option
}

type StatsOptions = {
    Input: string
    Output: string option
    Metrics: MetricsLevel
    DetectStars: bool
    HeaderOnly: bool
    GroupBy: GroupBy
    SortBy: SortBy
    SortOrder: SortOrder
    MinMedian: float option
    MaxMad: float option
    MinSnr: float option
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
    printfn "  --header-only             Fast metadata-only mode (10-100x faster)"
    printfn "                              - No pixel statistics or star detection"
    printfn "                              - Ideal for inventory and session planning"
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
    printfn "                              name, median, snr, mad, fwhm, stars, exposure, date"
    printfn "  --sort-order <dir>        Sort direction: asc or desc (auto default)"
    printfn "  --min-median <value>      Filter out frames below median threshold"
    printfn "  --max-mad <value>         Filter out frames above MAD threshold"
    printfn "  --min-snr <value>         Filter out frames below SNR threshold"
    printfn ""
    printfn "Examples:"
    printfn "  # Pixel analysis with quality metrics"
    printfn "  xisfprep stats -i \"images/*.xisf\" --metrics all --group-by target,filter"
    printfn "  xisfprep stats -i \"Ha*.xisf\" --metrics all --sort-by snr --sort-order desc"
    printfn "  xisfprep stats -i \"*.xisf\" --detect-stars --sort-by fwhm --min-median 500"
    printfn ""
    printfn "  # Fast header-only analysis"
    printfn "  xisfprep stats -i \"session/*.xisf\" --header-only --group-by target,filter"
    printfn "  xisfprep stats -i \"archive/**/*.xisf\" --header-only -o inventory.csv"

let parseArgs (args: string array) : StatsOptions =
    // Track sortBy and sortOrder separately for auto-determination
    let rec parse (args: string list) (opts: StatsOptions) (sortByOpt: SortBy option) (sortOrderOpt: SortOrder option) =
        match args with
        | [] ->
            // Apply defaults and auto-determine sort order
            let sortBy = sortByOpt |> Option.defaultValue ByName
            let sortOrder =
                match sortOrderOpt with
                | Some order -> order
                | None ->
                    match sortBy with
                    | ByMedian | BySNR | ByStars | ByExposure -> Descending
                    | ByMAD | ByFWHM | ByName | ByDate -> Ascending
            { opts with SortBy = sortBy; SortOrder = sortOrder }
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value } sortByOpt sortOrderOpt
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parse rest { opts with Output = Some value } sortByOpt sortOrderOpt
        | "--header-only" :: rest ->
            parse rest { opts with HeaderOnly = true } sortByOpt sortOrderOpt
        | "--metrics" :: value :: rest ->
            let level = match value.ToLower() with
                        | "basic" -> Basic
                        | "histogram" -> Histogram
                        | "all" -> All
                        | _ -> failwithf "Unknown metrics level: %s" value
            parse rest { opts with Metrics = level } sortByOpt sortOrderOpt
        | "--detect-stars" :: rest ->
            parse rest { opts with DetectStars = true } sortByOpt sortOrderOpt
        | "--group-by" :: value :: rest ->
            let group = match value.ToLower() with
                        | "target" -> ByTarget
                        | "filter" -> ByFilter
                        | "target,filter" | "targetfilter" -> ByTargetAndFilter
                        | "imagetype" -> ByImageType
                        | _ -> failwithf "Unknown grouping: %s" value
            parse rest { opts with GroupBy = group } sortByOpt sortOrderOpt
        | "--sort-by" :: value :: rest ->
            let sort = match value.ToLower() with
                       | "name" -> ByName
                       | "median" -> ByMedian
                       | "snr" -> BySNR
                       | "mad" -> ByMAD
                       | "fwhm" -> ByFWHM
                       | "stars" -> ByStars
                       | "exposure" -> ByExposure
                       | "date" -> ByDate
                       | _ -> failwithf "Unknown sort metric: %s" value
            parse rest opts (Some sort) sortOrderOpt
        | "--sort-order" :: value :: rest ->
            let order = match value.ToLower() with
                        | "asc" | "ascending" -> Ascending
                        | "desc" | "descending" -> Descending
                        | _ -> failwithf "Unknown sort order: %s" value
            parse rest opts sortByOpt (Some order)
        | "--min-median" :: value :: rest ->
            parse rest { opts with MinMedian = Some (Double.Parse(value)) } sortByOpt sortOrderOpt
        | "--max-mad" :: value :: rest ->
            parse rest { opts with MaxMad = Some (Double.Parse(value)) } sortByOpt sortOrderOpt
        | "--min-snr" :: value :: rest ->
            parse rest { opts with MinSnr = Some (Double.Parse(value)) } sortByOpt sortOrderOpt
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        Output = None
        Metrics = Basic
        DetectStars = false
        HeaderOnly = false
        GroupBy = NoGrouping
        SortBy = ByName
        SortOrder = Ascending
        MinMedian = None
        MaxMad = None
        MinSnr = None
    }

    let opts = parse (List.ofArray args) defaults None None

    // Validation
    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"

    if opts.HeaderOnly && opts.DetectStars then
        failwith "--header-only and --detect-stars are incompatible (star detection requires pixel data)"

    if opts.HeaderOnly && (opts.MinMedian.IsSome || opts.MaxMad.IsSome || opts.MinSnr.IsSome) then
        failwith "--header-only is incompatible with --min-median, --max-mad, and --min-snr (require pixel statistics)"

    if opts.HeaderOnly then
        match opts.SortBy with
        | ByMedian | BySNR | ByMAD | ByFWHM | ByStars ->
            failwith "--header-only is incompatible with --sort-by median/snr/mad/fwhm/stars (require pixel statistics). Use --sort-by name/exposure/date instead."
        | _ -> ()

    opts

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

// ==================== Metadata Extraction Helpers ====================

module MetadataExtraction =
    let tryGetProperty (name: string) (props: XisfProperty seq) =
        props
        |> Seq.tryFind (fun p -> p.Id = name)
        |> Option.bind (fun p ->
            match p with
            | :? XisfStringProperty as sp -> Some sp.Value
            | _ -> None)

    let tryGetNumericProperty (name: string) (props: XisfProperty seq) =
        props
        |> Seq.tryFind (fun p -> p.Id = name)
        |> Option.bind (fun p ->
            match p with
            | :? XisfScalarProperty<single> as fp -> Some (float fp.Value)
            | :? XisfScalarProperty<double> as dp -> Some dp.Value
            | :? XisfScalarProperty<int> as ip -> Some (float ip.Value)
            | :? XisfScalarProperty<uint32> as up -> Some (float up.Value)
            | :? XisfStringProperty as sp ->
                match Double.TryParse(sp.Value) with
                | true, v -> Some v
                | false, _ -> None
            | _ -> None)

    let tryGetFitsKeyword (name: string) (elements: XisfCoreElement seq) =
        elements
        |> Seq.tryPick (fun e ->
            match e with
            | :? XisfFitsKeyword as fk when fk.Name = name -> Some fk.Value
            | _ -> None)

    let tryParseDate (value: string) =
        match DateTime.TryParse(value.Trim('\'', ' ')) with
        | true, d -> Some d
        | false, _ -> None

    let extractFromMetadata (metadata: XisfMetadataUnit) =
        let allElements =
            if metadata.Images.Count > 0 && not (isNull metadata.Images.[0].AssociatedElements) then
                metadata.Images.[0].AssociatedElements :> XisfCoreElement seq
            else
                Seq.empty

        let allProps =
            let globalProps = metadata.GlobalProperties :> XisfProperty seq
            let imageProps =
                if metadata.Images.Count > 0 && not (isNull metadata.Images.[0].Properties) then
                    metadata.Images.[0].Properties :> XisfProperty seq
                else
                    Seq.empty
            Seq.append globalProps imageProps

        let object' =
            tryGetProperty "Observation:Object:Name" allProps
            |> Option.orElse (tryGetProperty "OBJECT" allProps)

        let filter =
            tryGetProperty "Instrument:Filter:Name" allProps
            |> Option.orElse (tryGetProperty "FILTER" allProps)

        let imageType =
            tryGetFitsKeyword "IMAGETYP" allElements
            |> Option.map (fun v -> v.Trim('\'', ' ').ToUpper())

        let exposureTime =
            tryGetNumericProperty "Instrument:ExposureTime" allProps

        let observationDate =
            tryGetFitsKeyword "DATE-OBS" allElements
            |> Option.bind tryParseDate

        let sideOfPier =
            tryGetProperty "Instrument:Telescope:SideOfPier" allProps
            |> Option.orElse (tryGetFitsKeyword "PIERSIDE" allElements |> Option.map (fun v -> v.Trim('\'', ' ')))

        (object', filter, imageType, exposureTime, observationDate, sideOfPier)

// ==================== Image Analysis Functions ====================

let analyzeImageMetadata (filePath: string) : Async<AnalysisResult> =
    async {
        try
            let fileName = Path.GetFileName(filePath)
            let fileInfo = FileInfo(filePath)
            let fileSize = fileInfo.Length

            let reader = new XisfReader()
            let metadata = reader.ReadMetadataFromFile(filePath)

            if metadata.Images.Count = 0 then
                return { FilePath = filePath; Stats = None; Error = Some "No images found in file" }
            else
                let img = metadata.Images.[0]
                let (object', filter, imageType, exposureTime, observationDate, sideOfPier) =
                    MetadataExtraction.extractFromMetadata metadata

                let compression =
                    match img.DataBlockInfo.Compression with
                    | null -> "none"
                    | comp -> comp.ToString()

                let stats = {
                    FileName = fileName
                    Width = int img.Geometry.Width
                    Height = int img.Geometry.Height
                    Channels = int img.Geometry.ChannelCount
                    ChannelStats = [||]  // No pixel statistics in header-only mode
                    FileSize = fileSize
                    Compression = compression
                    Object = object'
                    Filter = filter
                    ImageType = imageType
                    ExposureTime = exposureTime
                    ObservationDate = observationDate
                    SideOfPier = sideOfPier
                    StarCount = None
                    MedianFWHM = None
                }
                return { FilePath = filePath; Stats = Some stats; Error = None }
        with ex ->
            return { FilePath = filePath; Stats = None; Error = Some ex.Message }
    }

let analyzeImage (filePath: string) (metricsLevel: MetricsLevel) (detectStars: bool) : Async<AnalysisResult> =
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

            // Extract additional metadata (exposure, date, pier side)
            let allProps =
                let globalProps = unit.GlobalProperties :> XisfProperty seq
                let imageProps =
                    if not (isNull img.Properties) then
                        img.Properties :> XisfProperty seq
                    else
                        Seq.empty
                Seq.append globalProps imageProps

            let exposureTime = MetadataExtraction.tryGetNumericProperty "Instrument:ExposureTime" allProps

            let observationDate =
                extractFitsKeyword img "DATE-OBS"
                |> Option.bind (fun v ->
                    match DateTime.TryParse(v.Trim('\'', ' ')) with
                    | true, d -> Some d
                    | false, _ -> None)

            let sideOfPier =
                MetadataExtraction.tryGetProperty "Instrument:Telescope:SideOfPier" allProps
                |> Option.orElse (extractFitsKeyword img "PIERSIDE")

            let compression =
                if img.PixelData :? InlineDataBlock then
                    let block = img.PixelData :?> InlineDataBlock
                    match block.Compression with
                    | null -> "none"
                    | comp -> comp.ToString()
                else
                    "embedded"

            // Read pixels using PixelIO (handles all sample formats)
            let pixelFloats = PixelIO.readPixelsAsFloat img

            let width = int img.Geometry.Width
            let height = int img.Geometry.Height
            let channels = int img.Geometry.ChannelCount
            let pixelCount = width * height

            let channelStats =
                Array.init channels (fun ch ->
                    let values = Array.init pixelCount (fun pix ->
                        pixelFloats.[pix * channels + ch]
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
                                    pixelFloats.[pix * channels + ch]
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

            let stats = {
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
                ExposureTime = exposureTime
                ObservationDate = observationDate
                SideOfPier = sideOfPier
                StarCount = starCount
                MedianFWHM = medianFWHM
            }
            return { FilePath = filePath; Stats = Some stats; Error = None }
        with ex ->
            return { FilePath = filePath; Stats = None; Error = Some ex.Message }
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
        | ByExposure -> fun a b ->
            let expA = a.ExposureTime |> Option.defaultValue 0.0
            let expB = b.ExposureTime |> Option.defaultValue 0.0
            compare expA expB
        | ByDate -> fun a b ->
            let dateA = a.ObservationDate |> Option.defaultValue DateTime.MinValue
            let dateB = b.ObservationDate |> Option.defaultValue DateTime.MinValue
            compare dateA dateB

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

type GroupAggregate = {
    FrameCount: int
    TotalIntegration: float
    AverageExposure: float
    DateRange: (DateTime * DateTime) option
    TotalFileSize: int64
}

let calculateGroupAggregate (stats: ImageStats[]) : GroupAggregate =
    let frameCount = stats.Length

    let totalIntegration =
        stats
        |> Array.choose (fun s -> s.ExposureTime)
        |> Array.sum

    let averageExposure =
        if frameCount > 0 then totalIntegration / float frameCount
        else 0.0

    let dates =
        stats
        |> Array.choose (fun s -> s.ObservationDate)
        |> Array.sort

    let dateRange =
        if dates.Length > 0 then
            Some (dates.[0], dates.[dates.Length - 1])
        else
            None

    let totalFileSize =
        stats |> Array.sumBy (fun s -> s.FileSize)

    {
        FrameCount = frameCount
        TotalIntegration = totalIntegration
        AverageExposure = averageExposure
        DateRange = dateRange
        TotalFileSize = totalFileSize
    }

let formatDuration (seconds: float) =
    let totalMinutes = int (seconds / 60.0)
    let hours = totalMinutes / 60
    let minutes = totalMinutes % 60

    if hours > 0 then
        sprintf "%dh %dm" hours minutes
    else
        sprintf "%dm" minutes

let formatDateRange (dateRange: (DateTime * DateTime) option) =
    match dateRange with
    | None -> "No dates"
    | Some (first, last) when first = last ->
        first.ToString("yyyy-MM-dd")
    | Some (first, last) ->
        sprintf "%s to %s" (first.ToString("yyyy-MM-dd")) (last.ToString("yyyy-MM-dd"))

let printStatsHeaderOnly (stats: ImageStats[]) (groupBy: GroupBy) (sortBy: SortBy) (sortOrder: SortOrder) =
    printfn ""
    printfn "Image Statistics (Header-Only Mode):"
    printfn "================================================================================"

    let groups = groupStats stats groupBy
    let sortedGroups =
        groups |> Array.map (fun (groupName, groupStats) ->
            (groupName, sortStats groupStats sortBy sortOrder))

    for (groupName, groupStats) in sortedGroups do
        let aggregate = calculateGroupAggregate groupStats

        printfn ""
        printfn "================================================================================"
        printfn "GROUP: %s" groupName
        printfn "================================================================================"
        printfn "  Frames:             %d" aggregate.FrameCount
        printfn "  Total Integration:  %s (%.0f seconds)" (formatDuration aggregate.TotalIntegration) aggregate.TotalIntegration
        printfn "  Average Exposure:   %.2f seconds" aggregate.AverageExposure
        printfn "  Date Range:         %s" (formatDateRange aggregate.DateRange)
        printfn "  Total Size:         %s" (formatFileSize aggregate.TotalFileSize)

        if groupBy = NoGrouping then
            printfn ""
            printfn "Files:"
            printfn "--------------------------------------------------------------------------------"
            for imgStats in groupStats do
                let exposure = imgStats.ExposureTime |> Option.map (sprintf "%.0fs") |> Option.defaultValue "-"
                let date = imgStats.ObservationDate |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm")) |> Option.defaultValue "-"
                let size = formatFileSize imgStats.FileSize
                printfn "  %-60s  %8s  %16s  %10s" imgStats.FileName exposure date size

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

    let isHeaderOnly =
        stats.Length > 0 && stats.[0].ChannelStats.Length = 0

    let hasAdvanced =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].MAD.IsSome

    let hasHistogram =
        stats.Length > 0 &&
        stats.[0].ChannelStats.Length > 0 &&
        stats.[0].ChannelStats.[0].Histogram.IsSome

    let hasFitsMetadata =
        stats.Length > 0 &&
        (stats.[0].Object.IsSome || stats.[0].Filter.IsSome || stats.[0].ImageType.IsSome)

    let hasStarData =
        stats.Length > 0 &&
        (stats.[0].StarCount.IsSome || stats.[0].MedianFWHM.IsSome)

    if isHeaderOnly then
        let metadata = if hasFitsMetadata then ",Object,Filter,ImageType" else ""
        let header = sprintf "FileName,Width,Height,Channels,FileSize,Compression%s,ExposureTime,ObservationDate,SideOfPier" metadata
        writer.WriteLine(header)

        for imgStats in stats do
            let metadataStr =
                if hasFitsMetadata then
                    let obj = imgStats.Object |> Option.defaultValue ""
                    let flt = imgStats.Filter |> Option.defaultValue ""
                    let typ = imgStats.ImageType |> Option.defaultValue ""
                    sprintf ",%s,%s,%s" obj flt typ
                else ""

            let exposure = imgStats.ExposureTime |> Option.map string |> Option.defaultValue ""
            let obsDate = imgStats.ObservationDate |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm:ss")) |> Option.defaultValue ""
            let pier = imgStats.SideOfPier |> Option.defaultValue ""

            writer.WriteLine(sprintf "%s,%d,%d,%d,%d,%s%s,%s,%s,%s"
                imgStats.FileName
                imgStats.Width
                imgStats.Height
                imgStats.Channels
                imgStats.FileSize
                imgStats.Compression
                metadataStr
                exposure
                obsDate
                pier)
    else
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
                let opts = parseArgs args

                let inputDir = Path.GetDirectoryName(opts.Input)
                let pattern = Path.GetFileName(opts.Input)
                let actualDir = if String.IsNullOrEmpty(inputDir) then "." else inputDir

                if not (Directory.Exists(actualDir)) then
                    Log.Error($"Input directory not found: {actualDir}")
                    return 1
                else
                    let files = Directory.GetFiles(actualDir, pattern) |> Array.sort

                    if files.Length = 0 then
                        Log.Error($"No files found matching pattern: {opts.Input}")
                        return 1
                    else
                        let plural = if files.Length = 1 then "" else "s"
                        let mode = if opts.HeaderOnly then "(header-only mode)" else ""
                        printfn $"Found {files.Length} file{plural} to analyze {mode}"

                        let! results =
                            if opts.HeaderOnly then
                                files
                                |> Array.map analyzeImageMetadata
                                |> Async.Parallel
                            else
                                files
                                |> Array.map (fun f -> analyzeImage f opts.Metrics opts.DetectStars)
                                |> Async.Parallel

                        let allStats = results |> Array.choose (fun r -> r.Stats)
                        let failures = results |> Array.filter (fun r -> r.Error.IsSome)

                        if allStats.Length = 0 then
                            Log.Error("No images could be analyzed")
                            return 1
                        else
                            // Apply filters
                            let filteredStats = applyFilters allStats opts.MinMedian opts.MaxMad opts.MinSnr

                            if filteredStats.Length = 0 then
                                Log.Warning("No images passed filtering criteria")
                                return 1
                            else
                                if filteredStats.Length < allStats.Length then
                                    let filtered = allStats.Length - filteredStats.Length
                                    let filteredPlural = if filtered = 1 then "" else "s"
                                    printfn $"Filtered out {filtered} file{filteredPlural} based on quality thresholds"

                                if opts.HeaderOnly then
                                    printStatsHeaderOnly filteredStats opts.GroupBy opts.SortBy opts.SortOrder
                                else
                                    printStats filteredStats opts.GroupBy opts.SortBy opts.SortOrder

                                match opts.Output with
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

                                if failures.Length > 0 then
                                    printfn ""
                                    printfn "================================================================================"
                                    let failPlural = if failures.Length = 1 then "" else "s"
                                    printfn $"Failed to process {failures.Length} file{failPlural}:"
                                    printfn "================================================================================"
                                    for failure in failures do
                                        let fileName = Path.GetFileName(failure.FilePath)
                                        let error = failure.Error |> Option.defaultValue "Unknown error"
                                        printfn $"  - {fileName}: {error}"

                                return if allStats.Length = files.Length then 0 else 1
            with ex ->
                Log.Error($"Error: {ex.Message}")
                printfn ""
                printfn "Run 'xisfprep stats --help' for usage information"
                return 1
        }

        Async.RunSynchronously computation
