module Algorithms.StarDetection

open System

type DetectedStar = {
    X: float
    Y: float
    FWHM: float
    HFR: float
    Peak: float
    Flux: float
    Background: float
    SNR: float
    Eccentricity: float
    Saturated: bool
}

type DetectionParams = {
    Threshold: float
    GridSize: int
    MinFWHM: float
    MaxFWHM: float
    MaxEccentricity: float
    MaxStars: int option
}

type StarStatistics = {
    Channel: int
    Count: int
    MedianFWHM: float
    StdDevFWHM: float
    MedianHFR: float
    StdDevHFR: float
    MedianEccentricity: float
    StdDevEccentricity: float
    MedianSNR: float
    StdDevSNR: float
}

let defaultParams = {
    Threshold = 5.0
    GridSize = 128
    MinFWHM = 1.5
    MaxFWHM = 20.0
    MaxEccentricity = 0.5
    MaxStars = Some 5000
}

let estimateLocalBackground (values: float[]) width height gridSize =
    let gridWidth = (width + gridSize - 1) / gridSize
    let gridHeight = (height + gridSize - 1) / gridSize

    let gridMedians = Array2D.zeroCreate gridHeight gridWidth

    for gy in 0 .. gridHeight - 1 do
        for gx in 0 .. gridWidth - 1 do
            let startX = gx * gridSize
            let startY = gy * gridSize
            let endX = min (startX + gridSize) width
            let endY = min (startY + gridSize) height

            let blockValues = ResizeArray<float>()
            for y in startY .. endY - 1 do
                for x in startX .. endX - 1 do
                    blockValues.Add(values.[y * width + x])

            if blockValues.Count > 0 then
                let sorted = blockValues.ToArray()
                Array.sortInPlace sorted
                let mid = sorted.Length / 2
                let median =
                    if sorted.Length % 2 = 0 then
                        (sorted.[mid - 1] + sorted.[mid]) / 2.0
                    else
                        sorted.[mid]
                gridMedians.[gy, gx] <- median

    let backgroundMap = Array.zeroCreate (width * height)

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let gxf = float x / float gridSize
            let gyf = float y / float gridSize

            let gx0 = int gxf |> min (gridWidth - 1)
            let gy0 = int gyf |> min (gridHeight - 1)
            let gx1 = min (gx0 + 1) (gridWidth - 1)
            let gy1 = min (gy0 + 1) (gridHeight - 1)

            let tx = gxf - float gx0
            let ty = gyf - float gy0

            let v00 = gridMedians.[gy0, gx0]
            let v10 = gridMedians.[gy0, gx1]
            let v01 = gridMedians.[gy1, gx0]
            let v11 = gridMedians.[gy1, gx1]

            let v0 = v00 * (1.0 - tx) + v10 * tx
            let v1 = v01 * (1.0 - tx) + v11 * tx
            let interpolated = v0 * (1.0 - ty) + v1 * ty

            backgroundMap.[y * width + x] <- interpolated

    backgroundMap

type Component = {
    Id: int
    Pixels: ResizeArray<int * int>
}

let findConnectedComponents (residual: float[]) width height threshold =
    let labels = Array.create (width * height) 0
    let mutable nextLabel = 1
    let components = ResizeArray<Component>()

    let floodFill startX startY label =
        let comp = { Id = label; Pixels = ResizeArray() }
        let stack = ResizeArray<int * int>()
        stack.Add((startX, startY))

        while stack.Count > 0 do
            let (x, y) = stack.[stack.Count - 1]
            stack.RemoveAt(stack.Count - 1)

            if x >= 0 && x < width && y >= 0 && y < height then
                let idx = y * width + x
                if labels.[idx] = 0 && residual.[idx] > threshold then
                    labels.[idx] <- label
                    comp.Pixels.Add((x, y))

                    stack.Add((x - 1, y))
                    stack.Add((x + 1, y))
                    stack.Add((x, y - 1))
                    stack.Add((x, y + 1))

        if comp.Pixels.Count > 0 then
            components.Add(comp)

    for y in 0 .. height - 1 do
        for x in 0 .. width - 1 do
            let idx = y * width + x
            if labels.[idx] = 0 && residual.[idx] > threshold then
                floodFill x y nextLabel
                nextLabel <- nextLabel + 1

    components.ToArray()

let testConnectedComponents (values: float[]) width height mad threshold =
    let gridSize = 128
    let bgMap = estimateLocalBackground values width height gridSize

    let residual = Array.init (width * height) (fun i -> values.[i] - bgMap.[i])

    let thresholdValue = threshold * mad

    let components = findConnectedComponents residual width height thresholdValue

    let totalPixels = components |> Array.sumBy (fun c -> c.Pixels.Count)
    let avgSize = if components.Length > 0 then float totalPixels / float components.Length else 0.0

    let sortedBySizeDesc =
        components
        |> Array.sortByDescending (fun c -> c.Pixels.Count)
        |> Array.truncate (min 10 components.Length)

    printfn "Connected Components Detection:"
    printfn "  Threshold:        %.2f (%.1f * MAD)" thresholdValue threshold
    printfn "  Components:       %d" components.Length
    printfn "  Total Pixels:     %d" totalPixels
    printfn "  Avg Size:         %.1f pixels" avgSize
    printfn ""
    printfn "Top 10 components by size:"
    for c in sortedBySizeDesc do
        printfn "    Component %d: %d pixels" c.Id c.Pixels.Count
    printfn ""

    components

let measureStar (values: float[]) (bgMap: float[]) (comp: Component) width height =
    if comp.Pixels.Count = 0 then None
    else
        let pixels = comp.Pixels.ToArray()

        let mutable sumX = 0.0
        let mutable sumY = 0.0
        let mutable sumIntensity = 0.0
        let mutable peak = 0.0
        let mutable peakX = 0
        let mutable peakY = 0

        for (x, y) in pixels do
            let idx = y * width + x
            let intensity = values.[idx]
            sumX <- sumX + float x * intensity
            sumY <- sumY + float y * intensity
            sumIntensity <- sumIntensity + intensity
            if intensity > peak then
                peak <- intensity
                peakX <- x
                peakY <- y

        let cx = sumX / sumIntensity
        let cy = sumY / sumIntensity

        let background = bgMap.[peakY * width + peakX]

        let mutable sumDist2 = 0.0
        for (x, y) in pixels do
            let dx = float x - cx
            let dy = float y - cy
            let dist2 = dx * dx + dy * dy
            let idx = y * width + x
            let intensity = values.[idx]
            sumDist2 <- sumDist2 + dist2 * intensity

        let hfr = sqrt (sumDist2 / sumIntensity)

        let fwhm = 2.0 * hfr / 1.177

        let mutable sumXX = 0.0
        let mutable sumYY = 0.0
        let mutable sumXY = 0.0
        for (x, y) in pixels do
            let dx = float x - cx
            let dy = float y - cy
            let idx = y * width + x
            let intensity = values.[idx]
            sumXX <- sumXX + dx * dx * intensity
            sumYY <- sumYY + dy * dy * intensity
            sumXY <- sumXY + dx * dy * intensity

        let mxx = sumXX / sumIntensity
        let myy = sumYY / sumIntensity
        let mxy = sumXY / sumIntensity

        let trace = mxx + myy
        let det = mxx * myy - mxy * mxy
        let discriminant = trace * trace - 4.0 * det
        let eccentricity =
            if discriminant >= 0.0 && trace > 0.0 then
                let lambda1 = (trace + sqrt discriminant) / 2.0
                let lambda2 = (trace - sqrt discriminant) / 2.0
                if lambda1 > 0.0 then
                    1.0 - sqrt (lambda2 / lambda1)
                else
                    0.0
            else
                0.0

        let snr = if sumIntensity > 0.0 then sumIntensity / sqrt sumIntensity else 0.0

        let saturated = peak >= 65535.0

        Some {
            X = cx
            Y = cy
            FWHM = fwhm
            HFR = hfr
            Peak = peak
            Flux = sumIntensity
            Background = background
            SNR = snr
            Eccentricity = eccentricity
            Saturated = saturated
        }

let filterStar (star: DetectedStar) (settings: DetectionParams) =
    star.FWHM >= settings.MinFWHM &&
    star.FWHM <= settings.MaxFWHM &&
    star.Eccentricity <= settings.MaxEccentricity &&
    not star.Saturated

let testStarMeasurement (values: float[]) width height mad threshold =
    let gridSize = 128
    let bgMap = estimateLocalBackground values width height gridSize
    let residual = Array.init (width * height) (fun i -> values.[i] - bgMap.[i])
    let thresholdValue = threshold * mad
    let components = findConnectedComponents residual width height thresholdValue

    let minSize = 3
    let filteredComponents =
        components
        |> Array.filter (fun c -> c.Pixels.Count >= minSize)

    let stars =
        filteredComponents
        |> Array.choose (fun c -> measureStar values bgMap c width height)
        |> Array.filter (fun s -> filterStar s defaultParams)

    let sortedByFlux = stars |> Array.sortByDescending (fun s -> s.Flux) |> Array.truncate 20

    printfn "Star Measurement:"
    printfn "  Raw components:   %d" components.Length
    printfn "  Size filtered:    %d (>= %d pixels)" filteredComponents.Length minSize
    printfn "  Quality filtered: %d stars" stars.Length
    printfn ""
    printfn "Top 20 stars by flux:"
    printfn "  %6s  %8s  %8s  %6s  %6s  %6s  %8s  %5s" "X" "Y" "Flux" "FWHM" "HFR" "Eccen" "Peak" "SNR"
    for s in sortedByFlux do
        printfn "  %6.1f  %8.1f  %8.0f  %6.2f  %6.2f  %6.3f  %8.0f  %5.1f"
            s.X s.Y s.Flux s.FWHM s.HFR s.Eccentricity s.Peak s.SNR
    printfn ""

    if stars.Length > 0 then
        let fwhms = stars |> Array.map (fun s -> s.FWHM) |> Array.sort
        let hfrs = stars |> Array.map (fun s -> s.HFR) |> Array.sort
        let eccentricities = stars |> Array.map (fun s -> s.Eccentricity) |> Array.sort
        let snrs = stars |> Array.map (fun s -> s.SNR) |> Array.sort

        let median arr =
            let mid = Array.length arr / 2
            if Array.length arr % 2 = 0 then
                (arr.[mid - 1] + arr.[mid]) / 2.0
            else
                arr.[mid]

        let stdDev arr =
            let avg = Array.average arr
            sqrt ((arr |> Array.sumBy (fun x -> (x - avg) * (x - avg))) / float (Array.length arr))

        printfn "Summary Statistics:"
        printfn "  FWHM:         %.2f ± %.2f px (median: %.2f)" (Array.average fwhms) (stdDev fwhms) (median fwhms)
        printfn "  HFR:          %.2f ± %.2f px (median: %.2f)" (Array.average hfrs) (stdDev hfrs) (median hfrs)
        printfn "  Eccentricity: %.3f ± %.3f (median: %.3f)" (Array.average eccentricities) (stdDev eccentricities) (median eccentricities)
        printfn "  SNR:          %.1f ± %.1f (median: %.1f)" (Array.average snrs) (stdDev snrs) (median snrs)
        printfn ""

    stars

let testLocalBackground (values: float[]) width height =
    let gridSize = 128
    let bgMap = estimateLocalBackground values width height gridSize

    let globalMedian =
        let sorted = Array.copy values
        Array.sortInPlace sorted
        let mid = sorted.Length / 2
        if sorted.Length % 2 = 0 then
            (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else
            sorted.[mid]

    let avgBackground = Array.average bgMap
    let minBackground = Array.min bgMap
    let maxBackground = Array.max bgMap

    printfn "Local Background Map Statistics:"
    printfn "  Global Median:    %.2f" globalMedian
    printfn "  Avg Background:   %.2f" avgBackground
    printfn "  Min Background:   %.2f" minBackground
    printfn "  Max Background:   %.2f" maxBackground
    printfn "  Range:            %.2f" (maxBackground - minBackground)
    printfn ""

    bgMap

type ChannelStarResults = {
    Channel: int
    ChannelName: string
    Stars: DetectedStar[]
    Statistics: StarStatistics
}

type StarDetectionResults = {
    Channels: ChannelStarResults[]
    MatchedCount: int option
}

let computeStarStatistics (stars: DetectedStar[]) channel channelName =
    if stars.Length = 0 then
        { Channel = channel
          Count = 0
          MedianFWHM = 0.0
          StdDevFWHM = 0.0
          MedianHFR = 0.0
          StdDevHFR = 0.0
          MedianEccentricity = 0.0
          StdDevEccentricity = 0.0
          MedianSNR = 0.0
          StdDevSNR = 0.0 }
    else
        let fwhms = stars |> Array.map (fun s -> s.FWHM) |> Array.sort
        let hfrs = stars |> Array.map (fun s -> s.HFR) |> Array.sort
        let eccentricities = stars |> Array.map (fun s -> s.Eccentricity) |> Array.sort
        let snrs = stars |> Array.map (fun s -> s.SNR) |> Array.sort

        let median arr =
            let mid = Array.length arr / 2
            if Array.length arr % 2 = 0 then
                (arr.[mid - 1] + arr.[mid]) / 2.0
            else
                arr.[mid]

        let stdDev arr =
            let avg = Array.average arr
            sqrt ((arr |> Array.sumBy (fun x -> (x - avg) * (x - avg))) / float (Array.length arr))

        { Channel = channel
          Count = stars.Length
          MedianFWHM = median fwhms
          StdDevFWHM = stdDev fwhms
          MedianHFR = median hfrs
          StdDevHFR = stdDev hfrs
          MedianEccentricity = median eccentricities
          StdDevEccentricity = stdDev eccentricities
          MedianSNR = median snrs
          StdDevSNR = stdDev snrs }

let detectStarsInChannel (values: float[]) width height mad channel channelName (settings: DetectionParams) =
    let bgMap = estimateLocalBackground values width height settings.GridSize
    let residual = Array.init (width * height) (fun i -> values.[i] - bgMap.[i])
    let thresholdValue = settings.Threshold * mad
    let components = findConnectedComponents residual width height thresholdValue

    let minSize = 3
    let stars =
        components
        |> Array.filter (fun c -> c.Pixels.Count >= minSize)
        |> Array.choose (fun c -> measureStar values bgMap c width height)
        |> Array.filter (fun s -> filterStar s settings)

    let cappedStars =
        match settings.MaxStars with
        | Some max -> stars |> Array.sortByDescending (fun s -> s.Flux) |> Array.truncate max
        | None -> stars

    let statistics = computeStarStatistics cappedStars channel channelName

    { Channel = channel
      ChannelName = channelName
      Stars = cappedStars
      Statistics = statistics }

let matchStars (stars1: DetectedStar[]) (stars2: DetectedStar[]) tolerance =
    let mutable matchCount = 0
    for s1 in stars1 do
        let found = stars2 |> Array.exists (fun s2 ->
            let dx = s1.X - s2.X
            let dy = s1.Y - s2.Y
            let dist = sqrt (dx * dx + dy * dy)
            dist < tolerance
        )
        if found then
            matchCount <- matchCount + 1
    matchCount

let detectStars (channelValues: (float[] * float * int * string)[]) width height (settings: DetectionParams) =
    let channelResults =
        channelValues
        |> Array.map (fun (values, mad, channel, channelName) ->
            detectStarsInChannel values width height mad channel channelName settings
        )

    let matchedCount =
        if channelResults.Length > 1 then
            let baseStars = channelResults.[0].Stars
            let matchCounts =
                channelResults.[1..]
                |> Array.map (fun cr -> matchStars baseStars cr.Stars 1.5)
            Some (Array.min matchCounts)
        else
            None

    { Channels = channelResults
      MatchedCount = matchedCount }

let printStarDetectionResults (results: StarDetectionResults) =
    printfn ""
    printfn "Star Detection Results:"
    printfn "================================================================================"
    printfn ""

    for cr in results.Channels do
        let stats = cr.Statistics
        printfn "Channel: %s" cr.ChannelName
        printfn "  Detected:     %s stars" (stats.Count.ToString("N0"))
        printfn "  FWHM:         %.2f ± %.2f px (median: %.2f)" (stats.MedianFWHM) (stats.StdDevFWHM) (stats.MedianFWHM)
        printfn "  HFR:          %.2f ± %.2f px (median: %.2f)" (stats.MedianHFR) (stats.StdDevHFR) (stats.MedianHFR)
        printfn "  Eccentricity: %.3f ± %.3f (median: %.3f)" (stats.MedianEccentricity) (stats.StdDevEccentricity) (stats.MedianEccentricity)
        printfn "  SNR:          %.1f ± %.1f (median: %.1f)" (stats.MedianSNR) (stats.StdDevSNR) (stats.MedianSNR)
        printfn ""

    match results.MatchedCount with
    | Some count ->
        let baseCount = results.Channels.[0].Statistics.Count
        let percentage = if baseCount > 0 then (float count / float baseCount) * 100.0 else 0.0
        printfn "Cross-Matched: %s stars (%.1f%%)" (count.ToString("N0")) percentage
        printfn ""
    | None -> ()
