module Algorithms.HotPixelCorrection

open System
open Serilog
open FsToolkit.ErrorHandling
open Algorithms.Statistics
open Algorithms.DirectionalGradients
open Algorithms.Debayering

type AsyncResult<'T, 'E> = Async<Result<'T, 'E>>

type StarProtectionMethod =
    | NoProtection
    | Isolation
    | Ratio

type DetectionConfig = {
    KHot: float
    KCold: float
    TileSize: int
    Overlap: int
    BayerPattern: BayerPattern option
    StarProtection: StarProtectionMethod
    StarProtectionRatio: float
}

type TileInfo = {
    X: int
    Y: int
    Width: int
    Height: int
}

type DetectionResult = {
    HotPixelCount: int
    ColdPixelCount: int
    TotalCorrected: int
}

type HotPixelError =
    | ImageTooSmall of width: int * height: int
    | InvalidConfig of message: string

    override this.ToString() =
        match this with
        | ImageTooSmall (w, h) -> $"Image too small: {w}×{h} (minimum: 3×3 for correction)"
        | InvalidConfig msg -> $"Invalid configuration: {msg}"

let VNG_BORDER = 2
let GLOBAL_OUTLIER_SIGMA = 8.0

let generateTileGrid (width: int) (height: int) (tileSize: int) (overlap: int) : TileInfo[] =
    if width < tileSize && height < tileSize then
        [| { X = 0; Y = 0; Width = width; Height = height } |]
    else
        let step = tileSize - overlap
        let tilesX = max 1 ((width + step - 1) / step)
        let tilesY = max 1 ((height + step - 1) / step)

        [| for ty in 0 .. tilesY - 1 do
             for tx in 0 .. tilesX - 1 do
                 let x = tx * step
                 let y = ty * step
                 let w = min tileSize (width - x)
                 let h = min tileSize (height - y)
                 { X = x; Y = y; Width = w; Height = h } |]

let computeGlobalStats
    (pixels: float[])
    (width: int)
    (height: int)
    (pattern: BayerPattern option) : (float * float)[] =

    match pattern with
    | None ->
        let median = calculateMedian pixels
        let mad = calculateMAD pixels median
        [| (median, mad) |]
    | Some pat ->
        [| for colorValue in 0 .. 3 do
             let colorPixels =
                 [| for y in 0 .. height - 1 do
                      for x in 0 .. width - 1 do
                          let pixelColor = getColor x y pat
                          if int pixelColor = colorValue then
                              pixels.[y * width + x] |]
             let median = calculateMedian colorPixels
             let mad = calculateMAD colorPixels median
             (median, mad) |]

let computeTileStats
    (pixels: float[])
    (width: int)
    (tile: TileInfo)
    (pattern: BayerPattern option) : (float * float)[] =

    match pattern with
    | None ->
        let tilePixels =
            [| for y in tile.Y .. tile.Y + tile.Height - 1 do
                 for x in tile.X .. tile.X + tile.Width - 1 do
                     pixels.[y * width + x] |]
        let median = calculateMedian tilePixels
        let mad = calculateMAD tilePixels median
        [| (median, mad) |]
    | Some pat ->
        [| for colorValue in 0 .. 3 do
             let colorPixels =
                 [| for y in tile.Y .. tile.Y + tile.Height - 1 do
                      for x in tile.X .. tile.X + tile.Width - 1 do
                          let pixelColor = getColor x y pat
                          if int pixelColor = colorValue then
                              pixels.[y * width + x] |]
             if colorPixels.Length > 0 then
                 let median = calculateMedian colorPixels
                 let mad = calculateMAD colorPixels median
                 (median, mad)
             else
                 (0.0, 0.0) |]

let isDefectivePixel
    (pixel: float)
    (localMedian: float)
    (localMAD: float)
    (globalMedian: float)
    (globalMAD: float)
    (kHot: float)
    (kCold: float) : bool =

    let localHot = pixel > localMedian + kHot * localMAD
    let localCold = pixel < localMedian - kCold * localMAD
    let globalExtreme = abs (pixel - globalMedian) > GLOBAL_OUTLIER_SIGMA * globalMAD

    (localHot || localCold) || globalExtreme

let isIsolatedDefect
    (pixels: float[])
    (width: int)
    (height: int)
    (x: int)
    (y: int)
    (localMedian: float)
    (localMAD: float)
    (kHot: float) : bool =

    let threshold = localMedian + kHot * localMAD
    let mutable brightNeighbors = 0

    for dy in -1 .. 1 do
        for dx in -1 .. 1 do
            if dx <> 0 || dy <> 0 then
                let nx = x + dx
                let ny = y + dy
                if nx >= 0 && nx < width && ny >= 0 && ny < height then
                    let neighborVal = pixels.[ny * width + nx]
                    if neighborVal > threshold then
                        brightNeighbors <- brightNeighbors + 1

    // Isolated if ≤1 bright neighbor (allows for 1 adjacent hot pixel)
    brightNeighbors <= 1

let hasHotPixelRatio
    (pixels: float[])
    (width: int)
    (height: int)
    (x: int)
    (y: int)
    (pixel: float)
    (minRatio: float) : bool =

    let neighbors = ResizeArray<float>()

    for dy in -1 .. 1 do
        for dx in -1 .. 1 do
            if dx <> 0 || dy <> 0 then
                let nx = x + dx
                let ny = y + dy
                if nx >= 0 && nx < width && ny >= 0 && ny < height then
                    neighbors.Add(pixels.[ny * width + nx])

    if neighbors.Count > 0 then
        let medianNeighbor = calculateMedian (neighbors.ToArray())
        if medianNeighbor > 0.0 then
            let ratio = pixel / medianNeighbor
            // Hot pixel if ratio exceeds threshold (e.g., 4.0 means 4x brighter than neighbors)
            ratio >= minRatio
        else
            true // If neighbors are zero/negative, treat as potential hot pixel
    else
        true // Edge case: no neighbors, assume hot pixel

let getNeighborOffsetsInDirection (pattern: BayerPattern option) (dir: Direction) : (int * int)[] =
    match pattern with
    | None ->
        // Monochrome: immediate neighbor in this direction (distance-1)
        match dir with
        | N  -> [| (0, -1) |]
        | NE -> [| (1, -1) |]
        | E  -> [| (1, 0) |]
        | SE -> [| (1, 1) |]
        | S  -> [| (0, 1) |]
        | SW -> [| (-1, 1) |]
        | W  -> [| (-1, 0) |]
        | NW -> [| (-1, -1) |]
    | Some _ ->
        // Bayer: same-color neighbor in this direction (distance-2, skip opposite color)
        match dir with
        | N  -> [| (0, -2) |]
        | NE -> [| (2, -2) |]
        | E  -> [| (2, 0) |]
        | SE -> [| (2, 2) |]
        | S  -> [| (0, 2) |]
        | SW -> [| (-2, 2) |]
        | W  -> [| (-2, 0) |]
        | NW -> [| (-2, -2) |]

let correctPixelVNG
    (pixels: float[])
    (width: int)
    (height: int)
    (x: int)
    (y: int)
    (pattern: BayerPattern option) : float =

    let getPixel px py =
        let px' = max 0 (min (width - 1) px)
        let py' = max 0 (min (height - 1) py)
        pixels.[py' * width + px']

    let gradients = allDirections |> Array.map (fun dir -> (dir, computeGradient getPixel x y dir))
    let validDirs = selectSmoothDirections gradients

    let neighborOffsets =
        validDirs |> Array.collect (getNeighborOffsetsInDirection pattern)

    let neighborValues =
        neighborOffsets
        |> Array.map (fun (dx, dy) -> getPixel (x + dx) (y + dy))

    if neighborValues.Length > 0 then
        Array.average neighborValues
    else
        allDirections
        |> Array.collect (getNeighborOffsetsInDirection pattern)
        |> Array.map (fun (dx, dy) -> getPixel (x + dx) (y + dy))
        |> Array.average

let correctPixelMedian
    (pixels: float[])
    (width: int)
    (height: int)
    (x: int)
    (y: int)
    (pattern: BayerPattern option) : float =

    let getPixel px py =
        let px' = max 0 (min (width - 1) px)
        let py' = max 0 (min (height - 1) py)
        pixels.[py' * width + px']

    let neighbors =
        match pattern with
        | None ->
            [| for dy in -1 .. 1 do
                 for dx in -1 .. 1 do
                     if dx <> 0 || dy <> 0 then
                         getPixel (x + dx) (y + dy) |]
        | Some _ ->
            allDirections
            |> Array.collect (getNeighborOffsetsInDirection pattern)
            |> Array.map (fun (dx, dy) -> getPixel (x + dx) (y + dy))

    if neighbors.Length > 0 then
        calculateMedian neighbors
    else
        pixels.[y * width + x]

let detectAndCorrectTile
    (pixels: float[])
    (width: int)
    (height: int)
    (tile: TileInfo)
    (config: DetectionConfig)
    (globalStats: (float * float)[]) : (int * float)[] =

    let localStats = computeTileStats pixels width tile config.BayerPattern

    let corrections = ResizeArray<int * float>()

    for y in tile.Y .. tile.Y + tile.Height - 1 do
        for x in tile.X .. tile.X + tile.Width - 1 do
            let idx = y * width + x
            let pixel = pixels.[idx]

            let colorIdx =
                match config.BayerPattern with
                | None -> 0
                | Some pat -> int (getColor x y pat)

            let (localMedian, localMAD) = localStats.[colorIdx]
            let (globalMedian, globalMAD) = globalStats.[colorIdx]

            if isDefectivePixel pixel localMedian localMAD globalMedian globalMAD config.KHot config.KCold then
                // Apply star protection filter
                let shouldCorrect =
                    match config.StarProtection with
                    | NoProtection -> true
                    | Isolation ->
                        isIsolatedDefect pixels width height x y localMedian localMAD config.KHot
                    | Ratio ->
                        hasHotPixelRatio pixels width height x y pixel config.StarProtectionRatio

                if shouldCorrect then
                    let corrected =
                        if x >= VNG_BORDER && x < width - VNG_BORDER &&
                           y >= VNG_BORDER && y < height - VNG_BORDER then
                            correctPixelVNG pixels width height x y config.BayerPattern
                        else
                            correctPixelMedian pixels width height x y config.BayerPattern

                    corrections.Add((idx, corrected))

    corrections.ToArray()

let processImage
    (pixels: float[])
    (width: int)
    (height: int)
    (config: DetectionConfig) : Result<float[] * DetectionResult, HotPixelError> =

    result {
        if width < 3 || height < 3 then
            return! Error (ImageTooSmall (width, height))

        Log.Verbose("Computing global statistics...")
        let globalStats = computeGlobalStats pixels width height config.BayerPattern

        Log.Verbose("Generating tile grid: {TileSize}×{TileSize}, overlap: {Overlap}",
                    config.TileSize, config.TileSize, config.Overlap)
        let tiles = generateTileGrid width height config.TileSize config.Overlap

        Log.Verbose("Processing {TileCount} tiles...", tiles.Length)

        let correctionSum = Array.zeroCreate pixels.Length
        let correctionCount = Array.zeroCreate pixels.Length

        for tile in tiles do
            let tileCorrections = detectAndCorrectTile pixels width height tile config globalStats

            for (idx, correctedValue) in tileCorrections do
                correctionSum.[idx] <- correctionSum.[idx] + correctedValue
                correctionCount.[idx] <- correctionCount.[idx] + 1

        let correctedPixels = Array.copy pixels
        let mutable hotCount = 0
        let mutable coldCount = 0

        for i in 0 .. pixels.Length - 1 do
            if correctionCount.[i] > 0 then
                let avgCorrection = correctionSum.[i] / float correctionCount.[i]
                if avgCorrection > correctedPixels.[i] then
                    coldCount <- coldCount + 1
                else
                    hotCount <- hotCount + 1
                correctedPixels.[i] <- avgCorrection

        let result = {
            HotPixelCount = hotCount
            ColdPixelCount = coldCount
            TotalCorrected = hotCount + coldCount
        }

        Log.Information("Corrected {Total} pixels ({Hot} hot, {Cold} cold)",
                        result.TotalCorrected, result.HotPixelCount, result.ColdPixelCount)

        return (correctedPixels, result)
    }
