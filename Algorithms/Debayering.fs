module Algorithms.Debayering

open FsToolkit.ErrorHandling
open Algorithms.DirectionalGradients

// --- Types ---

/// Bayer pattern types
type BayerPattern =
    | RGGB
    | BGGR
    | GRBG
    | GBRG

    override this.ToString() =
        match this with
        | RGGB -> "RGGB"
        | BGGR -> "BGGR"
        | GRBG -> "GRBG"
        | GBRG -> "GBRG"

/// Debayering errors
type DebayeringError =
    | InvalidChannelCount of found: int
    | InvalidDimensions of width: int * height: int

    override this.ToString() =
        match this with
        | InvalidChannelCount found ->
            $"Cannot debayer: image has {found} channels (expected 1 for monochrome Bayer data)"
        | InvalidDimensions (w, h) ->
            $"Invalid image dimensions: {w}x{h}"

/// Configuration for debayering operation
type DebayeringConfig = {
    Pattern: BayerPattern
}

/// Result of debayering operation - three separate channel arrays
type DebayeringResult = {
    RedChannel: float[]
    GreenChannel: float[]
    BlueChannel: float[]
    Width: int
    Height: int
}

// --- Validation ---

/// Validate that image is monochrome (single channel)
let validateMonochrome (channels: int) : Result<unit, DebayeringError> =
    if channels <> 1 then
        Error (InvalidChannelCount channels)
    else
        Ok ()

/// Validate image dimensions
let validateDimensions (width: int) (height: int) : Result<unit, DebayeringError> =
    if width <= 0 || height <= 0 then
        Error (InvalidDimensions (width, height))
    else
        Ok ()

// --- Core Algorithm ---

// VNG kernel requires 2-pixel margin from image edges
let VNG_KERNEL_BORDER = 2

// Color channel indices
type Channel = R = 0 | G = 1 | B = 2

/// Determine which color a pixel position represents based on Bayer pattern
let getColor (x: int) (y: int) (pattern: BayerPattern) : Channel =
    let evenRow = y % 2 = 0
    let evenCol = x % 2 = 0
    match pattern with
    | RGGB ->
        if evenRow && evenCol then Channel.R
        elif evenRow && not evenCol then Channel.G
        elif not evenRow && evenCol then Channel.G
        else Channel.B
    | BGGR ->
        if evenRow && evenCol then Channel.B
        elif evenRow && not evenCol then Channel.G
        elif not evenRow && evenCol then Channel.G
        else Channel.R
    | GRBG ->
        if evenRow && evenCol then Channel.G
        elif evenRow && not evenCol then Channel.R
        elif not evenRow && evenCol then Channel.B
        else Channel.G
    | GBRG ->
        if evenRow && evenCol then Channel.G
        elif evenRow && not evenCol then Channel.B
        elif not evenRow && evenCol then Channel.R
        else Channel.G


/// Average values, with fallback to 0.0 if empty
let private average (values: float array) =
    if values.Length > 0 then Array.average values
    else 0.0

/// Get fallback average from array
let private getFallback (fallbackValues: float array) =
    if fallbackValues.Length > 0 then Array.average fallbackValues
    else 0.0

/// VNG (Variable Number of Gradients) interpolation for interior pixels
let private interpolateVNG (getPixel: int -> int -> float) (x: int) (y: int) (color: Channel) (value: float) (bayerPattern: BayerPattern) =
    let gradients = allDirections |> Array.map (fun dir -> (dir, computeGradient getPixel x y dir))
    let minGrad = gradients |> Array.map snd |> Array.min
    let maxGrad = gradients |> Array.map snd |> Array.max
    let threshold = minGrad + (maxGrad - minGrad) / 2.0
    let validDirs = gradients |> Array.filter (fun (_, g) -> g <= threshold) |> Array.map fst
    let validDirsSet = Set.ofArray validDirs

    match color with
    | Channel.R ->
        // Red pixel - interpolate G and B
        // G-neighbors are at N, S, E, W
        let gDirs = [| N; E; S; W |]
        let gFallbacks = gDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let gValues = gDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        // B-neighbors are at NE, SE, SW, NW
        let bDirs = [| NE; SE; SW; NW |]
        let bFallbacks = bDirs |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))
        let bValues = bDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))

        let g = if gValues.Length > 0 then average gValues else getFallback gFallbacks
        let b = if bValues.Length > 0 then average bValues else getFallback bFallbacks
        (value, g, b)

    | Channel.B ->
        // Blue pixel - interpolate G and R
        // G-neighbors are at N, S, E, W
        let gDirs = [| N; E; S; W |]
        let gFallbacks = gDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let gValues = gDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        // R-neighbors are at NE, SE, SW, NW
        let rDirs = [| NE; SE; SW; NW |]
        let rFallbacks = rDirs |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))
        let rValues = rDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with NE -> getPixel (x+1) (y-1) | SE -> getPixel (x+1) (y+1) | SW -> getPixel (x-1) (y+1) | _ -> getPixel (x-1) (y-1))

        let g = if gValues.Length > 0 then average gValues else getFallback gFallbacks
        let r = if rValues.Length > 0 then average rValues else getFallback rFallbacks
        (r, g, value)

    | Channel.G ->
        // Green pixel - interpolate R and B
        // Determine which neighbors are R and which are B based on Bayer pattern
        let (rDirs, bDirs) =
            match bayerPattern with
            // For RGGB, G on R-row (y%2=0) has R neighbors E/W, B neighbors N/S
            | RGGB -> if y % 2 = 0 then ([| E; W |], [| N; S |]) else ([| N; S |], [| E; W |])
            // For BGGR, G on B-row (y%2=0) has B neighbors E/W, R neighbors N/S
            | BGGR -> if y % 2 = 0 then ([| N; S |], [| E; W |]) else ([| E; W |], [| N; S |])
            // For GRBG, G on G-row (y%2=0) has R neighbors N/S, B neighbors E/W
            | GRBG -> if y % 2 = 0 then ([| N; S |], [| E; W |]) else ([| E; W |], [| N; S |])
            // For GBRG, G on G-row (y%2=0) has B neighbors N/S, R neighbors E/W
            | GBRG -> if y % 2 = 0 then ([| E; W |], [| N; S |]) else ([| N; S |], [| E; W |])

        let rFallbacks = rDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let rValues = rDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        let bFallbacks = bDirs |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)
        let bValues = bDirs |> Array.filter (fun d -> validDirsSet.Contains d) |> Array.map (fun dir -> match dir with N -> getPixel x (y-1) | E -> getPixel (x+1) y | S -> getPixel x (y+1) | _ -> getPixel (x-1) y)

        let r = if rValues.Length > 0 then average rValues else getFallback rFallbacks
        let b = if bValues.Length > 0 then average bValues else getFallback bFallbacks
        (r, value, b)

    | _ -> failwith "Invalid channel"

/// Debayer a Bayer mosaic image to three separate RGB channel arrays using VNG interpolation
let debayerPixels
    (pixels: float[])
    (width: int)
    (height: int)
    (config: DebayeringConfig) : Result<DebayeringResult, DebayeringError> =

    // Validation
    result {
        do! validateDimensions width height

        let pixelCount = width * height

        let getPixel x y =
            // Clamp coordinates to image bounds (replicate edge pixels)
            let x' = max 0 (min (width - 1) x)
            let y' = max 0 (min (height - 1) y)
            pixels.[y' * width + x']

        // Three separate output channels
        let redChannel = Array.zeroCreate pixelCount
        let greenChannel = Array.zeroCreate pixelCount
        let blueChannel = Array.zeroCreate pixelCount

        // Pass 1: VNG interpolation for interior pixels
        for y = VNG_KERNEL_BORDER to height - VNG_KERNEL_BORDER - 1 do
            for x = VNG_KERNEL_BORDER to width - VNG_KERNEL_BORDER - 1 do
                let idx = y * width + x
                let color = getColor x y config.Pattern
                let value = getPixel x y
                let r, g, b = interpolateVNG getPixel x y color value config.Pattern
                redChannel.[idx] <- r
                greenChannel.[idx] <- g
                blueChannel.[idx] <- b

        // Pass 2: Edge replication for borders (copy nearest interior VNG result)
        for y = 0 to height - 1 do
            for x = 0 to width - 1 do
                let isBorder = x < VNG_KERNEL_BORDER || x >= width - VNG_KERNEL_BORDER ||
                               y < VNG_KERNEL_BORDER || y >= height - VNG_KERNEL_BORDER
                if isBorder then
                    let idx = y * width + x
                    let srcX = max VNG_KERNEL_BORDER (min (width - VNG_KERNEL_BORDER - 1) x)
                    let srcY = max VNG_KERNEL_BORDER (min (height - VNG_KERNEL_BORDER - 1) y)
                    let srcIdx = srcY * width + srcX
                    redChannel.[idx] <- redChannel.[srcIdx]
                    greenChannel.[idx] <- greenChannel.[srcIdx]
                    blueChannel.[idx] <- blueChannel.[srcIdx]

        return {
            RedChannel = redChannel
            GreenChannel = greenChannel
            BlueChannel = blueChannel
            Width = width
            Height = height
        }
    }
