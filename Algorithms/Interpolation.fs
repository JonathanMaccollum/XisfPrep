module Algorithms.Interpolation

open System

// Re-use interpolation method type from Align
// This avoids circular dependency - Align will pass the enum value

/// Sample using nearest neighbor interpolation
/// Returns None if outside bounds
let sampleNearest (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float option =
    let ix = int (round x)
    let iy = int (round y)
    if ix >= 0 && ix < width && iy >= 0 && iy < height then
        Some pixels.[iy * width + ix]
    else
        None

/// Sample using bilinear interpolation
/// Returns None if outside bounds (needs 2x2 neighborhood)
let sampleBilinear (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float option =
    let x0 = int (floor x)
    let y0 = int (floor y)
    let x1 = x0 + 1
    let y1 = y0 + 1

    if x0 >= 0 && x1 < width && y0 >= 0 && y1 < height then
        let fx = x - float x0
        let fy = y - float y0

        let v00 = pixels.[y0 * width + x0]
        let v10 = pixels.[y0 * width + x1]
        let v01 = pixels.[y1 * width + x0]
        let v11 = pixels.[y1 * width + x1]

        let v0 = v00 * (1.0 - fx) + v10 * fx
        let v1 = v01 * (1.0 - fx) + v11 * fx
        Some (v0 * (1.0 - fy) + v1 * fy)
    else
        None

/// Cubic interpolation weight (Catmull-Rom spline)
let private cubicWeight (t: float) : float =
    let at = abs t
    if at < 1.0 then
        ((3.0 * at - 5.0) * at * at + 2.0) * 0.5
    elif at < 2.0 then
        (((-at + 5.0) * at - 8.0) * at + 4.0) * 0.5
    else
        0.0

/// Sinc function: sin(πx)/(πx), with sinc(0) = 1
let private sinc (x: float) : float =
    if abs x < 1e-8 then 1.0
    else
        let px = System.Math.PI * x
        sin px / px

/// Lanczos kernel weight
let private lanczosWeight (x: float) (a: float) : float =
    let ax = abs x
    if ax < 1e-8 then 1.0
    elif ax < a then sinc x * sinc (x / a)
    else 0.0

/// Sample using bicubic interpolation (Catmull-Rom)
/// Returns None if outside bounds (needs 4x4 neighborhood)
let sampleBicubic (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float option =
    let x0 = int (floor x)
    let y0 = int (floor y)

    // Need 1 pixel before and 2 after the center
    if x0 >= 1 && x0 + 2 < width && y0 >= 1 && y0 + 2 < height then
        let fx = x - float x0
        let fy = y - float y0

        let mutable sum = 0.0
        for dy in -1 .. 2 do
            let wy = cubicWeight (fy - float dy)
            for dx in -1 .. 2 do
                let wx = cubicWeight (fx - float dx)
                let px = pixels.[(y0 + dy) * width + (x0 + dx)]
                sum <- sum + px * wx * wy
        Some sum
    else
        None

/// Sample using Lanczos-3 interpolation (6x6 kernel)
/// Returns None if outside bounds (needs 6x6 neighborhood)
let sampleLanczos3 (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float option =
    let x0 = int (floor x)
    let y0 = int (floor y)

    // Need 2 pixels before and 3 after the center (6x6 kernel)
    if x0 >= 2 && x0 + 3 < width && y0 >= 2 && y0 + 3 < height then
        let fx = x - float x0
        let fy = y - float y0

        let mutable sum = 0.0
        let mutable weightSum = 0.0

        for dy in -2 .. 3 do
            let wy = lanczosWeight (fy - float dy) 3.0
            for dx in -2 .. 3 do
                let wx = lanczosWeight (fx - float dx) 3.0
                let w = wx * wy
                let px = pixels.[(y0 + dy) * width + (x0 + dx)]
                sum <- sum + px * w
                weightSum <- weightSum + w

        // Normalize by weight sum to handle edge effects
        if weightSum > 1e-8 then Some (sum / weightSum)
        else Some sum
    else
        None

/// Interpolation method enumeration
type InterpolationMethod =
    | Nearest
    | Bilinear
    | Bicubic
    | Lanczos3

/// Unified sampling function
/// Returns 0.0 for out-of-bounds (black fill)
/// Clamps output to [0, 65535] to prevent interpolation ringing artifacts
let sample (method: InterpolationMethod) (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float =
    let result =
        match method with
        | Nearest -> sampleNearest pixels width height x y
        | Bilinear -> sampleBilinear pixels width height x y
        | Bicubic -> sampleBicubic pixels width height x y
        | Lanczos3 -> sampleLanczos3 pixels width height x y

    match result with
    | Some v -> max 0.0 (min 65535.0 v)  // Clamp to valid range
    | None -> 0.0  // Black fill for out-of-bounds

/// Sample with explicit boundary handling
let sampleWithBounds (method: InterpolationMethod) (pixels: float[]) (width: int) (height: int) (x: float) (y: float) : float option =
    match method with
    | Nearest -> sampleNearest pixels width height x y
    | Bilinear -> sampleBilinear pixels width height x y
    | Bicubic -> sampleBicubic pixels width height x y
    | Lanczos3 -> sampleLanczos3 pixels width height x y
