# Image Transformation Design

Technical design for the `align` output mode transformation step.

## Overview

The transformation step applies a computed similarity transform to resample target images into the reference frame. This is the final piece of the alignment pipeline after star detection and matching.

## Current State

### What's Implemented
- **Star Detection** (`StarDetection.fs`): Connected component analysis, PSF measurement
- **Matching Algorithms**:
  - Triangle ratio matching (`TriangleMatch.fs`)
  - Center-seeded expanding match with KD-tree and RANSAC (`SpatialMatch.fs`)
- **Transform Estimation**: Similarity transform (scale, rotation, translation) with RANSAC outlier rejection
- **Output Modes**: `detect` and `match` visualization modes working

### What's Needed
- Pixel resampling using the inverse transform
- Interpolation kernels (nearest, bilinear, bicubic)
- Boundary handling
- Per-file transformation processing in `Align` output mode

## Architecture

### Module Structure

```
Algorithms/
├── StarDetection.fs      # Star detection (existing)
├── SpatialMatch.fs       # Matching + transforms (existing)
├── TriangleMatch.fs      # Triangle matching (existing)
└── Interpolation.fs      # NEW: Resampling kernels

Commands/
└── Align.fs              # Add processAlignFile function
```

### Type Definitions

The existing `SimilarityTransform` type in `SpatialMatch.fs` is sufficient:

```fsharp
type SimilarityTransform = {
    A: float      // cos(θ) * scale
    B: float      // sin(θ) * scale
    Dx: float     // X translation
    Dy: float     // Y translation
    Scale: float
    Rotation: float  // degrees
}
```

The `InterpolationMethod` enum in `Align.fs` defines available methods:

```fsharp
type InterpolationMethod =
    | Nearest
    | Bilinear
    | Bicubic
```

## Coordinate System

### Transform Direction

The transform maps **target → reference**:
```
Reference = Transform(Target)
```

For resampling, we need the inverse:
```
Target = InverseTransform(Reference)
```

This is already provided by `invertTransform` in `SpatialMatch.fs`.

### Pixel Sampling Model

For each output pixel `(ox, oy)`:
1. Apply inverse transform to get target coordinates `(tx, ty)`
2. Sample target image at `(tx, ty)` using interpolation
3. Handle out-of-bounds coordinates (return 0 or edge extend)

## Interpolation Module

### New File: `Algorithms/Interpolation.fs`

```fsharp
module Algorithms.Interpolation

/// Sample a channel at fractional coordinates
/// pixels: single-channel float array
/// Returns None if outside bounds (or extend edge)
val sampleNearest : pixels:float[] -> width:int -> height:int -> x:float -> y:float -> float option
val sampleBilinear : pixels:float[] -> width:int -> height:int -> x:float -> y:float -> float option
val sampleBicubic : pixels:float[] -> width:int -> height:int -> x:float -> y:float -> float option

/// Unified sampling interface
val sample : InterpolationMethod -> pixels:float[] -> width:int -> height:int -> x:float -> y:float -> float option
```

### Interpolation Algorithms

#### Nearest Neighbor
```fsharp
let sampleNearest (pixels: float[]) width height x y =
    let ix = int (round x)
    let iy = int (round y)
    if ix >= 0 && ix < width && iy >= 0 && iy < height then
        Some pixels.[iy * width + ix]
    else
        None
```

#### Bilinear
```fsharp
let sampleBilinear (pixels: float[]) width height x y =
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
```

#### Bicubic (Catmull-Rom)
Uses a 4x4 kernel with cubic weights:
```fsharp
let private cubicWeight t =
    let at = abs t
    if at < 1.0 then
        (3.0 * at - 5.0) * at * at + 2.0
    elif at < 2.0 then
        ((-at + 5.0) * at - 8.0) * at + 4.0
    else
        0.0
    |> (*) 0.5

let sampleBicubic (pixels: float[]) width height x y =
    let x0 = int (floor x)
    let y0 = int (floor y)

    // Need 2 pixels on each side
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
```

### Boundary Handling

Two strategies:

1. **Black fill (default)**: Return 0.0 for out-of-bounds pixels
2. **Edge extend**: Clamp coordinates to valid range

For astronomical images, black fill is typically preferred to avoid introducing artifacts at edges.

## Transformation Process

### Per-File Processing

Add `processAlignFile` function in `Align.fs`:

```fsharp
let processAlignFile
    (inputPath: string) (outputDir: string) (suffix: string) (overwrite: bool)
    (outputFormat: XisfSampleFormat option) (detectionParams: DetectionParams)
    (refStars: DetectedStar[]) (imageWidth: int) (imageHeight: int)
    (opts: AlignOptions)
    : Async<bool> =
    async {
        // 1. Read target image
        // 2. Detect stars in target
        // 3. Run matching to get transform
        // 4. Apply inverse transform to resample
        // 5. Write output
    }
```

### Resampling Loop

The core transformation:

```fsharp
let transformImage
    (targetPixels: float[])
    (width: int)
    (height: int)
    (channels: int)
    (transform: SimilarityTransform)
    (interpolation: InterpolationMethod)
    : float[] =

    let inverse = invertTransform transform
    let outputPixels = Array.zeroCreate (width * height * channels)

    // Process in parallel for performance
    Array.Parallel.init (width * height) (fun pixIdx ->
        let ox = pixIdx % width
        let oy = pixIdx / width

        // Transform output coords to target coords
        let (tx, ty) = applyTransform inverse (float ox) (float oy)

        // Sample each channel
        for ch in 0 .. channels - 1 do
            let channelOffset = ch * width * height
            let channelPixels =
                if channels = 1 then targetPixels
                else Array.init (width * height) (fun i -> targetPixels.[i * channels + ch])

            let value =
                match sample interpolation channelPixels width height tx ty with
                | Some v -> v
                | None -> 0.0  // Black fill for out-of-bounds

            outputPixels.[pixIdx * channels + ch] <- value
    ) |> ignore

    outputPixels
```

## Output Headers

Add FITS keywords documenting the applied transform:

```fsharp
let createAlignHeaders (transform: SimilarityTransform) (matchCount: int) =
    [|
        XisfFitsKeyword("ALIGNED", "T", "Image has been aligned") :> XisfCoreElement
        XisfFitsKeyword("ALIGNREF", refFileName, "Reference frame") :> XisfCoreElement
        XisfFitsKeyword("XSHIFT", sprintf "%.2f" transform.Dx, "X shift (pixels)") :> XisfCoreElement
        XisfFitsKeyword("YSHIFT", sprintf "%.2f" transform.Dy, "Y shift (pixels)") :> XisfCoreElement
        XisfFitsKeyword("ROTATION", sprintf "%.4f" transform.Rotation, "Rotation (degrees)") :> XisfCoreElement
        XisfFitsKeyword("SCALE", sprintf "%.6f" transform.Scale, "Scale factor") :> XisfCoreElement
        XisfFitsKeyword("STARMTCH", matchCount.ToString(), "Matched star pairs") :> XisfCoreElement
        XisfFitsKeyword("INTERP", interpolationName, "Interpolation method") :> XisfCoreElement
        XisfFitsKeyword("HISTORY", "", "Aligned by XisfPrep Align v1.0") :> XisfCoreElement
    |]
```

## Performance Considerations

### Parallelization

- Use `Array.Parallel.init` for the resampling loop (embarrassingly parallel)
- Continue using `Async.Parallel` for batch file processing

### Memory Layout

For bicubic interpolation, accessing a 4x4 neighborhood benefits from good cache locality. Consider processing in tiles for large images, though current implementation should be adequate for typical astronomical image sizes (4k-16k pixels).

### Channel Separation

For RGB images, extract channels before transformation to avoid repeated index calculations:

```fsharp
// Pre-extract channels once
let channelArrays =
    [| for ch in 0 .. channels - 1 ->
        Array.init (width * height) (fun i -> targetPixels.[i * channels + ch]) |]

// Then sample directly from channel arrays
```

## Future Extensions

### Distortion Correction (Phase 2)

The current similarity transform handles translation, rotation, and uniform scale. Real optical systems have field distortion. To add distortion correction:

1. **After similarity transform**: Apply polynomial or radial distortion model
2. **Types to add**:
   ```fsharp
   type DistortionModel =
       | None
       | Polynomial of coefficients: float[]
       | RadialTangential of k1:float * k2:float * p1:float * p2:float
   ```
3. **Modified pipeline**:
   ```
   Output → InverseSimilarity → InverseDistortion → Sample
   ```

### Advanced Interpolation (Phase 2)

- **Lanczos**: Better frequency response, less ringing than bicubic
- **Configurable kernel size**: Lanczos-2, Lanczos-3

### Sub-pixel Accuracy Verification

Implement quality metrics to verify alignment accuracy:
- Re-detect stars in output, compare positions to reference
- Compute RMS residual error

## Testing Strategy

### Unit Tests for Interpolation

1. **Identity transform**: Output should match input (with interpolation smoothing)
2. **Pure translation**: Integer shifts should be exact
3. **Known transforms**: Apply rotation, verify star positions

### Visual Validation

1. **Blink comparison**: Toggle between aligned and reference
2. **Difference image**: Subtract aligned from reference, should show minimal residual

### Edge Cases

- Images with few stars (< 10)
- Images with different dimensions than reference (error case)
- Saturated regions affecting interpolation

## Implementation Order

1. **Interpolation module** (`Algorithms/Interpolation.fs`)
   - Implement all three interpolation methods
   - Add unified `sample` function

2. **Transform function** in `Align.fs`
   - `transformImage` function
   - Channel handling for RGB

3. **Process align file**
   - `processAlignFile` function
   - Integrate with existing matching pipeline

4. **Wire up output mode**
   - Complete the `Align` case in `run` function
   - Add proper headers and metadata

5. **Testing and validation**
   - Test with real astronomical data
   - Verify star positions post-alignment

## Design Decisions

1. **Boundary handling**: Black fill (0) for out-of-bounds pixels - standard for astronomy

2. **Output format**: Default to float32 to preserve interpolation precision

3. **Console output**: Show transform parameters per file, keep it tidy:
   ```
   dx=1.2 dy=-0.8 rot=0.12° -> filename_a.xisf
   ```

4. **No MaxShift validation**: Removed as unnecessary scope creep - trust the RANSAC-refined transform
