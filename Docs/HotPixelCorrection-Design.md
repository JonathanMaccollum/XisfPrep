# Hot/Cold Pixel Detection and Correction

## Overview

Tiled detection and correction of hot/cold pixels in linear astrophotography data using VNG-style directional interpolation. Operates on Bayer mosaic or debayered RGB/mono images.

**Key Features:**
- Local adaptive statistics (128×128 tiles with configurable overlap)
- VNG-based directional interpolation from same-color neighbors
- Bayer pattern aware (pre-debayer processing)
- Parallel file processing (tiles sequential per file)
- Railway-oriented error handling

---

## Command Interface

### Usage

```bash
xisfprep [--verbose|--quiet] hotpixel -i <pattern> -o <dir> [options]
```

### Arguments

**Required:**
- `-i, --input <pattern>` - Input files (wildcards supported)
- `-o, --output <dir>` - Output directory

**Detection:**
- `--k-hot <value>` - Hot pixel sigma threshold (default: 5.0)
- `--k-cold <value>` - Cold pixel sigma threshold (default: 5.0)
- `--k <value>` - Set both thresholds

**Bayer:**
- `--bayer <pattern>` - RGGB, BGGR, GRBG, GBRG (omit for debayered/mono)

**Tiling:**
- `--overlap <pixels>` - 8, 12, 16, 24, 32 (default: 24)

**Common:**
- `--overwrite` - Replace existing files
- `--suffix <text>` - Output suffix (default: "_corrected")
- `--parallel <n>` - Max parallel file processing (default: CPU count)
- `--output-format <fmt>` - uint8, uint16, uint32, float32, float64

### Examples

```bash
# Bayer data before debayering
xisfprep hotpixel -i "*.xisf" -o "fixed/" --bayer RGGB

# Aggressive cold pixel removal
xisfprep hotpixel -i "*.xisf" -o "out/" --k-hot 5.0 --k-cold 3.5

# Few files, max CPU utilization
xisfprep hotpixel -i "file*.xisf" -o "out/" --parallel 2
```

---

## Architecture

### Design Decisions

**Tiled Processing:**
- Fixed 128×128 tile size
- Configurable overlap (prevents boundary artifacts)
- Merge strategy: average corrections in overlap zones
- Handles images < 128×128 (single tile, no tiling)

**Detection Logic:**
- Global statistics: median, MAD per color plane (entire image)
- Local statistics: median, MAD per tile per color plane
- Defect flagging: `(local_outlier) OR (global_extreme_outlier)`
  - Local: `pixel > median_local + k × MAD_local`
  - Global: `|pixel - median_global| > 8 × MAD_global` (hard-coded catch-all)

**Correction Method:**
- VNG directional interpolation (interior pixels ≥2px from edge)
- Median replacement (edge pixels, VNG fallback)
- Same-color neighbor interpolation (Bayer awareness)

**Bayer Handling:**
- Pre-debayer only
- Four independent color planes: R, G1, G2, B
- Same-color neighbors at distance 2 in mosaic
- Explicit `--bayer` flag required (no auto-detection)

**Parallelism Strategy:**
- **File-level only:** Multiple files processed concurrently via `SharedInfra.BatchProcessing`
- **Tile-level sequential:** Tiles within each file processed sequentially (for-loop)
- Default `--parallel` = CPU count (user can override)
- **Rationale:** Avoids oversubscription complexity
  - File parallelism already saturates CPU (I/O + processing mixed)
  - Adding tile parallelism → `file_threads × tile_threads` total threads
  - Simpler implementation: no nested async coordination
- Total threads = `--parallel` value (typically ~CPU count)

**Edge Cases:**
- Images < 128×128: Single tile, no tiling
- Pixels within 2px border: Median correction (VNG needs 5×5 kernel)
- Tiles with >50% defects: Log warning, continue
- No valid neighbors: Skip correction, log error

**Validation:**
- k values > 0.0
- Overlap ∈ {8, 12, 16, 24, 32}
- Bayer pattern ∈ {RGGB, BGGR, GRBG, GBRG} if specified

---

## Algorithm Details

### Detection Flow

```
1. Load XISF image → float[] pixels
2. Compute global statistics (median, MAD) per color plane
3. Generate tile grid (128×128 with overlap)
4. For each tile (in parallel):
   a. Compute local statistics (median, MAD) per color plane
   b. For each pixel:
      - Get color plane index (Bayer) or channel
      - Compare against local + global thresholds
      - Flag defective pixels
   c. Correct defective pixels:
      - VNG interpolation (interior)
      - Median replacement (edges)
   d. Return corrections: (pixel_index, corrected_value)[]
5. Merge tiles:
   - Accumulate: correction_sum[idx] += value, correction_count[idx] += 1
   - Average: pixel[idx] = correction_sum[idx] / correction_count[idx]
6. Write corrected pixels to XISF (preserve metadata)
```

### VNG Correction

```fsharp
// Compute gradients in 8 directions
let gradients = allDirections |> Array.map (fun dir ->
    (dir, computeGradient getPixel x y dir))

// Select smooth directions (threshold = min + (max - min) / 2)
let validDirs = selectSmoothDirections gradients

// Get same-color neighbors in valid directions (distance 2 for Bayer)
let neighbors = validDirs |> Array.collect (getSameColorNeighbors x y)

// Average valid neighbors (fallback: all 8 directions if none valid)
Array.average neighbors
```

### Gradient Computation

Uses 5×5 neighborhood, computes directional gradient magnitude:
```
N:  |pixel(x,y-1) - pixel(x,y+1)| + |pixel(x,y-2) - pixel(x,y)|
NE: |pixel(x+1,y-1) - pixel(x-1,y+1)| + |pixel(x+2,y-2) - pixel(x,y)|
... (8 directions)
```

---

## Module Organization

### `Algorithms/DirectionalGradients.fs`

Shared gradient computation (extracted from Debayering):

```fsharp
type Direction = N | NE | E | SE | S | SW | W | NW

val computeGradient:
    getPixel:(int -> int -> float) -> x:int -> y:int -> dir:Direction -> float

val selectSmoothDirections:
    gradients:(Direction * float)[] -> Direction[]
```

### `Algorithms/HotPixelCorrection.fs`

Core detection and correction logic:

```fsharp
type DetectionConfig = {
    KHot: float
    KCold: float
    TileSize: int
    Overlap: int
    BayerPattern: BayerPattern option
}

type DetectionResult = {
    HotPixelCount: int
    ColdPixelCount: int
    TotalCorrected: int
}

val generateTileGrid: width:int -> height:int -> tileSize:int -> overlap:int -> TileInfo[]
val computeGlobalStats: pixels:float[] -> width:int -> height:int -> pattern:BayerPattern option -> (float * float)[]
val computeTileStats: pixels:float[] -> width:int -> tile:TileInfo -> pattern:BayerPattern option -> (float * float)[]
val correctPixelVNG: pixels:float[] -> width:int -> height:int -> x:int -> y:int -> pattern:BayerPattern option -> float
val correctPixelMedian: pixels:float[] -> width:int -> height:int -> x:int -> y:int -> pattern:BayerPattern option -> float
val detectAndCorrectTile: pixels:float[] -> width:int -> height:int -> tile:TileInfo -> config:DetectionConfig -> globalStats:(float * float)[] -> (int * float)[]
val processImage: pixels:float[] -> width:int -> height:int -> config:DetectionConfig -> Result<float[] * DetectionResult, HotPixelError>
```

### `Commands/HotPixel.fs`

CLI command implementation:

```fsharp
type HotPixelOptions = {
    Input: string
    Output: string
    KHot: float
    KCold: float
    Overlap: int
    BayerPattern: BayerPattern option
    Suffix: string
    Overwrite: bool
    MaxParallel: int
    OutputFormat: XisfSampleFormat option
}

type HotPixelError =
    | LoadFailed of XisfIO.XisfError
    | ProcessingFailed of HotPixelCorrection.HotPixelError
    | SaveFailed of XisfIO.XisfError

val parseArgs: string[] -> HotPixelOptions
val processFile: inputPath:string -> outputPaths:string list -> opts:HotPixelOptions -> Async<Result<unit, HotPixelError>>
val run: string[] -> int
```

**Processing Pipeline:**
1. File discovery via `Directory.GetFiles` with pattern
2. Batch processing via `SharedInfra.BatchProcessing.processBatch`
3. Per-file async processing:
   - Load XISF with `XisfIO.loadImageWithPixels`
   - Read pixels with `PixelIO.readPixelsAsFloat`
   - Build `DetectionConfig` from options
   - Call `HotPixelCorrection.processImage`
   - Write corrected pixels with `XisfIO` (preserve metadata)
   - Log per-file statistics

**Parallelism Implementation:**
```fsharp
// File-level parallelism via SharedInfra.BatchProcessing
let! exitCode = SharedInfra.BatchProcessing.processBatch batchConfig buildOutputPaths processFile

// Within processFile: sequential tile processing
for tile in tiles do
    let tileCorrections = detectAndCorrectTile pixels width height tile config globalStats
    // Accumulate corrections...
```

**Note:** No `Async.Parallel` for tiles - sequential for-loop avoids oversubscription.

### Modified Modules

**`Algorithms/Debayering.fs`:**
- Now uses `DirectionalGradients` module
- Removed duplicate `Direction` type and `computeGradient`

**`Program.fs`:**
- Added `"hotpixel", Commands.HotPixel.run` to `commandHandlers`
- Updated help text

---

## File Structure

```
Algorithms/
  DirectionalGradients.fs    # Shared gradient computation
  HotPixelCorrection.fs      # Detection and correction core
  Debayering.fs              # Uses DirectionalGradients
  Statistics.fs              # Median, MAD, k-sigma clipping

Commands/
  HotPixel.fs                # CLI implementation
  SharedInfra.fs             # Batch processing utilities

Program.fs                   # Command routing
PixelIO.fs                   # XISF pixel I/O
XisfIO.fs                    # XISF file I/O
```

---

## Dependencies

**Internal:**
- `Algorithms.Statistics` - median, MAD calculations
- `Algorithms.Debayering` - Bayer pattern types, `getColor`
- `Algorithms.DirectionalGradients` - gradient computation
- `Commands.SharedInfra` - batch processing, argument parsing
- `PixelIO` - pixel data conversion
- `XisfIO` - XISF file operations

**External:**
- `FsToolkit.ErrorHandling` - railway-oriented programming
- `Serilog` - structured logging
- `XisfLib.Core` - XISF format library

---

## Performance

**Memory:**
- Full image pixel array: ~width × height × channels × 4 bytes
- Correction accumulators: 2 × width × height × 4 bytes
- Example 20MP mono: 80MB pixels + 160MB accumulators = 240MB

**CPU Utilization:**
- File-level parallelism prevents oversubscription
- User-controlled via `--parallel` (default: CPU count)
- Tiles processed sequentially within each file (no nested parallelism)
- Total threads ≈ CPU count

**Optimization Targets:**
- Gradient computation (hot path, called per pixel)
- Median calculation (use partial sort for large windows)
- Bayer color extraction (minimize array allocations)

---

## Integration

**Preprocessing Pipeline:**
Position in workflow: After calibration, before debayering

```bash
# Typical workflow
xisfprep calibrate -i "lights/*.xisf" -o "calibrated/" --dark dark.xisf --flat flat.xisf
xisfprep hotpixel -i "calibrated/*.xisf" -o "corrected/" --bayer RGGB
xisfprep debayer -i "corrected/*.xisf" -o "debayered/"
```

**Future `preprocess` Integration:**
```bash
xisfprep preprocess -i "lights/*.xisf" -o "processed/" \
  --dark dark.xisf --flat flat.xisf \
  --correct-pixels --k 5.0 --bayer RGGB \
  --debayer --align
```

---

## Future Enhancements

- Defect map visualization (overlay showing corrected pixels)
- Bayer pattern auto-detection from XISF `CFAPattern` property
- Metadata recording (add XISF property with correction parameters)
- Adaptive tile sizing (content-aware, varying background complexity)
- Cosmic ray detection (temporal multi-frame analysis)
- Dark frame analysis (build hot pixel library from master darks)
- Morphological clustering (handle grouped defects differently)
