# Align Inline Calibration Design

Design for adding inline calibration to the alignment pipeline, enabling calibration and alignment in a single pass without intermediate files.

## Motivation

Currently, the workflow requires separate calibration and alignment steps:

```
Raw Lights → Calibrate → Calibrated Lights → Align → Aligned Lights
```

This creates intermediate files and requires disk I/O between steps. Inline calibration enables:

```
Raw Lights → Align (with inline calibration) → Aligned & Calibrated Lights
```

**Benefits**:
- Fewer intermediate files (saves disk space)
- Single-pass processing (saves time)
- Simplified workflow
- Consistent with Integrate.fs pattern (which already has inline calibration for master dark/flat creation)

## Architecture Decision

### Unified Calibration Module

Extract and unify calibration logic from three existing commands:
- **Calibrate.fs**: Full calibration pipeline (bias + dark + flat + pedestal)
- **Integrate.fs**: Inline calibration for master creation (bias + dark only)
- **Align.fs**: Will use full calibration inline (bias + dark + flat)

Create a shared `Algorithms/Calibration.fs` module to eliminate duplication and ensure consistency.

### Data Flow in Align with Inline Calibration

```
Input Light Frame (raw)
         ↓
Load Master Frames (once for all frames)
  - Master Bias
  - Master Dark
  - Master Flat
         ↓
Apply Calibration (per-pixel)
  Output = ((Light - Bias - Dark) / Flat)
         ↓
Detect Stars (on calibrated pixels)
         ↓
Match to Reference Stars
         ↓
Compute Transform (similarity + optional RBF)
         ↓
Apply Transform (with interpolation)
         ↓
Output Aligned Frame (calibrated + aligned)
```

**Critical Decision**: Calibration happens BEFORE star detection, not after alignment. This ensures:
1. Star detection works on clean calibrated data
2. Flat field correction removes vignetting before matching
3. Output frames are both calibrated and aligned

## Module Organization

### New Module: `Algorithms/Calibration.fs`

Core calibration logic extracted from existing commands:

**Responsibilities**:
- Load master frames (bias, dark, flat) into memory
- Validate dimensions and formats
- Apply per-pixel calibration formula
- Track calibration state (uncalibrated masters)
- Generate calibration metadata (FITS headers, properties)

**Core Types**:
- `CalibrationMasters`: Holds loaded master frame pixel data, dimensions, and format metadata
- `CalibrationConfig`: Configuration for what masters to use and their state (calibrated/uncalibrated)
- `CalibrationResult`: Output including calibrated pixels and diagnostic counts

**Core Functions**:
- `loadMasters`: Async load master frames from disk into memory
- `calibratePixels`: Apply calibration formula to light frame pixels
- `buildCalibrationHeaders`: Generate FITS headers for provenance tracking
- `buildCalibrationProperties`: Generate XISF properties for metadata
- `validateDimensions`: Check master frame dimensions match target image

### Modified Module: `Commands.Align`

Enhanced to support inline calibration:

**New Configuration**:
- Add calibration frame paths to `AlignOptions` record
- Add calibration state flags (uncalibrated masters)
- Wire through CLI argument parsing

**Modified Processing Flow**:
- Load calibration masters once at startup (shared across all input files)
- In per-file processing: apply calibration before star detection
- Extract channel for detection from calibrated pixels
- Rest of alignment pipeline unchanged (matching, transform, interpolation, write)

### Refactored Modules

**`Commands.Integrate`**:
- Remove inline `InlineCalibration` nested module
- Use `Algorithms.Calibration` instead
- Simplified, consistent with other commands

**`Commands.Calibrate`**:
- Remove duplicate calibration logic
- Use `Algorithms.Calibration` for core operations
- Keep command-specific concerns (CLI parsing, file I/O, progress reporting)

## Configuration and CLI Arguments

### New Arguments for Align

```
Inline Calibration:
  --bias, -b <file>         Master bias frame
  --bias-level <value>      Constant bias value (alternative to --bias)
  --dark, -d <file>         Master dark frame
  --flat, -f <file>         Master flat frame
  --uncalibrated-dark       Dark is raw (not bias-subtracted)
  --uncalibrated-flat       Flat is raw (not bias/dark-subtracted)
```

### Validation Rules

Consistent with Calibrate.fs:
- `--bias` and `--bias-level` are mutually exclusive
- `--uncalibrated-dark` requires `--bias` or `--bias-level`
- `--uncalibrated-flat` requires both `--bias` (or `--bias-level`) and `--dark`
- All master frames must have matching dimensions

### Omitted from Initial Design

Features from Calibrate.fs NOT included in initial Align integration:
- **`--optimize-dark`**: Dark scaling optimization can be added later if needed
- **`--pedestal`**: Not needed for alignment workflow (pedestal is for preventing clipping in final output)

These can be added later if use cases emerge.

## Algorithm Details

### Calibration Formula

Standard astrophotography calibration:

**Base formula**: `Output = ((Light - Bias - Dark) / Flat)`

**State handling**:
- If **dark is uncalibrated** (raw): Subtract bias from dark before applying
- If **flat is uncalibrated** (raw): Apply full calibration to flat before division
- Division by flat uses flat median for normalization to preserve overall brightness
- Skip division if flat pixel is zero (handles vignetting edge cases)

### Processing Steps

**Per-pixel calibration**:
1. Subtract bias (from frame or constant level)
2. Subtract dark (applying bias correction if dark is uncalibrated)
3. Divide by flat (applying full calibration if flat is uncalibrated)
4. Apply pedestal offset (for Calibrate command output)
5. Clamp to valid range [0, 65535]

**Master frame loading**:
- Masters loaded once at startup, held in memory
- All masters validated for matching dimensions
- Flat median calculated during load for normalization
- Format precision tracked for output format selection

## Output Metadata

### FITS Headers

Calibration provenance tracked via HISTORY entries:
- Calibration mode (inline vs standalone)
- Master frame filenames (bias, dark, flat)
- Master frame states (calibrated vs uncalibrated)
- Processing software and version

Combined with existing headers:
- Calibration headers (if applied)
- Alignment headers (transform parameters, matched stars)
- Distortion headers (if RBF correction applied)

### XISF Properties

Calibration parameters stored as structured properties:
- Master frame paths (full paths for reproducibility)
- Master frame states (calibrated boolean flags)
- Flat median value (for reference)
- Dark scale factor (if optimization enabled)

Property namespace: `Calibration:*` (e.g., `Calibration:BiasFrame`, `Calibration:DarkCalibrated`)

## Performance Considerations

### Memory Usage

**Master Frames**: Loaded once and held in memory for the entire run
- Typical 16MP image: ~64MB per master frame (float array)
- Bias + Dark + Flat: ~192MB total
- Acceptable for modern systems

**Light Frames**: Processed one at a time
- Read → Calibrate → Detect → Align → Write → Release
- No batch loading (unlike Integrate.fs which loads all frames)

### Computational Cost

**Per Frame Overhead**:
1. Calibration: O(n) pixel operations (3-4 operations per pixel)
2. Star detection: O(n) for background estimation + O(k) for PSF fitting
3. Matching: O(k²) for triangles or O(k log k) for expanding
4. Transformation: O(n) with interpolation

Calibration adds minimal overhead (~5-10% vs star detection).

### I/O Optimization

**Sequential Processing**:
- Frames processed one at a time (unlike Integrate's parallel loading)
- Masters loaded once from disk (not per-frame)
- Output written incrementally (not batched)

**Parallelism**:
- Multiple frames can be processed in parallel (existing `--parallel` flag)
- Each worker has shared read access to master frames
- No lock contention (read-only access)

## Future Extensibility

### Pipeline Composition

Design allows for future pre-processing steps in a composable pipeline:

```
Input Frame
    ↓
[Calibration]        ← This design
    ↓
[Cosmetic Correction]  ← Future: Hot pixel removal
    ↓
[Debayer]            ← Future: CFA → RGB conversion
    ↓
[Binning]            ← Future: Resolution reduction
    ↓
Star Detection
    ↓
Alignment
```

Each step:
- Takes `float[]` pixels as input
- Returns `float[]` pixels as output
- Optional (enabled via CLI flags)
- Independent of other steps

### Abstraction Pattern

```fsharp
type PreProcessingStep = {
    Name: string
    Apply: float[] -> int -> int -> int -> float[]
    GetMetadata: unit -> XisfCoreElement[] * XisfProperty[]
}

let buildPipeline (opts: AlignOptions) =
    [
        if opts.BiasFrame.IsSome || ... then
            yield calibrationStep
        if opts.RemoveHotPixels then
            yield hotPixelStep
        if opts.DebayerPattern.IsSome then
            yield debayerStep
    ]

let applyPipeline (pixels: float[]) (steps: PreProcessingStep list) =
    steps |> List.fold (fun px step -> step.Apply px w h c) pixels
```

This allows arbitrary combinations of pre-processing without modifying Align.fs core logic.

## Implementation Order

### Phase 1: Extract Calibration Module
1. Create `Algorithms/Calibration.fs`
2. Define types (`CalibrationMasters`, `CalibrationConfig`, `CalibrationResult`)
3. Implement core functions:
   - `loadMasters`
   - `calibratePixels`
   - `buildCalibrationHeaders`
   - `buildCalibrationProperties`
   - `validateDimensions`
4. Add unit tests for calibration formulas

### Phase 2: Refactor Existing Commands
1. **Integrate.fs**: Replace `InlineCalibration` module with `Algorithms.Calibration`
2. **Calibrate.fs**: Replace inline functions with `Algorithms.Calibration`
3. Verify existing functionality unchanged (regression tests)

### Phase 3: Enhance Align.fs
1. Add calibration options to `AlignOptions` record
2. Add CLI argument parsing for calibration flags
3. Add validation rules for calibration config
4. Modify `run` function to load masters once
5. Modify `processAlignFile` to apply calibration before detection
6. Add calibration metadata to output headers/properties
7. Update help text with calibration section

### Phase 4: Testing
1. **Unit tests**: Calibration formula edge cases
2. **Integration tests**:
   - Align with inline calibration vs separate calibrate→align
   - Verify output matches
   - Test uncalibrated master frame handling
3. **Performance tests**: Measure overhead vs non-calibrated alignment

### Phase 5: Documentation
1. Update `README.md` with inline calibration workflow
2. Add examples to help text
3. Update `QuickRef.md` with new flags

## Example Usage

### Basic: Calibrate and Align in One Pass

```bash
xisfprep align \
  -i "lights/*.xisf" \
  -o "aligned/" \
  --bias "masters/bias.xisf" \
  --dark "masters/dark.xisf" \
  --flat "masters/flat.xisf" \
  --auto-reference
```

### Advanced: Uncalibrated Masters

```bash
xisfprep align \
  -i "lights/*.xisf" \
  -o "aligned/" \
  --bias "masters/bias_raw.xisf" \
  --dark "masters/dark_raw.xisf" \
  --flat "masters/flat_raw.xisf" \
  --uncalibrated-dark \
  --uncalibrated-flat
```

### With Distortion Correction

```bash
xisfprep align \
  -i "lights/*.xisf" \
  -o "aligned/" \
  -b "masters/bias.xisf" \
  -d "masters/dark.xisf" \
  -f "masters/flat.xisf" \
  --distortion wendland \
  --interpolation lanczos3
```

### Validation Mode: Detect on Calibrated Data

```bash
# See how calibration affects star detection
xisfprep align \
  -i "lights/*.xisf" \
  -o "validation/" \
  --output-mode detect \
  --bias "masters/bias.xisf" \
  --dark "masters/dark.xisf" \
  --flat "masters/flat.xisf"
```

## Design Decisions

### 1. Calibration Before Detection
**Decision**: Apply calibration before star detection, not after alignment

**Rationale**:
- Flat field correction removes vignetting that affects star detection
- Dark subtraction removes hot pixels that could be false detections
- Bias subtraction centers data around zero
- Output frames are fully processed (calibrated + aligned)

**Alternative Considered**: Align then calibrate
- Would require detecting stars on uncalibrated data
- Output would be aligned but uncalibrated (not useful)
- Would need additional calibration step anyway

### 2. Shared Calibration Module
**Decision**: Extract common logic to `Algorithms/Calibration.fs`

**Rationale**:
- Eliminates duplication across three commands
- Ensures consistency in calibration formula
- Simplifies testing (single implementation)
- Enables future reuse (e.g., in stacking algorithms)

**Alternative Considered**: Keep logic in each command
- More duplication and maintenance burden
- Risk of divergence between implementations

### 3. No Pedestal Support
**Decision**: Don't include `--pedestal` flag in initial design

**Rationale**:
- Pedestal is for preventing negative values in final output
- Alignment works in float space (can handle negative values)
- Pedestal is a Calibrate.fs concern for final deliverable images
- Can be added later if use cases emerge

### 4. Masters Loaded Once
**Decision**: Load master frames once at startup, reuse for all input files

**Rationale**:
- Significant I/O savings (vs loading per-file)
- Memory cost is acceptable (~200MB for typical masters)
- Simplifies parallelism (shared read-only access)

**Alternative Considered**: Load per-file
- Massive I/O overhead
- No benefit (masters are identical for all files)

### 5. Output Always Calibrated
**Decision**: If calibration frames are provided, output is always calibrated

**Rationale**:
- Consistent behavior (no "align without calibration" mode when masters provided)
- Simplifies user understanding
- Matches user intent (if you provide masters, you want calibration)

**Alternative Considered**: Add `--skip-calibration` flag
- Confusing (why provide masters if not using them?)
- Adds complexity for minimal benefit

---

## Open Questions

### 1. Output Format
**Question**: Should calibrated+aligned output default to Float32 like Integrate, or preserve input format like Calibrate?

**Options**:
- **Float32**: Preserves interpolation precision, consistent with integration workflow
- **Preserve input**: Maintains original bit depth, smaller file sizes

**Recommendation**: Default to Float32 for interpolation precision, allow override with `--output-format`

### 2. Flat Median Calculation
**Question**: Should flat median be calculated per-channel or across all channels?

**Options**:
- **Per-channel**: Handles color imbalance in flat
- **All channels**: Simpler, assumes balanced flat

**Recommendation**: Per-channel (already implemented in Calibrate.fs)

### 3. Progress Reporting
**Question**: Should calibration step report progress separately from alignment?

**Options**:
- **Separate**: "Calibrating frame 1/10... Aligning frame 1/10..."
- **Combined**: "Processing frame 1/10 (calibrate + align)..."

**Recommendation**: Combined (simpler, less verbose)

---

## Success Criteria

Implementation is complete when:

1. ✅ `Algorithms/Calibration.fs` module exists with core functions
2. ✅ `Integrate.fs` and `Calibrate.fs` refactored to use shared module
3. ✅ `Align.fs` accepts calibration CLI arguments
4. ✅ Calibration is applied before star detection in align pipeline
5. ✅ Output metadata includes calibration provenance
6. ✅ Regression tests pass for existing commands
7. ✅ Integration tests verify calibrate→align equals align-with-inline-cal
8. ✅ Help text documents new workflow
9. ✅ Performance overhead is < 15% vs non-calibrated alignment

---

## Dependencies

### Module Dependency Graph

```
Module: Commands.Align
    ├── Algorithms.Calibration  [NEW]
    ├── Algorithms.StarDetection
    ├── Algorithms.TriangleMatch
    ├── Algorithms.SpatialMatch
    ├── Algorithms.RBFTransform
    ├── Algorithms.Interpolation
    ├── Algorithms.Painting
    ├── Algorithms.OutputImage
    └── Algorithms.Statistics

Module: Commands.Integrate
    ├── Algorithms.Calibration  [NEW - replaces nested InlineCalibration module]
    └── Algorithms.Statistics

Module: Commands.Calibrate
    ├── Algorithms.Calibration  [NEW - replaces inline functions]
    └── Algorithms.Statistics

Module: Algorithms.Calibration  [NEW]
    ├── XisfLib.Core  (XISF I/O types)
    ├── Algorithms.Statistics  (median calculation)
    └── System.* (base library)
```

### Dependency Properties
- `Algorithms.Calibration` is a leaf module with no dependencies on other `Algorithms.*` modules (except Statistics)
- Commands depend on Algorithms, never the reverse
- No circular dependencies
- Clean separation: Algorithms contain reusable logic, Commands contain CLI and workflow concerns
