# Align Command

## Overview

Star-based image registration via detection, matching, and geometric transformation. Operates on raw pixel values without preprocessing.

For workflows requiring calibration (bias/dark/flat) or binning before alignment, use the `preprocess` command.

## Output Modes

### Detect Mode (`--output-mode detect`)

Validates star detection parameters. Outputs black background with detected stars as circles (radius = 2× FWHM).

**Use cases:** Verify threshold settings, check FWHM filters, validate grid size

### Match Mode (`--output-mode match`)

Validates star matching between reference and target. Outputs visualization with:
- Reference stars as faint gaussians
- Matched target stars as circles
- Unmatched target stars as X markers

**Use cases:** Verify correspondences, diagnose alignment failures, compare algorithms

### Align Mode (`--output-mode align`) - Default

Registers images to reference frame. Outputs transformed Float32 image.

**Process:** Detect → Match → Estimate transform → Apply distortion (optional) → Resample

**Optional outputs:**
- `--include-detection-model` - Star visualization alongside aligned image
- `--include-distortion-model` - Distortion heatmap alongside aligned image

### Distortion Mode (`--output-mode distortion`)

Analyzes optical distortion field. Outputs RGB heatmap (blue=0-2px, cyan=2-4px, green=4-6px, yellow=6-8px, red=>8px) with control points as white circles.

Console output with `--show-distortion-stats` shows ASCII heatmap and statistics.

## Pipeline Flow

### Reference Analysis (once per batch)

1. Load reference image
2. Detect stars (minimum 10 required)
3. Form triangles (for triangle algorithm)

### Target Processing (per image, mode-dependent)

**Detect:** Load → Detect → Visualize → Save

**Match:** Load → Detect → Match → Visualize → Save

**Align:** Load → Detect → Match → Transform → Save

**Distortion:** Load → Detect → Match → Compute field → Visualize → Save

## Star Detection

Grid-based background estimation with PSF fitting.

**Parameters:**
- `--threshold <σ>` - Detection sensitivity (default: 5.0)
- `--grid-size <px>` - Background grid (default: 128)
- `--min-fwhm <px>` - Minimum FWHM (default: 1.5, filters hot pixels)
- `--max-fwhm <px>` - Maximum FWHM (default: 20.0, filters extended objects)
- `--max-eccentricity <0-1>` - Maximum ellipticity (default: 0.5, filters cosmic rays)
- `--max-stars <n>` - Limit to brightest N stars (default: 20000)

**Process:** Extract luminance → Calculate MAD → Grid background subtraction → Detect maxima → Fit PSF → Filter → Sort by flux

## Star Matching

### Triangle Matching (`--algorithm triangle`)

Forms triangles from brightest stars, matches by normalized edge ratios (scale/rotation invariant), votes for correspondences.

**Parameters:**
- `--ratio-tolerance` - Ratio tolerance (default: 0.05 = 5%)
- `--max-stars-triangles` - Stars for triangles (default: 100)
- `--min-votes` - Votes required (default: 3)

**Best for:** Standard fields, moderate rotation (<45°)

### Expanding Matching (`--algorithm expanding`) - Default

Anchor-based matching that grows outward with spatial constraints. More robust for large rotations and partial overlap.

**Parameters:**
- `--anchor-stars` - Anchor count (default: 12)
- `--anchor-spread` - Distribution: `center` (default) or `grid` (3×3)
- `--ratio-tolerance` - Anchor tolerance (default: 0.05)
- `--min-votes` - Anchor votes required (default: 3)

**Best for:** Large rotations, partial overlap, mosaics

## Transformation

Similarity transform (translation, rotation, scale) estimated from matched pairs, applied via inverse mapping with interpolation.

**Interpolation** (`--interpolation`):
- `nearest` - Fast, preserves values, blocky
- `bilinear` - Smooth, fast, slight blur
- `bicubic` - Sharper, moderate cost
- `lanczos3` - Best quality, slower (default)

## Distortion Correction

Radial Basis Function (RBF) correction for optical distortions beyond global transform.

**Kernels** (`--distortion`):
- `none` - Similarity only (default)
- `wendland` - Local RBF, fast (recommended)
- `tps` - Thin-plate spline, global smoothness
- `imq` - Inverse multiquadric

**Parameters:**
- `--rbf-support <factor>` - Support radius (default: 3.0)
- `--rbf-regularization` - Regularization (default: 1e-6)

**When to use:** Field curvature, pincushion/barrel distortion, fast optics

## Reference Selection

**Options:**
- `--reference <file>` - Explicit reference
- `--auto-reference` - Auto-select (currently uses first file)
- Default: First file

Choose reference with good star distribution and high SNR. Minimum 10 stars required, 50+ recommended.

## Batch Processing

Reference analyzed once, targets processed in parallel.

**Output:** `<basename><suffix>.xisf` in output directory
- Default suffixes: `_a` (align), `_det` (detect), `_mat` (match), `_dist` (distortion)
- Override with `--suffix`

**Parallelism:** `--parallel <n>` (default: CPU cores)

**Overwrite:** `--overwrite` flag (default: skip existing)

**Errors:** Reference failure aborts batch, individual failures logged but don't stop others

## Output Format

**Aligned images:** Float32, [0,1] normalized, ready for stacking

**Visualizations:** Preserves input format (or `--output-format`)

**Metadata:** XSHIFT, YSHIFT, ROTATION, SCALE, STARMTCH, MATCHED, UNMATCHD, STARCOUNT, STARTHRES, STARGRID, HISTORY

## Examples

```bash
# Validate detection
xisfprep align -i "*.xisf" -o "validation/" --output-mode detect

# Standard alignment
xisfprep align -i "*.xisf" -o "aligned/" -r "reference.xisf"

# With distortion correction
xisfprep align -i "*.xisf" -o "aligned/" --distortion wendland

# Analyze distortion
xisfprep align -i "*.xisf" -o "analysis/" --output-mode distortion --show-distortion-stats
```

## Scope

**Does:** Star registration, geometric transformation, works on raw images

**Does NOT:** Calibration, binning (use `preprocess`), stacking (use `integrate`), cosmetic correction, gradient removal

## Exit Codes

- `0` - Success
- `1` - Validation, reference analysis, or file processing failure
