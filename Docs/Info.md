# XISF Preprocessing CLI - Design Document

## Overview

Command-line application for batch preprocessing of astrophotography images in XISF format. Supports monochrome and multi-channel images with operations for calibration, debayering, alignment, and integration.

**Target Users:** Astrophotographers processing XISF files from capture software (NINA, N.I.N.A., etc.)

## Core Philosophy

**Verb-based commands** - Each preprocessing step is a discrete command
**Wildcard support** - Batch file selection via glob patterns
**Sensible defaults** - Minimal required arguments for common workflows
**Preserve metadata** - Maintain FITS keywords and properties through pipeline
**Memory efficiency** - Process large file sets without excessive RAM

## Command Structure

```
xisfprep <verb> [options]

Verbs:
  calibrate   Apply bias/dark/flat calibration frames
  debayer     Convert Bayer mosaic to RGB
  headers     Extract FITS keywords and HISTORY values from files
  align       Register images to reference
  integrate   Stack/combine multiple images
  stats       Calculate image statistics
  stars       Detect stars and generate visualization
  convert     Format conversion and export
  bin         Downsample images by binning pixels
  inspect     Diagnostic inspection of XISF file structure
```

## Command Details

### `calibrate`

Apply calibration frames to light frames.

**Usage:**
```bash
xisfprep calibrate --input "lights/*.xisf" --output "calibrated/" \
  [--bias "master_bias.xisf" | --bias-level 500] \
  [--dark "master_dark.xisf"] \
  [--flat "master_flat.xisf"] \
  [--uncalibrated-dark] \
  [--uncalibrated-flat] \
  [--optimize-dark] \
  [--pedestal 100]
```

**Arguments:**
- `--input, -i` (required) - Input light frames (wildcards supported)
- `--output, -o` (required) - Output directory for calibrated frames
- `--bias, -b` (optional) - Master bias frame
- `--bias-level` (optional) - Constant bias value (alternative to --bias)
- `--dark, -d` (optional) - Master dark frame
- `--flat, -f` (optional) - Master flat frame
- `--uncalibrated-dark` (flag) - Master dark is raw (not bias-subtracted)
- `--uncalibrated-flat` (flag) - Master flat is raw (not bias/dark-subtracted)
- `--optimize-dark` (flag) - Optimize dark scaling for temperature/exposure differences
  - Requires: `--uncalibrated-dark` and (`--bias` or `--bias-level`)
- `--pedestal` (optional) - Output pedestal added after calibration [0-65535] (default: 0)
- `--suffix` (optional) - Output filename suffix (default: "_cal")

**Algorithm:**
```
Output = ((Light - Bias - Dark) / Flat) + Pedestal
```

1. Subtract bias (frame or constant)
2. Subtract dark
3. Divide by flat (normalized to median)
4. Add output pedestal
5. Clamp to [0, 65535]

**Bias Options:**
- `--bias <file>` - Use master bias frame
- `--bias-level <value>` - Use constant bias value (mutually exclusive with --bias)
- Use `--bias-level` when bias is uniform or creating master darks

**Output Pedestal:**
- Added after calibration to prevent clipping
- Shifts histogram away from zero
- Typical values: 50-200 ADU
- Monitor output histogram to verify effectiveness

**Master Frame Calibration State:**

Default assumes pre-calibrated masters (standard workflow):
- Master dark = bias-subtracted dark current
- Master flat = bias/dark-subtracted, normalized

Use flags for raw masters:
- `--uncalibrated-dark` - Dark is raw (still contains bias)
- `--uncalibrated-flat` - Flat is raw (still contains bias/dark)

**Dark Optimization:**

When `--optimize-dark` is enabled, the calibration algorithm finds the optimal scaling factor for the dark frame that minimizes noise in the calibrated output. This compensates for temperature and exposure time differences between dark and light frames.

The algorithm:
1. Uses k-sigma clipped noise estimation (MAD-based)
2. Golden section search to find optimal scale factor
3. Applies scaling: `Dark_scaled = k * (Dark - Bias)`
4. Records scale factor in output metadata as `Calibration:DarkScaleFactor`

Typical use cases:
- Darks taken at different sensor temperature than lights
- Reusing dark library across varying ambient temperatures
- Dark exposure time differs from light exposure time

Requirements:
- `--uncalibrated-dark` - Dark must be raw (algorithm subtracts bias internally)
- `--bias` or `--bias-level` - Bias required for dark calibration

**Validation:**
- Requires at least one of: `--bias`, `--bias-level`, `--dark`, `--flat`, or `--pedestal`
- Dimensions and channels must match exactly
- `--bias` and `--bias-level` are mutually exclusive
- If `--uncalibrated-dark`: bias (frame or constant) required
- If `--uncalibrated-flat`: bias and dark required
- If `--optimize-dark`: requires `--dark`, `--uncalibrated-dark`, and bias
- Pedestal range: [0, 65535]

---

### `debayer`

Convert Bayer mosaic (single-channel) to RGB using VNG interpolation.

**Usage:**
```bash
xisfprep debayer --input "mono/*.xisf" --output "rgb/" \
  [--pattern RGGB] \
  [--algorithm vng] \
  [--split] \
  [--overwrite]
```

**Arguments:**
- `--input, -i` (required) - Input Bayer mosaic files (wildcards supported)
- `--output, -o` (required) - Output directory for RGB files
- `--pattern, -p` (optional) - Bayer pattern override (RGGB, BGGR, GRBG, GBRG)
- `--algorithm` (optional) - Interpolation algorithm (default: vng)
  - `vng` - Variable Number of Gradients (default, high quality)
  - `bilinear` - Simple bilinear (fast, lower quality)
- `--suffix` (optional) - Output filename suffix (default: "_d")
- `--split` (flag) - Output as separate R, G, B monochrome files instead of combined RGB
  - Creates `_R.xisf`, `_G.xisf`, `_B.xisf` files with FILTER keyword set
  - Useful for treating OSC data like narrowband for independent channel processing
- `--overwrite` (flag) - Overwrite existing output files (default: skip existing with warning)
- `--parallel <n>` (optional) - Number of parallel operations (default: CPU cores)

**Process:**
1. Reads Bayer pattern from FITS keyword `BAYERPAT` or uses override
2. Validates input is single-channel
3. Applies VNG interpolation (proven in debayer.fsx)
4. Outputs 3-channel RGB image (or separate R/G/B monochrome with --split)
5. Removes ColorFilterArray and Bayer FITS keywords from output

**Validation:**
- Input must be single-channel (monochrome)
- Input must not already be RGB color space
- Pattern must be valid Bayer pattern

---

### `headers`

Extract specific FITS keywords and HISTORY comment values from multiple files.

**Usage:**
```bash
xisfprep headers --input "calibrated/*.xisf" \
  [--keys FILTER,CCD-TEMP,EXPOSURE] \
  [--history "masterBias.fileName,masterDark.fileName,masterFlat.fileName"]
```

**Arguments:**
- `--input, -i` (required) - Input files (wildcards supported)
- `--keys` (optional) - Comma-separated FITS keywords to extract
- `--history` (optional) - Comma-separated HISTORY comment patterns to extract
  - Matches patterns like `ImageCalibration.masterBias.fileName: <value>`

**Output:**
Table with filename and requested values:
```
File                              | FILTER | masterBias.fileName                    | masterFlat.fileName
wr134_B_2023-06-14_01-39.xisf     | B      | 20220506.MasterBias.Gain.56...         | 20230613.MasterFlatCal.B.xisf
wr134_R_2023-06-14_02-10.xisf     | R      | 20220506.MasterBias.Gain.56...         | 20230613.MasterFlatCal.R.xisf
```

---

### `align`

Register images to reference frame using star detection and triangle matching.

**Usage:**
```bash
xisfprep align --input "images/*.xisf" --output "aligned/" \
  [--reference "best_frame.xisf"] \
  [--auto-reference] \
  [--output-mode align] \
  [--max-shift 50] \
  [--interpolation bicubic] \
  [--threshold 5.0] \
  [--ratio-tolerance 0.05] \
  [--intensity 1.0]
```

**Arguments:**
- `--input, -i` (required) - Input image files (wildcards supported)
- `--output, -o` (required) - Output directory for aligned files
- `--reference, -r` (optional) - Reference frame to align to (first file if omitted)
- `--auto-reference` (flag) - Auto-select best reference (highest star count/SNR)
- `--suffix` (optional) - Output filename suffix (default: "_a")
- `--overwrite` (flag) - Overwrite existing output files
- `--output-format` (optional) - Output sample format (default: preserve input)

**Output Mode:**
- `--output-mode` (optional) - Output type (default: align)
  - `detect` - Show detected stars only (validation)
  - `match` - Show matched star correspondences (validation)
  - `align` - Apply transformation (default)

**Alignment Parameters:**
- `--max-shift` (optional) - Maximum pixel shift allowed (default: 100)
- `--interpolation` (optional) - Resampling method (default: bicubic)
  - `nearest` - Nearest neighbor (preserves original values)
  - `bilinear` - Bilinear (smooth, fast)
  - `bicubic` - Bicubic (smooth, higher quality)

**Detection Parameters:**
- `--threshold` (optional) - Detection threshold in sigma (default: 5.0)
- `--grid-size` (optional) - Background grid size in pixels (default: 128)
- `--min-fwhm` (optional) - Minimum FWHM filter in pixels (default: 1.5)
- `--max-fwhm` (optional) - Maximum FWHM filter in pixels (default: 20.0)
- `--max-eccentricity` (optional) - Maximum eccentricity filter (default: 0.5)
- `--max-stars` (optional) - Maximum stars to detect (default: 20000)

**Matching Parameters:**
- `--algorithm` (optional) - Matching algorithm (default: expanding)
  - `triangle` - Triangle ratio matching (legacy)
  - `expanding` - Center-seeded expanding match with RANSAC (default, recommended)
- `--anchor-stars` (optional) - Anchor stars for expanding algorithm (default: 12)
- `--anchor-spread` (optional) - Anchor distribution method (default: center)
  - `center` - Select from central region only
  - `grid` - Distribute across 3×3 grid
- `--ratio-tolerance` (optional) - Triangle ratio matching tolerance (default: 0.05)
- `--max-stars-triangles` (optional) - Stars used for triangle formation (default: 100)
- `--min-votes` (optional) - Minimum votes for valid correspondence (default: 3)

**Visualization Parameters:**
- `--intensity` (optional) - Marker brightness 0-1 (default: 1.0)

**Output Modes:**

*Detect Mode:* (`--output-mode detect`)
- Output: Black image with detected stars as circles
- Suffix: `_det`
- Purpose: Validate star detection is working correctly

*Match Mode:* (`--output-mode match`)
- Output: Black image showing star correspondences
- Suffix: `_match`
- Markers:
  - Circle: Matched stars (found in both reference and target)
  - X: Unmatched stars (detected but not matched)
  - Gaussian: Reference star positions
- Purpose: Validate triangle matching quality

*Align Mode:* (`--output-mode align`, default)
- Output: Transformed/aligned image
- Suffix: `_a`
- Purpose: Actual image alignment

*Distortion Mode:* (`--output-mode distortion`)
- Output: RGB heatmap visualization of distortion field
- Suffix: `_dist`
- Color scale: Blue (low) → Cyan → Green → Yellow → Red (high)
- Max scale: 10px distortion
- White circles: RBF control point locations
- Purpose: Visualize non-linear distortion correction field

**Distortion Correction:**
- `--distortion <type>` - Correction method (default: none)
  - `none` - Similarity transform only (translation, rotation, scale)
  - `wendland` - Compact support RBF (recommended, O(log N) evaluation)
  - `tps` - Thin-plate spline (global, O(N) evaluation)
  - `imq` - Inverse multiquadric (global, O(N) evaluation)
- `--rbf-support <factor>` - Wendland support radius factor (default: 3.0)
- `--rbf-regularization <value>` - Regularization parameter (default: 1e-6)

**Diagnostic Outputs:**
- `--show-distortion-stats` - Print console ASCII heatmap in distortion mode
  - Shows 10-column grid with distortion magnitude at each cell
  - Includes min/max/mean/stddev statistics
  - ASCII shading: · (<2px), ░ (2-4px), ▒ (4-6px), ▓ (6-8px), █ (≥8px)
- `--include-distortion-model` - Output distortion heatmap alongside aligned images
  - Creates `_dist.xisf` file for each aligned image
  - Useful for quality control and troubleshooting
- `--include-detection-model` - Output detection visualization alongside aligned images
  - Creates `_det.xisf` file showing matched/unmatched stars
  - Useful for validating star detection quality

**Process (Expanding Algorithm - Default):**
1. Detect stars in reference frame
2. Select anchor stars from center (or grid) in reference and target
3. Form triangles from anchor stars
4. Match triangles by ratio similarity and vote for correspondences
5. Estimate initial similarity transform from anchor matches
6. Propagate matches using KD-tree spatial lookups
7. Refine transform with RANSAC outlier rejection
8. Re-propagate with tighter radius for final refinement
9. Apply transform based on output mode

**Process (Triangle Algorithm - Legacy):**
1. Detect stars in reference frame
2. Form triangles from brightest N reference stars
3. For each target frame:
   a. Detect stars
   b. Form triangles from brightest N target stars
   c. Match triangles by ratio similarity
   d. Vote for star correspondences
   e. Estimate similarity transform (dx, dy, rotation, scale)
   f. Apply transform based on output mode

**Output Headers:**
```
STARCOUNT =             1247 / Detected stars in target
REFSTARS  =             1302 / Detected stars in reference
MATCHED   =             4268 / Stars with correspondences
MATCHPCT  =             98.3 / Match percentage
ALIGNDX   =           -12.34 / X translation (pixels)
ALIGNDY   =             5.67 / Y translation (pixels)
ALIGNROT  =             0.12 / Rotation (degrees)
ALIGNSCL  =           1.0001 / Scale factor
STARTHRES =              5.0 / Detection threshold (sigma)
RATIOTOL  =             0.05 / Triangle ratio tolerance
ALIGNALG  =        expanding / Matching algorithm used
```

**Validation:**
- All input images must have same dimensions
- All input images must have same channel count
- Must detect minimum 10 stars for alignment

**Future Enhancement:**
- Actual image transformation in align mode (currently only match mode is functional)
- Drizzle integration for sub-pixel alignment

---

### `integrate`

Stack/combine multiple images with rejection and normalization.

**Usage:**
```bash
xisfprep integrate --input "subs/*.xisf" --output "master.xisf" \
  [--combination average] \
  [--normalization multiplicative] \
  [--rejection linearfit] \
  [--rejection-norm none] \
  [--low-sigma 2.5] \
  [--high-sigma 2.0] \
  [--iterations 3] \
  [--bias "master_bias.xisf" | --bias-level 500] \
  [--dark "master_dark.xisf"] \
  [--uncalibrated-dark] \
  [--pedestal 100] \
  [--overwrite]
```

**Arguments:**
- `--input, -i` (required) - Input image files to stack (wildcards supported)
- `--output, -o` (required) - Output integrated file path
- `--combination` (optional) - Pixel combination method (default: average)
  - `average` - Mean (default for lights)
  - `median` - Median (robust but slower)
- `--normalization` (optional) - Image normalization (default: multiplicative)
  - `none` - No normalization
  - `additive` - Additive: `P' = P + (K - m_i)`
  - `multiplicative` - Multiplicative: `P' = P * (K / m_i)` (default)
  - `additive-scaling` - Additive with scale: `P' = P*s_i + (K - m_i)`
  - `multiplicative-scaling` - Multiplicative with scale: `P' = P*s_i * (K / m_i)`
- `--rejection` (optional) - Pixel rejection algorithm (default: none)
  - `none` - No rejection (default)
  - `minmax` - Min/max clipping (requires --low-count, --high-count)
  - `sigma` - Iterative sigma clipping (requires --low-sigma, --high-sigma)
  - `linearfit` - Linear fit clipping (requires --low-sigma, --high-sigma)
- `--rejection-norm` (optional) - Normalize pixels before rejection statistics (default: none)
  - `none` - Use raw pixel values for statistics
  - `scale-offset` - `(P - mean) / stddev` normalization
  - `equalize-flux` - `P / median` normalization (useful for flats with varying exposure)
- `--low-sigma` (optional) - Low rejection threshold (default: 2.5)
- `--high-sigma` (optional) - High rejection threshold (default: 2.0)
- `--low-count` (optional) - Drop N lowest pixels (for minmax, default: 1)
- `--high-count` (optional) - Drop N highest pixels (for minmax, default: 1)
- `--iterations` (optional) - Rejection iterations (default: 3)
- `--overwrite` (flag) - Overwrite existing output file (default: skip with warning)

**Inline Calibration (optional):**

Calibrate each input frame in-memory before stacking. Eliminates intermediate file I/O when creating master darks or master flats.

- `--bias, -b` (optional) - Master bias frame for inline calibration
- `--bias-level` (optional) - Constant bias value (alternative to --bias)
- `--dark, -d` (optional) - Master dark frame for inline calibration
- `--uncalibrated-dark` (flag) - Dark is raw (not bias-subtracted); requires --bias or --bias-level
- `--pedestal` (optional) - Pedestal added after calibration [0-65535] (default: 0)

**Inline calibration use cases:**
- Creating master darks: subtract bias from raw darks before integration
- Creating master flats: subtract bias/dark from raw flats before integration

**NOT intended for:**
- Master bias creation (no calibration needed, use integrate directly)
- Light frame stacking (use separate calibrate command for full calibration with flat)

**Process:**
1. Load all input images (proven in stack_simple.fsx)
2. Calculate per-channel statistics for normalization
3. Process each pixel position across image stack:
   - Apply normalization
   - Apply rejection algorithm
   - Combine surviving pixels
4. Output stacked image with integration metadata
5. Preserve camera/telescope properties from first frame
6. Add integration history to FITS keywords

**Validation:**
- All images must have same dimensions
- All images must have same channel count
- All images must have same sample format (UInt16, etc.)
- Minimum 3 images required for rejection algorithms
- `--bias` and `--bias-level` are mutually exclusive
- `--uncalibrated-dark` requires `--dark` and (`--bias` or `--bias-level`)
- Pedestal range: [0, 65535]

**Output Metadata:**
- `ImageIntegration:NumberOfImages`
- `ImageIntegration:PixelCombination`
- `ImageIntegration:PixelRejection`
- `ImageIntegration:OutputNormalization`
- FITS `HISTORY` entries for all integration settings
- Preserves `IMAGETYP` keyword from input files if present

---

### `stats`

Calculate and display image statistics with grouping and sorting capabilities. Supports both full pixel-level analysis and fast header-only metadata scanning.

**Usage:**
```bash
# Full pixel analysis with quality metrics
xisfprep stats --input "images/*.xisf" \
  [--output stats.csv] \
  [--metrics all] \
  [--detect-stars] \
  [--group-by target,filter] \
  [--sort-by median] \
  [--sort-order desc] \
  [--min-median 1000] \
  [--max-mad 50] \
  [--min-snr 1.0]

# Fast metadata-only scan for inventory and session planning
xisfprep stats --input "session/*.xisf" \
  --header-only \
  [--group-by target,filter] \
  [--output inventory.csv]
```

**Arguments:**
- `--input, -i` (required) - Input image files (wildcards supported)
- `--output, -o` (optional) - Output CSV file path
- `--header-only` (flag) - Fast metadata-only mode (see below)
- `--metrics` (optional) - Statistics to calculate (default: basic, ignored with --header-only)
  - `basic` - Mean, median, stddev, min, max
  - `all` - Basic + MAD, SNR, file size, compression, FITS metadata
  - `histogram` - Include histogram data (256 bins)
- `--detect-stars` (flag) - Run star detection and display star statistics
  - **Smart skip**: Automatically skipped for BIAS, DARK, FLAT frames
  - Only runs for LIGHT frames where stars are expected
  - **Incompatible with --header-only**
- `--group-by` (optional) - Grouping strategy (default: none)
  - `target` - Group by OBJECT FITS keyword
  - `filter` - Group by FILTER FITS keyword
  - `target,filter` - Group by target, then filter (recommended)
  - `imagetype` - Group by frame type (BIAS, DARK, FLAT, LIGHT)
- `--sort-by` (optional) - Sort frames within groups (default: name)
  - `name` - Alphabetical by filename
  - `median` - By median pixel value (signal level) [requires pixel analysis]
  - `snr` - By signal-to-noise ratio [requires --metrics all]
  - `mad` - By median absolute deviation [requires --metrics all]
  - `fwhm` - By focus quality [requires --detect-stars]
  - `stars` - By star count [requires --detect-stars]
  - `exposure` - By exposure time [header-only mode]
  - `date` - By observation date [header-only mode]
- `--sort-order` (optional) - Sort direction (default: depends on metric)
  - `asc` - Ascending (low to high)
  - `desc` - Descending (high to low)
  - **Auto defaults**: `desc` for median/snr/stars/exposure, `asc` for mad/fwhm/name/date
- `--min-median` (optional) - Filter out frames below median threshold [requires pixel analysis]
- `--max-mad` (optional) - Filter out frames above MAD threshold [requires --metrics all]
- `--min-snr` (optional) - Filter out frames below SNR threshold [requires --metrics all]

**Output:**
- Console table with statistics per file and channel
- Grouped output with clear section headers
- Optional CSV export for analysis with grouping columns
- Star detection results (when enabled, auto-skipped for calibration frames)

**Header-Only Mode (`--header-only`):**

Fast metadata-only analysis that reads FITS headers and XISF properties **without loading pixel data**. Designed for quick inventory, session planning, and data organization.

**Performance:** 10-100x faster than full pixel analysis for large datasets

**Data Extracted:**
- File information: filename, file size, dimensions, channels
- FITS metadata: Object (target), Filter, ImageType, observation date
- Instrument metadata: exposure time, side of pier, binning
- Image geometry: width, height, channel count, sample format
- Compression: codec type

**Group Aggregation:**
When combined with `--group-by`, displays per-group statistics only (no individual file listings):
- Frame count
- Total integration time (sum of exposures)
- Average exposure time
- Date range (first to last observation)
- Total file size

Individual file listings are only shown when NOT grouping. For detailed file-level data with grouping, use CSV export via `--output`.

**Typical Use Cases:**
- Quickly inventory large archives (1000s of files in seconds)
- Plan stacking sessions by reviewing available data
- Calculate total integration time per target/filter
- Generate session reports without loading images
- Export metadata for spreadsheet analysis

**Compatible Options:**
- `--group-by` - All grouping strategies work (target, filter, imagetype, etc.)
- `--sort-by name, exposure, date` - Sorting by metadata fields
- `--output` - CSV export fully supported

**Incompatible Options:**
- `--detect-stars` - requires pixel data
- `--metrics` - ignored (metadata is always collected in header-only mode)
- `--sort-by median, snr, mad, fwhm, stars` - require pixel statistics
- `--min-median`, `--max-mad`, `--min-snr` - require pixel statistics

**Metrics - Basic (Pixel Analysis):**
- Per-channel mean, median, standard deviation
- Min/max pixel values
- Image dimensions and channel count

**Metrics - All (Pixel Analysis):**
- All basic metrics plus:
- **FITS Metadata**: Object (target), Filter, ImageType
- Median Absolute Deviation (MAD) per channel
- Signal-to-Noise Ratio (SNR) per channel
- File size (bytes, KB, MB)
- Compression codec and ratio

**Metrics - Histogram (Pixel Analysis):**
- All basic metrics plus:
- 256-bin histogram per channel
- CSV export includes Bin0...Bin255 columns

**Star Detection (--detect-stars):**
- **Smart skip**: Automatically disabled for BIAS, DARK, FLAT frames
- Only processes LIGHT frames where stars are expected
- Local background estimation (128px grid with bilinear interpolation)
- Connected component analysis for star grouping
- Per-star measurements:
  - Sub-pixel centroid position (X, Y)
  - Focus quality: FWHM, HFR
  - Photometry: peak, flux, background, SNR
  - Shape quality: eccentricity
- Quality filtering (size, FWHM range, eccentricity, saturation)
- Cross-channel matching for RGB images
- Statistical summaries:
  - Star count (capped at 20000 by default)
  - Median FWHM ± standard deviation
  - Median HFR ± standard deviation
  - Median eccentricity ± standard deviation
  - Cross-matched star count (RGB only)

**Use Cases:**

*Pixel Analysis Mode (default):*
- **Pre-filter data**: Identify low-quality frames to exclude before stacking
- **Sort by quality**: Find best/worst frames quickly (by SNR, FWHM, etc.)
- Compare calibration frame quality
- Assess image focus quality via FWHM
- Select best frames for reference
- Validate alignment quality
- **Save processing time**: Auto-skip star detection on calibration frames

*Header-Only Mode:*
- **Fast inventory**: Scan 1000s of files in seconds
- **Session planning**: Review captured data and plan integration strategy
- **Integration time calculations**: See total exposure per target/filter
- **Data organization**: Group and export metadata for tracking
- **Archive management**: Quickly catalog large image libraries

**Examples:**
```bash
# PIXEL ANALYSIS - Group Vulpecula data by filter, sort by SNR (best first)
xisfprep stats -i "Vulpecula*.xisf" --metrics all --group-by filter --sort-by snr --sort-order desc

# PIXEL ANALYSIS - Find best Ha frames (median > 500, MAD < 50, sorted by FWHM)
xisfprep stats -i "Ha*.xisf" --metrics all --detect-stars \
  --min-median 500 --max-mad 50 --sort-by fwhm -o ha_best.csv

# PIXEL ANALYSIS - Review all targets with grouping, export for Excel analysis
xisfprep stats -i "*.xisf" --metrics all --group-by target,filter -o session.csv

# PIXEL ANALYSIS - Check focus across a night (sort by FWHM, no star detection on darks/flats)
xisfprep stats -i "lights/*.xisf" --detect-stars --sort-by fwhm

# HEADER-ONLY - Fast inventory of entire night's captures grouped by target and filter
# Shows only aggregated summaries per group (no individual file listings)
xisfprep stats -i "session_2025-03-15/*.xisf" --header-only --group-by target,filter

# HEADER-ONLY - Calculate total integration time for archive (1000s of files, seconds to complete)
# Aggregates only; detailed file data exported to CSV
xisfprep stats -i "archive/**/*.xisf" --header-only --group-by target -o archive_inventory.csv

# HEADER-ONLY - Review what filters were used for each target (aggregates only)
xisfprep stats -i "Bin4x/*.xisf" --header-only --group-by target,filter --sort-by exposure

# HEADER-ONLY - List all files without grouping (shows individual file details)
xisfprep stats -i "lights/*.xisf" --header-only --sort-by date -o lights_inventory.csv
```

---

### `stars`

Detect stars and generate visualization overlay for validation.

**Usage:**
```bash
xisfprep stars --input "images/*.xisf" --output "validation/" \
  [--threshold 5.0] \
  [--grid-size 128] \
  [--min-fwhm 1.5] \
  [--max-fwhm 20.0] \
  [--max-eccentricity 0.5] \
  [--max-stars 20000] \
  [--marker circle] \
  [--scale-by fwhm] \
  [--intensity 1.0] \
  [--suffix "_stars"]
```

**Arguments:**
- `--input, -i` (required) - Input image files (wildcards supported)
- `--output, -o` (optional) - Output directory (default: same as input)
- `--suffix` (optional) - Output filename suffix (default: "_stars")
- `--overwrite` (flag) - Overwrite existing output files
- `--output-format` (optional) - Output sample format (default: preserve input)

**Detection Parameters:**
- `--threshold` (optional) - Detection threshold in sigma (default: 5.0)
- `--grid-size` (optional) - Background estimation grid size in pixels (default: 128)
- `--min-fwhm` (optional) - Minimum FWHM filter in pixels (default: 1.5)
- `--max-fwhm` (optional) - Maximum FWHM filter in pixels (default: 20.0)
- `--max-eccentricity` (optional) - Maximum eccentricity filter (default: 0.5)
- `--max-stars` (optional) - Maximum stars to detect (default: 20000)

**Visualization Options:**
- `--marker` (optional) - Marker type (default: circle)
  - `circle` - Draw circles around stars
  - `crosshair` - Draw crosshair markers
  - `gaussian` - Paint Gaussian profiles matching FWHM
- `--scale-by` (optional) - Scale marker size by property (default: fwhm)
  - `fwhm` - Scale by star FWHM
  - `flux` - Scale by star brightness
  - `fixed` - Fixed marker size
- `--intensity` (optional) - Marker brightness 0-1 (default: 1.0)

**Process:**
1. Load input image
2. Run star detection per channel with configured parameters
3. Create output image with same geometry (black background)
4. Paint synthetic markers at detected star positions
5. Copy all headers, add star statistics headers
6. Write output file

**Output Headers:**
```
STARCOUNT =             1247 / Number of detected stars
MEDFWHM   =             3.42 / Median FWHM in pixels
MEDHFR    =             2.01 / Median HFR in pixels
MEDECCEN  =            0.089 / Median eccentricity
MEDSNR    =             45.2 / Median signal-to-noise
STARTHRES =              5.0 / Detection threshold (sigma)
STARGRID  =              128 / Background grid size
```

**Use Cases:**
- Validate star detection parameters before alignment
- Visualize detection quality across different images
- Debug detection issues (too many/few stars, wrong threshold)
- Compare detection across filters or conditions

**Examples:**
```bash
# Basic - detect and visualize with defaults
xisfprep stars -i "lights/*.xisf" -o "validation/"

# Tune detection for narrowband (lower threshold, larger FWHM range)
xisfprep stars -i "Ha*.xisf" -o "validation/" --threshold 3.0 --max-fwhm 25.0

# Crosshair markers scaled by flux
xisfprep stars -i "image.xisf" -o "check/" --marker crosshair --scale-by flux

# Gaussian profiles for accurate FWHM visualization
xisfprep stars -i "focus_test.xisf" --marker gaussian --scale-by fwhm
```

---

### `convert`

Format conversion and export.

**Usage:**
```bash
xisfprep convert --input "result.xisf" --output "result.fits" \
  [--format fits] \
  [--compression none]
```

**Arguments:**
- `--input, -i` (required) - Input XISF file
- `--output, -o` (required) - Output file path
- `--format` (optional) - Output format (default: inferred from extension)
  - `xisf` - XISF format
  - `fits` - FITS format
  - `tiff` - 16-bit TIFF
- `--compression` (optional) - Compression codec (default: lz4hcsh for XISF)
  - `none` - Uncompressed
  - `lz4` - LZ4 fast compression
  - `lz4hc` - LZ4 high compression
  - `lz4sh` - LZ4 with byte shuffling
  - `lz4hcsh` - LZ4HC with byte shuffling (default)
  - `zlib` - Zlib compression
  - `zlibsh` - Zlib with byte shuffling

**Process:**
- Read XISF input
- Convert to target format
- Preserve metadata when supported
- Apply compression settings

**Validation:**
- Target format must support sample format
- Warns if metadata will be lost

---

### `bin`

Downsample images by binning pixels together.

**Usage:**
```bash
xisfprep bin --input "images/*.xisf" --output "binned/" \
  [--factor 2] \
  [--method average]
```

**Arguments:**
- `--input, -i` (required) - Input image files (wildcards supported)
- `--output, -o` (required) - Output directory for binned files
- `--factor` (optional) - Binning factor: 2, 3, 4, 5, or 6 (default: 2)
- `--method` (optional) - Binning method (default: average)
  - `average` - Average binning (preserves flux, default)
  - `median` - Median binning (robust to outliers)
  - `sum` - Sum binning (adds pixel values)
- `--suffix <text>` (optional) - Output filename suffix (default: _binNx where N is factor)

**Process:**
1. Group pixels in NxN blocks based on binning factor
2. Apply selected method to each block
3. Result image is 1/N width and 1/N height
4. Update FITS keywords for pixel scale and binning
5. Preserve all other metadata

**Use Cases:**
- Quick preview of large images
- Reduce file size for testing workflows
- Match image scales between different instruments
- Speed up processing for alignment tests

**Validation:**
- Image dimensions must be divisible by binning factor
- Binning factor must be 2, 3, 4, 5, or 6
- Warns if image dimensions will be truncated

**Examples:**
```bash
xisfprep bin -i "lights/*.xisf" -o "binned/" --factor 2
xisfprep bin -i "lights/*.xisf" -o "preview/" --factor 4 --method median
```

---

### `inspect`

Diagnostic inspection of XISF file structure and metadata.

**Usage:**
```bash
xisfprep inspect --input "file.xisf" \
  [--raw-xml] \
  [--attachments] \
  [--preview]
```

**Arguments:**
- `--input, -i` (required) - Input XISF file to inspect
- `--raw-xml` (flag) - Dump raw XML header to console
- `--attachments` (flag) - Show detailed attachment block analysis
- `--preview` (flag) - Show 128x128 pixel sample statistics from image center

**Output:**
Default output includes:
- File header validation (signature, header length)
- XML structure overview (metadata properties, image geometry)
- Image details (format, color space, dimensions, key FITS keywords)
- Validation summary

**Use Cases:**
- Debug problematic XISF files that won't load
- Verify file integrity before processing
- Understand file structure and metadata layout
- Diagnose attachment block issues
- Quick inspection of key FITS keywords (OBJECT, FILTER, EXPTIME, etc.)

**Examples:**
```bash
# Basic inspection
xisfprep inspect -i "image.xisf"

# Show full XML header for debugging
xisfprep inspect -i "image.xisf" --raw-xml

# Full diagnostic with attachment analysis and pixel preview
xisfprep inspect -i "image.xisf" --attachments --preview
```

---

## Sample Format Support

### Overview

XISF supports multiple sample formats for pixel data. XisfPrep processes all formats internally as floating-point [0-65535] range, then outputs in the requested format.

**Supported Input Formats:**
- `UInt8` - 8-bit unsigned integer [0-255]
- `UInt16` - 16-bit unsigned integer [0-65535] (most common from cameras)
- `UInt32` - 32-bit unsigned integer [0-4294967295]
- `Float32` - 32-bit float (PixInsight standard for processed images)
- `Float64` - 64-bit double precision float

**Unsupported Formats:**
- `UInt64`, `Complex32`, `Complex64` - Will fail with clear error message

### Processing Pipeline

**1. Input → Internal Representation (PixelIO.readPixelsAsFloat)**
- Reads source format from XISF metadata
- Converts to `float[]` in normalized [0-65535] ADU range
- **Auto-denormalization for PixInsight compatibility:**
  - Detects `bounds="0:1"` attribute (PixInsight standard)
  - If present: multiplies Float32/Float64 values by 65535
  - If absent: uses raw float values
  - Ensures all processing works in consistent ADU range

**2. Processing**
- All math operations (calibration, binning, integration, etc.) work on `float[]` at [0-65535] range
- Preserves full precision during calculations
- No format-specific code in processing logic

**3. Internal → Output (PixelIO.writePixelsFromFloat)**
- Converts from internal [0-65535] range to target format
- **Auto-normalization for Float32/Float64 output:**
  - Divides by 65535 to produce [0,1] range
  - Sets `bounds="0:1"` attribute (XISF spec REQUIRED for float formats)
- **Integer formats (UInt8, UInt16, UInt32):**
  - Clamps to appropriate range ([0-255], [0-65535], [0-4294967295])
  - Does NOT set bounds attribute (XISF spec: bounds SHOULD NOT be specified for default ranges)
  - Uses implicit default range per spec: [0, 2^n-1]

### Default Output Format Behavior

**By default, each command preserves input format:**

| Input Format | Default Output | Rationale |
|--------------|----------------|-----------|
| UInt8 | UInt8 | Preserve original precision |
| UInt16 | UInt16 | Most common camera format |
| UInt32 | UInt32 | Preserve high bit depth |
| Float32 | Float32 [0,1] | PixInsight compatibility |
| Float64 | Float64 [0,1] | Preserve maximum precision |

**Exception - `calibrate` command:**
- **Uses max precision of all inputs** (light frame + all master frames)
- Rationale: Avoid precision loss when masters are higher precision than lights
- Example: UInt16 lights + Float32 masters → Float32 output
- Precision hierarchy: UInt8 < UInt16 < UInt32 < Float32 < Float64

**Exception - `integrate` command:**
- **UInt16 inputs → Float32 output** (default)
- Rationale: Stacking multiple UInt16 images produces fractional pixel values that exceed UInt16 range
- Example: Average of 10 images with values [5000, 5001, ...] = 5000.5 (requires float)
- Preserves stacking precision and dynamic range
- Already Float32/Float64 inputs → Preserve input format

### Global `--output-format` Flag

Override default output format on any command that produces image outputs.

**Syntax:**
```bash
--output-format <format>
```

**Supported format values:**
- `uint8`, `u8` - 8-bit unsigned integer
- `uint16`, `u16` - 16-bit unsigned integer
- `uint32`, `u32` - 32-bit unsigned integer
- `float32`, `f32`, `float` - 32-bit float [0,1] with bounds
- `float64`, `f64`, `double` - 64-bit double [0,1] with bounds

**Examples:**
```bash
# Force UInt16 output for Float32 PixInsight master bias
xisfprep calibrate -i "lights/*.xisf" -o "cal/" --bias "master_bias_float32.xisf" --output-format uint16

# Preserve maximum precision during calibration
xisfprep calibrate -i "lights/*.xisf" -o "cal/" --bias bias.xisf --output-format float32

# Force Float32 for downstream PixInsight processing
xisfprep bin -i "images/*.xisf" -o "binned/" --factor 2 --output-format float32
```

**Available on commands:**
- `calibrate` - Calibrated light frames
- `debayer` - RGB images
- `bin` - Binned images
- `integrate` - Stacked/integrated images
- NOT available on `stats` - No image output
- NOT available on `convert` - Has its own `--format` flag

### Downconversion Warnings

**When downconverting from higher to lower precision, XisfPrep warns:**

```bash
# Float32 → UInt16 (precision loss warning)
Warning: Downconverting from Float32 to UInt16 - precision will be lost

# Float64 → Float32 (precision loss warning)
Warning: Downconverting from Float64 to Float32 - precision will be reduced

# UInt32 → UInt16 (range loss warning)
Warning: Downconverting from UInt32 to UInt16 - values above 65535 will be clamped
```

**No warning for:**
- Same format (Float32 → Float32)
- Upconversion (UInt16 → Float32)
- Standard integrate behavior (UInt16 → Float32)

### PixInsight Compatibility

**PixInsight Float32 Standard:**
- Processed images: Float32 with `bounds="0:1"`
- Master calibration frames: Float32 normalized [0,1]
- Calibrated lights: Float32 normalized [0,1]

**XisfPrep Compatibility:**
✓ **Reading PixInsight Float32 files:**
- Detects `bounds="0:1"` attribute
- Auto-denormalizes [0,1] → [0-65535] for processing
- Example: PixInsight master bias value 0.015 → 983 ADU internally

✓ **Writing Float32 for PixInsight:**
- Automatically normalizes [0-65535] → [0,1]
- Sets `bounds="0:1"` attribute
- PixInsight recognizes and processes correctly

✓ **Round-trip:**
- PixInsight → XisfPrep → PixInsight preserves format
- No precision loss with `--output-format float32`

**XISF Specification Compliance:**

Per XISF 1.0 Specification (Section 8.6.4 - Image Core Element):

✅ **bounds attribute for Float32/Float64:**
- **REQUIRED** by spec for floating-point images
- Must be declared explicitly (no default)
- XisfPrep: Always sets `bounds="0:1"` for Float32/Float64 output

✅ **bounds attribute for UInt8/UInt16/UInt32:**
- **OPTIONAL** by spec for integer images
- **SHOULD NOT be specified** for default [0, 2^n-1] ranges
- Default if omitted: [0, 255] for UInt8, [0, 65535] for UInt16, [0, 4294967295] for UInt32
- XisfPrep: Does NOT set bounds for integer outputs (uses implicit spec defaults)

✅ **Linear data processing:**
- Spec recognizes "linear images, such as CCD and digital camera raw frames"
- Display functions are for visualization only, NOT preprocessing
- XisfPrep: All preprocessing operates on linear data in ADU range

### Sample Format Examples

**Example 1: PixInsight Master Bias → NINA Lights Calibration**
```bash
# Input: PixInsight Float32 master bias with bounds="0:1", value 0.015
# Input: NINA UInt16 lights, value 5000
# Processing:
#   - Bias: 0.015 * 65535 = 983 (denormalized)
#   - Light: 5000 (already in ADU)
#   - Result: 5000 - 983 = 4017
# Output: UInt16 value 4017 (preserves NINA format)
```

**Example 2: Integration with Precision Preservation**
```bash
xisfprep integrate -i "lights_uint16/*.xisf" -o "stacked.xisf"
# Input: 10 UInt16 frames, pixel values [5000, 5001, 5002, ..., 5009]
# Processing: Average = 5004.5 (fractional precision from stacking)
# Output: Float32 value 0.076385 (5004.5 / 65535) with bounds="0:1"
# Rationale: Float32 preserves fractional ADU from averaging
```

**Example 3: Binning with Format Preservation**
```bash
xisfprep bin -i "float32_image.xisf" -o "binned/" --factor 2
# Input: Float32 [0,1], pixels [0.5, 0.6, 0.7, 0.8]
# Processing:
#   - Read: [32767, 39321, 45875, 52428] (denormalized to ADU)
#   - Bin average: (32767 + 39321 + 45875 + 52428) / 4 = 42597.75
# Output: Float32 value 0.650345 (42597.75 / 65535) with bounds="0:1"
# Preserves Float32 format for PixInsight workflow
```

**Example 4: Force UInt16 for PixInsight Masters**
```bash
# PixInsight master bias is Float32, but we want UInt16 pipeline
xisfprep calibrate -i "darks/*.xisf" -o "darks_cal/" \
  --bias "pixinsight_master_bias_float32.xisf" \
  --output-format uint16

# Input: Float32 bias with bounds, value 0.015
# Processing: 0.015 * 65535 = 983 ADU (denormalized)
# Output: UInt16 (forced via --output-format)
# Allows pure UInt16 pipeline even with PixInsight masters
```

### Implementation Notes

**PixelIO Module Responsibilities:**
- Format detection from XISF metadata
- Bounds attribute detection for normalization
- Byte-level reading/writing for all formats
- Auto-denormalization for Float inputs with bounds
- Auto-normalization for Float outputs
- Downconversion warnings

**Command Responsibilities:**
- Parse `--output-format` flag (optional)
- Determine default output format based on command and input
- Pass format to PixelIO for writing
- Display downconversion warnings to user

**Validation:**
- Unsupported input formats fail early with clear message
- Invalid `--output-format` values show supported options
- Dimension/channel validation happens before format conversion

---

## Common Patterns

### Typical Calibration Workflow

```bash
# 1. Create master bias
xisfprep integrate --input "bias/*.xisf" --output "master_bias.xisf" \
  --combination median --normalization none --rejection sigma

# 2. Create master dark (with inline bias calibration - single command)
xisfprep integrate --input "darks/*.xisf" --output "master_dark.xisf" \
  --bias "master_bias.xisf" \
  --combination median --normalization none --rejection sigma

# 3. Create master flat (with inline bias/dark calibration - single command)
xisfprep integrate --input "flats/*.xisf" --output "master_flat.xisf" \
  --bias "master_bias.xisf" --dark "master_dark.xisf" --uncalibrated-dark \
  --combination median --normalization multiplicative \
  --rejection sigma --rejection-norm equalize-flux

# 4. Calibrate light frames
xisfprep calibrate --input "lights/*.xisf" --output "lights_cal/" \
  --bias "master_bias.xisf" --dark "master_dark.xisf" --flat "master_flat.xisf"

# 5. Debayer (if Bayer sensor)
xisfprep debayer --input "lights_cal/*.xisf" --output "lights_rgb/"

# 6. Align
xisfprep align --input "lights_rgb/*.xisf" --output "lights_aligned/" --auto-reference

# 7. Integrate final image
xisfprep integrate --input "lights_aligned/*.xisf" --output "final.xisf" \
  --normalization multiplicative --rejection linearfit --low-sigma 2.5 --high-sigma 2.0
```

### Traditional Workflow (separate calibrate + integrate)

For cases where you want intermediate calibrated files:

```bash
# 2a. Create master dark (traditional two-step)
xisfprep calibrate --input "darks/*.xisf" --output "darks_cal/" --bias "master_bias.xisf"
xisfprep integrate --input "darks_cal/*.xisf" --output "master_dark.xisf" \
  --combination median --normalization none

# 3a. Create master flat (traditional two-step)
xisfprep calibrate --input "flats/*.xisf" --output "flats_cal/" \
  --bias "master_bias.xisf" --dark "master_dark.xisf"
xisfprep integrate --input "flats_cal/*.xisf" --output "master_flat.xisf" \
  --combination median --normalization multiplicative \
  --rejection sigma --rejection-norm equalize-flux
```

### Monochrome Narrowband Workflow

```bash
# Process Ha channel (use pedestal for very dark sky backgrounds)
xisfprep calibrate --input "Ha/*.xisf" --output "Ha_cal/" \
  --bias bias.xisf --dark dark.xisf --flat flat_Ha.xisf --pedestal 100
xisfprep align --input "Ha_cal/*.xisf" --output "Ha_aligned/" --auto-reference
xisfprep integrate --input "Ha_aligned/*.xisf" --output "Ha_master.xisf" --rejection linearfit

# Process OIII channel
xisfprep calibrate --input "OIII/*.xisf" --output "OIII_cal/" \
  --bias bias.xisf --dark dark.xisf --flat flat_OIII.xisf --pedestal 100
xisfprep align --input "OIII_cal/*.xisf" --output "OIII_aligned/" --auto-reference
xisfprep integrate --input "OIII_aligned/*.xisf" --output "OIII_master.xisf" --rejection linearfit
```

### Creating Master Darks with Bias Constant

```bash
# Single command with inline bias constant subtraction
xisfprep integrate --input "darks/*.xisf" --output "master_dark.xisf" \
  --bias-level 500 \
  --combination median --normalization none --rejection sigma
```

Traditional two-step approach (creates intermediate files):
```bash
xisfprep calibrate --input "darks/*.xisf" --output "darks_cal/" --bias-level 500
xisfprep integrate --input "darks_cal/*.xisf" --output "master_dark.xisf" \
  --combination median --normalization none
```

---

## Global Options

Available for all commands:

```bash
--verbose, -v       Verbose output with progress details
--quiet, -q         Suppress all output except errors
--parallel N        Number of parallel threads (default: CPU count)
--overwrite         Overwrite existing output files (default: skip existing)
--help, -h          Show help for command
--version           Show version information
```

---

## Error Handling

**Validation Errors:**
- Clear error messages with file paths
- Early validation before processing begins
- Suggestion for common mistakes

**Processing Errors:**
- Continue processing remaining files when possible
- Report errors with context (file name, pixel position)
- Exit code reflects success/partial/failure

**Examples:**
```
Error: Cannot debayer 'image.xisf' - already RGB (3 channels)
Error: Dimension mismatch - Light: 6252x4176, Flat: 6248x4172
Warning: No BAYERPAT keyword found, using default RGGB
Warning: Output file 'calibrated/light_001_cal.xisf' already exists, skipping (use --overwrite to replace)
Info: Overwriting existing file 'calibrated/light_002_cal.xisf'
```

---

## Output Conventions

**File Naming:**
- Default suffix appended to original filename
- Output directory structure preserves input structure

**File Overwrite Protection:**
- **Default behavior: NEVER overwrite existing files**
- When output file already exists without `--overwrite` flag:
  - Warn: `Warning: Output file 'path/to/file.xisf' already exists, skipping (use --overwrite to replace)`
  - Skip processing for that file
  - Continue processing remaining files
- With `--overwrite` flag:
  - Info: `Overwriting existing file 'path/to/file.xisf'`
  - Replace existing output files
- Applies to all output types: image files, CSV exports, single-file outputs

**Metadata Preservation:**
- Copy all FITS keywords from input (unless modified by operation)
- Preserve camera properties (gain, binning, pixel size)
- Preserve telescope properties (focal length, aperture)
- Add processing history to FITS HISTORY keywords

**Compression:**
- Default: LZ4HC with shuffling (best balance)
- User can override per command
- Uncompressed option available for compatibility

---

## Performance Targets

**Throughput:**
- Calibration: 50+ images/minute (6K × 4K UInt16)
- Debayer: 30+ images/minute (VNG algorithm)
- Integration: 300 images in <5 minutes (with rejection)

**Memory:**
- Streaming processing where possible
- Cache only what's needed for rejection algorithms
- Parallel processing without excessive RAM

**Feedback:**
- Progress indicators for long operations
- File count and estimated time remaining
- Summary statistics on completion

---

## Dependencies

**XisfLib.Core:**
- XISF reading and writing
- Compression/decompression
- Metadata handling

**F# Libraries:**
- FSharp.Core for functional operations
- System.CommandLine for CLI parsing
- System.Numerics for SIMD operations (future)

**External:**
- K4os.Compression.LZ4 (transitive from XisfLib.Core)

---

## Future Enhancements

**Phase 2:**
- Cosmetic correction (hot pixel removal)
- Image transformation output in align mode
- Drizzle integration
- GPU acceleration for debayer and alignment

**Phase 3:**
- ✅ Star detection and analysis (IMPLEMENTED)
- ✅ Quality scoring (FWHM, eccentricity) (IMPLEMENTED)
- Auto-selection of best frames
- Plate solving integration

**Phase 4:**
- Scripted workflows (batch file processing)
- Configuration file support
- Plugin architecture for custom operations

---

## Project Structure

```
XisfPrep/
├── XisfPrep.fsproj          Core CLI project
├── Program.fs               Entry point and command dispatcher
├── Commands/
│   ├── Calibrate.fs         Calibration command
│   ├── Debayer.fs           Debayer command (from debayer.fsx)
│   ├── Align.fs             Alignment command
│   ├── Integrate.fs         Integration command (from stack_simple.fsx)
│   ├── Stats.fs             Statistics command
│   ├── Convert.fs           Conversion command
│   └── Inspect.fs           Diagnostic inspection command
├── Core/
│   ├── FileDiscovery.fs     Wildcard expansion and file listing
│   ├── Validation.fs        Input validation
│   ├── Progress.fs          Progress reporting
│   └── Metadata.fs          Metadata utilities
└── Algorithms/
    ├── Statistics.fs        Image statistics
    ├── Painting.fs          Visualization primitives
    ├── OutputImage.fs       Output image generation
    ├── StarDetection.fs     Star detection with centroid fitting
    ├── SpatialMatch.fs      KD-tree, RANSAC, expanding match algorithm
    └── TriangleMatch.fs     Triangle ratio matching (legacy)

Tests/
└── XisfPrep.Tests/
    ├── CalibrationTests.fs
    ├── DebayerTests.fs
    ├── IntegrationTests.fs
    └── RoundTripTests.fs
```

---

## Success Criteria

**Functionality:**
- Process real-world NINA capture sessions end-to-end
- Produce results matching reference software (PixInsight, ASTAP)
- Handle both monochrome and OSC (Bayer) workflows

**Usability:**
- Intuitive command structure
- Clear error messages
- Reasonable defaults minimize required arguments

**Performance:**
- Process typical session (300 subs) in under 10 minutes
- Memory usage scales with image size, not file count
- Parallel processing utilizes all CPU cores

**Quality:**
- Preserves 16-bit precision throughout pipeline
- Maintains metadata provenance
- Produces clean, artifact-free results
