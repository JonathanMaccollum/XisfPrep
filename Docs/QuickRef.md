# XisfPrep Quick Reference - Test Commands

Ready-to-run test commands for each XisfPrep command.

**Output Directory:** `D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs`

---

## stats

### Metadata-Only (Fast)
```bash
# All-Sky OSC lights - quick inventory
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*.xisf" --header-only

# Mono darks with grouping
dotnet run -- stats -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*.xisf" --header-only --group-by imagetype --sort-by date

# Grouped by target and filter
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINA\*.xisf" --header-only --group-by target,filter --sort-by exposure
```

### Full Pixel Analysis
```bash
# Single file with all metrics
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" --metrics all

# Multiple files with SNR sorting
dotnet run -- stats -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" --metrics all --sort-by snr
```

### Star Detection
```bash
# Single file with star detection
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Sii25nm_LIGHT_2025-07-04_01-39-46_0006_720.00s_2.00_High Gain 2CMS_0.50.xisf" --detect-stars

# With full metrics
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Sii25nm_LIGHT_2025-07-04_01-39-46_0006_720.00s_2.00_High Gain 2CMS_0.50.xisf" --detect-stars --metrics all

# Sort multiple files by FWHM (best focus first)
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINA\*.xisf" --detect-stars --sort-by fwhm

# Sort by star count
dotnet run -- stats -i "D:\Backups\Camera\Dropoff\NINA\*.xisf" --detect-stars --sort-by stars --sort-order desc
```

---

## stars

### Basic Star Visualization
```bash
# Single file - generate star overlay
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Sii25nm_LIGHT_2025-07-04_01-39-46_0006_720.00s_2.00_High Gain 2CMS_0.50.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite

# Dense Star Field
dotnet run -- stars -i "W:\Astrophotography\350mm\Vulpecula - LDN 792 LDN 784 LDN 768 and Stock 1\360.00s\Vulpecula - LDN 792 LDN 784 LDN 768 and Stock 1_G_LIGHT_2024-07-05_01-37-02_0007_360.00s_-0.00_0.35.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite


# Multiple files
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite
```

### Custom Detection Parameters
```bash
# Lower threshold for narrowband (more sensitive)
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Sii25nm_LIGHT_2025-07-04_01-39-46_0006_720.00s_2.00_High Gain 2CMS_0.50.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --threshold 3.0 --overwrite

# Wider FWHM range for out-of-focus images
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --min-fwhm 1.0 --max-fwhm 30.0 --overwrite
```

### Visualization Options
```bash
# Crosshair markers scaled by flux
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --marker crosshair --scale-by flux --overwrite

# Gaussian profiles matching FWHM
dotnet run -- stars -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --marker gaussian --overwrite
```

---

## align

### Detect Mode (Star Detection Validation)
```bash
# Single narrowband file
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Sii25nm_LIGHT_2025-07-04_01-39-46_0006_720.00s_2.00_High Gain 2CMS_0.50.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode detect --overwrite

# Dense star field
dotnet run -- align -i "W:\Astrophotography\350mm\Vulpecula - LDN 792 LDN 784 LDN 768 and Stock 1\360.00s\Vulpecula - LDN 792 LDN 784 LDN 768 and Stock 1_G_LIGHT_2024-07-05_01-37-02_0007_360.00s_-0.00_0.35.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode detect --overwrite

# Multiple files with custom detection params
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode detect --threshold 3.0 --overwrite
```

### Match Mode (Correspondence Validation)
```bash
# Default expanding algorithm (fast, accurate)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode match --overwrite

# More anchor stars for difficult fields
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode match --anchor-stars 20 --overwrite

# Grid distribution for wide-field images
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode match --anchor-spread grid --overwrite

# Triangle algorithm (legacy, more triangles)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode match --algorithm triangle --max-stars-triangles 50 --overwrite

# Dense star field matching
dotnet run -- align -i "W:\Astrophotography\350mm\Vulpecula - LDN 792 LDN 784 LDN 768 and Stock 1\360.00s\*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode match --overwrite
```

### Align Mode (Image Transformation)
```bash
# Full alignment with Lanczos-3 interpolation (default)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion wendland --include-distortion-model --include-detection-model --overwrite

# With bicubic interpolation (faster)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --interpolation bicubic --overwrite
```

### RBF Distortion Correction
```bash
# Wendland distortion correction (recommended - local, fast)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion wendland --overwrite

# TPS distortion correction (global, classic)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion tps --overwrite

# IMQ distortion correction (inverse multiquadric)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion imq --overwrite

# Custom Wendland support radius
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion wendland --rbf-support 4.0 --overwrite
```

### Diagnostic Outputs (Quality Control)
```bash
# Output distortion heatmap alongside aligned images (for QC)
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion wendland --include-distortion-model --overwrite

# Output detection visualization alongside aligned images
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --include-detection-model --overwrite

# Output both diagnostic files with aligned images
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --anchor-stars 20 --distortion wendland --include-distortion-model --include-detection-model --overwrite
```

### Distortion Visualization Mode
```bash
# Visualize distortion field with Wendland
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode distortion --distortion wendland --anchor-stars 20 --overwrite

# Visualize with TPS for comparison
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-mode distortion --distortion tps --anchor-stars 20 --overwrite

# Debug single file with console heatmap (check control point distribution and RMS)
# IMPORTANT: Must specify --reference to match your alignment run
dotnet run -- align -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT_2025-08-11_00-16-06_0014*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --reference "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT_2025-08-10_21-19-25_0000_720.00s_2.40_High Gain 2CMS_0.30.xisf" --output-mode distortion --distortion wendland --anchor-stars 20 --show-distortion-stats --overwrite
```

---

## bin

### Mono Data (Safe to Bin)
```bash
# Bin mono darks by 2x (average)
dotnet run -- bin -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\_Ha_DARK_2022-10-25_19-45-30_0000_360.00s_-10.00_0.00.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --factor 2 --overwrite

# Bin 4x with median
dotnet run -- bin -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --factor 4 --method median --overwrite
```

### Debayered RGB Data
```bash
# First debayer, then bin (correct workflow for OSC)
# Step 1: Debayer
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite

# Step 2: Bin the RGB output
dotnet run -- bin -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\*_15233_*_d.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --factor 2 --overwrite
```

### Sample Test Images
```bash
# Test images from sample folder
dotnet run -- bin -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Sample*Images\*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --factor 2 --overwrite
```

---

## calibrate

### Constant Bias Level
```bash
# Quick test with constant bias
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias-level 500 --overwrite
```

### Raw NINA Bias (UInt16)
```bash
# Single dark with raw bias
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\_Ha_DARK_2022-10-25_19-45-30_0000_360.00s_-10.00_0.00.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\Raw\_Ha_BIAS_2022-07-21_21-07-17_0008_0.10s_22.30_0.00.xisf" --pedestal 100 --overwrite

# All darks with raw bias + pedestal
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\Raw\_Ha_BIAS_2022-07-21_21-07-17_0008_0.10s_22.30_0.00.xisf" --pedestal 100 --overwrite
```

### PixInsight Float32 Master Bias (Tests PixelIO)
```bash
# Single dark with PI Float32 master - tests auto-denormalization
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\_Ha_DARK_2022-10-25_19-45-30_0000_360.00s_-10.00_0.00.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\20221025.MasterBias.Gain.0.Offset.25.300x0.01s.xisf" --pedestal 100 --overwrite

# All darks with PI Float32 master
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\20221025.MasterBias.Gain.0.Offset.25.300x0.01s.xisf" --pedestal 100 --overwrite
```

### All-Sky OSC Calibration (No Flats)
```bash
# Calibrate All-Sky OSC with bias (subset for testing)
dotnet run -- calibrate -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_1523*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias-level 100 --pedestal 50 --overwrite
```

---

## debayer

### All-Sky OSC Images
```bash
# Single file debayer
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite

# Multiple files with explicit pattern
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_1523*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --pattern RGGB --overwrite

# First 10 All-Sky files
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_1523[7-9]_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite

# Parallel processing test
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_137[0-2]*_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --parallel 8 --overwrite
```

### Output Format Override
```bash
# Debayer UInt16 input to Float32 output (for PixInsight compatibility)
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-format float32 --overwrite

# Force UInt16 output (default for camera data)
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --output-format uint16 --overwrite
```

### Split to Separate R/G/B Channels
```bash
# Split into separate monochrome R, G, B files (for LRGB-style processing)
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --split --overwrite
```

### Pipeline Test: Calibrate then Debayer
```bash
# Step 1: Calibrate OSC with bias
dotnet run -- calibrate -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*_15233_*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias-level 100 --pedestal 50 --overwrite

# Step 2: Debayer calibrated output
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\*_15233_*_cal.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overwrite
```

---

## hotpixel

### Basic Detection and Correction
```bash
# Single file with default settings (k=5.0)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT_2025-08-10_21-19-25_0000*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"

# Multiple files with verbose logging
dotnet run -- --verbose hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"

# Process subset for testing
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT_2025-08-10_21-*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"
```

### Custom Thresholds
```bash
# More aggressive detection (lower k = more pixels flagged)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --k 3.0

# Conservative detection (higher k = fewer pixels flagged)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --k 7.0

# Asymmetric thresholds (more aggressive for hot pixels)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --k-hot 4.0 --k-cold 6.0
```

### Star Protection (Critical for Preserving Real Stars)
```bash
# Ratio-based protection (recommended, default ratio=1.5)
# Protects stars by checking if pixel >> neighbors (hot pixels are 3-10x brighter)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --star-protection ratio

# Custom ratio threshold for undersampled stars (all-sky, wide-field)
# Lower ratio = more aggressive correction, but test PSF to ensure stars not damaged
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --star-protection ratio --star-ratio 1.2 --bayer RGGB

# Isolation-based protection (spatial filter: only corrects ≤1 bright neighbor)
# May miss hot pixels near bright stars
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --star-protection isolation

# No protection (testing/comparison - WILL damage stars!)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --star-protection none
```

**Star Protection Guide:**
- **Ratio 1.5** (default): Safe for oversampled stars (FWHM >2.5px), excellent hot pixel removal
- **Ratio 1.0-1.3**: Undersampled/all-sky images - TEST PSF first! May damage dim stars
- **Ratio >2.0**: Very conservative, may leave hot pixels near stars
- **Isolation**: Good star preservation but misses hot pixels adjacent to stars
- **None**: No protection - use only for comparison or non-stellar data

**IMPORTANT:** Always validate with PSF measurement (amplitude, FWHM, flux should be identical before/after)

### Tiling Configuration
```bash
# Faster processing with less overlap (16px instead of default 24px)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overlap 16

# More overlap for better blending (32px)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --overlap 32
```

### Bayer/CFA Data
```bash
# Process CFA before debayering (recommended workflow)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bayer RGGB --star-protection ratio --star-ratio 1.2

# Undersampled all-sky CFA - requires careful ratio tuning
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\All-Sky 2025-11-17__LIGHT_2025-11-18_00-19-51_15233_15.00s_4.80_0.00.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bayer RGGB --star-protection ratio --star-ratio 1.1 --overlap 16
```

### Debayered RGB Data
```bash
# Process debayered RGB images (no --bayer flag)
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\*_d.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"
```

### Pipeline Integration
```bash
# Typical workflow: Calibrate → HotPixel → Debayer → Align
# Step 1: Calibrate (if you have calibration frames)
dotnet run -- calibrate -i "D:\Backups\Camera\Dropoff\NINA\Sh2-108 Take 3_Ha3nm_LIGHT*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias-level 100 --pedestal 50

# Step 2: Hot/Cold pixel correction on calibrated Bayer data
dotnet run -- hotpixel -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\*_cal.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bayer RGGB

# Step 3: Debayer
dotnet run -- debayer -i "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\*_cal_corrected.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"
```

**Algorithm Details:**
- 128×128 tile processing with configurable overlap
- Local + global statistics (median, MAD) per color plane
- VNG-style directional gradient interpolation (preserves edges)
- Star protection via flux ratio or morphological isolation
- Processes 2500+ tiles per 26MP image in ~12 seconds
- Typical correction rate: 0.1-2% of pixels (depends on sensor/temperature)

**XISF Metadata Written:**
- Full processing parameters (k-hot, k-cold, star protection method/ratio, bayer pattern)
- Correction statistics (total corrected, hot count, cold count)
- Processing timestamp and version

---

## integrate

### Basic Stacking (UInt16 - pending PixelIO update)
```bash
# Stack first 5 darks (average)
dotnet run -- integrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\_Ha_DARK_2022-10-25_19-*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\master_dark.xisf" --overwrite

# Stack with median combination
dotnet run -- integrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\master_dark_median.xisf" --combination median --overwrite
```

### With Rejection
```bash
# Sigma clipping
dotnet run -- integrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\master_dark_sigma.xisf" --rejection sigma --low-sigma 2.5 --high-sigma 2.0 --overwrite

# Min/max clipping
dotnet run -- integrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\*360.00s*.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\master_dark_minmax.xisf" --rejection minmax --low-count 1 --high-count 1 --overwrite
```

---

## Diagnostics

### Dump XISF Metadata
```bash
# Raw NINA file
dotnet run --project XisfReaderPoc -- --dump "D:\Backups\Camera\Dropoff\NINACS\All-Sky\2025-11-17\All-Sky 2025-11-17__LIGHT_2025-11-17_17-45-03_15233_15.00s_10.10_0.00.xisf"

# PixInsight Float32 master
dotnet run --project XisfReaderPoc -- --dump "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\20221025.MasterBias.Gain.0.Offset.25.300x0.01s.xisf"

# Calibrated output
dotnet run --project XisfReaderPoc -- --dump "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs\_Ha_DARK_2022-10-25_19-45-30_0000_360.00s_-10.00_0.00_cal.xisf"
```

---

## Test File Inventory

| Type | Path | Count | Format | Notes |
|------|------|-------|--------|-------|
| All-Sky OSC | `D:\...\All-Sky\2025-11-17\*.xisf` | 2975 | UInt16 | Linear Bayer, 15s exposure |
| Mono Darks | `W:\...\DarkLibrary\...\20221025\*360.00s*.xisf` | 28 | UInt16 | 6m exposure, ASI533MM |
| PI Master Bias | `W:\...\BiasLibrary\...\20221025.MasterBias.*.xisf` | 1 | Float32 | Normalized [0,1] |
| Raw NINA Bias | `W:\...\BiasLibrary\...\Raw\_Ha_BIAS_*.xisf` | 1 | UInt16 | Raw acquisition |
| Sample Images | `D:\...\Testing\Sample*Images\*.xisf` | varies | UInt16 | Various test files |

---

## Output Directory Setup

```bash
# Ensure output directory exists
mkdir "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs"
```

---

## Float32 Format Tests (PixelIO Validation)

### Test PixInsight Interop
```bash
# UInt16 input + Float32 master → should auto-denormalize
dotnet run -- calibrate -i "W:\Astrophotography\DarkLibrary\ZWO ASI533MM Pro\20221025\_Ha_DARK_2022-10-25_19-45-30_0000_360.00s_-10.00_0.00.xisf" -o "D:\Backups\Camera\Dropoff\NINACS\Testing\Outputs" --bias "W:\Astrophotography\BiasLibrary\ZWO ASI533MM Pro\20221025.MasterBias.Gain.0.Offset.25.300x0.01s.xisf" --pedestal 100 --overwrite
```

**Expected:** No zero pixel warnings (PixelIO denormalizes Float32 [0,1] → [0,65535])
