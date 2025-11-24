# XisfPrep

Command-line tool for batch preprocessing astrophotography images in XISF format.

## Features

- **Calibrate** - Apply bias, dark, and flat calibration frames
- **Debayer** - Convert Bayer mosaic to RGB (VNG/bilinear)
- **Align** - Register images using star detection and triangle matching
- **Integrate** - Stack images with rejection and normalization
- **Stats** - Calculate image statistics with grouping and filtering
- **Stars** - Detect stars and generate visualizations
- **Bin** - Downsample images by pixel binning
- **Convert** - Format conversion (XISF, FITS, TIFF)
- **Inspect** - Diagnostic inspection of XISF structure

## Requirements

- .NET 8.0 or later

## Build

```bash
dotnet build
```

## Usage

```bash
# Calibrate light frames
xisfprep calibrate -i "lights/*.xisf" -o "calibrated/" \
  --bias master_bias.xisf --dark master_dark.xisf --flat master_flat.xisf

# Stack images
xisfprep integrate -i "calibrated/*.xisf" -o "master.xisf" \
  --rejection linearfit --normalization multiplicative

# Get image statistics
xisfprep stats -i "images/*.xisf" --metrics all --group-by filter
```

See `Docs/Info.md` for complete documentation.

## License

MIT License - see [LICENSE](LICENSE) for details.
