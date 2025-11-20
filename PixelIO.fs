module PixelIO

open System
open XisfLib.Core

/// Extract raw pixel data bytes from XISF image
let getPixelData (img: XisfImage) : byte[] =
    let data = img.PixelData
    if data :? InlineDataBlock then
        (data :?> InlineDataBlock).Data.ToArray()
    else
        failwith "Expected inline data block (embedded/attached data not supported)"

/// Check if image uses normalized [0,1] range based on bounds attribute
let isNormalized (img: XisfImage) : bool =
    not (isNull img.Bounds) &&
    abs (img.Bounds.Lower - 0.0) < 1e-6 &&
    abs (img.Bounds.Upper - 1.0) < 1e-6

/// Read a single pixel value from byte array based on sample format
let private readPixelValue (pixelData: byte[]) (offset: int) (format: XisfSampleFormat) : float =
    match format with
    | XisfSampleFormat.UInt8 ->
        float pixelData.[offset]

    | XisfSampleFormat.UInt16 ->
        let value = uint16 pixelData.[offset] ||| (uint16 pixelData.[offset + 1] <<< 8)
        float value

    | XisfSampleFormat.UInt32 ->
        let value =
            uint32 pixelData.[offset] |||
            (uint32 pixelData.[offset + 1] <<< 8) |||
            (uint32 pixelData.[offset + 2] <<< 16) |||
            (uint32 pixelData.[offset + 3] <<< 24)
        float value

    | XisfSampleFormat.Float32 ->
        float (BitConverter.ToSingle(pixelData, offset))

    | XisfSampleFormat.Float64 ->
        BitConverter.ToDouble(pixelData, offset)

    | _ ->
        failwithf "Unsupported sample format: %A (only UInt8, UInt16, UInt32, Float32, Float64 supported)" format

/// Get bytes per pixel for a sample format
let getBytesPerPixel (format: XisfSampleFormat) : int =
    match format with
    | XisfSampleFormat.UInt8 -> 1
    | XisfSampleFormat.UInt16 -> 2
    | XisfSampleFormat.UInt32 -> 4
    | XisfSampleFormat.UInt64 -> 8
    | XisfSampleFormat.Float32 -> 4
    | XisfSampleFormat.Float64 -> 8
    | XisfSampleFormat.Complex32 -> 8
    | XisfSampleFormat.Complex64 -> 16
    | _ -> failwithf "Unknown sample format: %A" format

/// Read all pixels from image into normalized float array [0, 65535] range
/// Handles UInt8, UInt16, UInt32, Float32, Float64
/// Automatically denormalizes Float32/Float64 [0,1] → [0,65535] based on bounds
let readPixelsAsFloat (img: XisfImage) : float[] =
    let pixelData = getPixelData img
    let format = img.SampleFormat
    let width = int img.Geometry.Width
    let height = int img.Geometry.Height
    let channels = int img.Geometry.ChannelCount
    let pixelCount = width * height * channels
    let bytesPerSample = getBytesPerPixel format
    let normalized = isNormalized img

    let floatData = Array.zeroCreate pixelCount

    for i = 0 to pixelCount - 1 do
        let offset = i * bytesPerSample
        let value = readPixelValue pixelData offset format

        // Denormalize if image is in [0,1] range (PixInsight standard for Float32)
        floatData.[i] <-
            if normalized then
                value * 65535.0  // [0,1] → [0,65535]
            else
                value

    floatData

/// Read pixels for a single channel into float array
let readChannelPixelsAsFloat (img: XisfImage) (channel: int) : float[] =
    let pixelData = getPixelData img
    let format = img.SampleFormat
    let width = int img.Geometry.Width
    let height = int img.Geometry.Height
    let channels = int img.Geometry.ChannelCount
    let pixelCount = width * height
    let bytesPerSample = getBytesPerPixel format
    let bytesPerPixel = channels * bytesPerSample
    let normalized = isNormalized img

    if channel < 0 || channel >= channels then
        failwithf "Channel %d out of range [0, %d)" channel (channels - 1)

    let floatData = Array.zeroCreate pixelCount

    for pix = 0 to pixelCount - 1 do
        let offset = (pix * bytesPerPixel) + (channel * bytesPerSample)
        let value = readPixelValue pixelData offset format

        floatData.[pix] <-
            if normalized then
                value * 65535.0
            else
                value

    floatData

/// Write a single pixel value to byte array based on sample format
let private writePixelValue (value: float) (pixelData: byte[]) (offset: int) (format: XisfSampleFormat) (normalize: bool) : unit =
    match format with
    | XisfSampleFormat.UInt8 ->
        let denorm = if normalize then value / 65535.0 * 255.0 else value
        let clamped = max 0.0 (min 255.0 denorm)
        pixelData.[offset] <- byte (round clamped)

    | XisfSampleFormat.UInt16 ->
        let clamped = max 0.0 (min 65535.0 value)
        let intValue = uint16 (round clamped)
        pixelData.[offset] <- byte (intValue &&& 0xFFus)
        pixelData.[offset + 1] <- byte ((intValue >>> 8) &&& 0xFFus)

    | XisfSampleFormat.UInt32 ->
        let clamped = max 0.0 (min 4294967295.0 value)
        let intValue = uint32 (round clamped)
        pixelData.[offset] <- byte (intValue &&& 0xFFu)
        pixelData.[offset + 1] <- byte ((intValue >>> 8) &&& 0xFFu)
        pixelData.[offset + 2] <- byte ((intValue >>> 16) &&& 0xFFu)
        pixelData.[offset + 3] <- byte ((intValue >>> 24) &&& 0xFFu)

    | XisfSampleFormat.Float32 ->
        let norm = if normalize then value / 65535.0 else value
        let floatValue = single norm
        let bytes = BitConverter.GetBytes(floatValue)
        Array.blit bytes 0 pixelData offset 4

    | XisfSampleFormat.Float64 ->
        let norm = if normalize then value / 65535.0 else value
        let bytes = BitConverter.GetBytes(norm)
        Array.blit bytes 0 pixelData offset 8

    | _ ->
        failwithf "Unsupported sample format for writing: %A" format

/// Write float array [0, 65535] to byte array in specified sample format
/// If normalize=true and format is Float32/Float64, converts to [0,1] range
let writePixelsFromFloat (floatData: float[]) (format: XisfSampleFormat) (normalize: bool) : byte[] =
    let pixelCount = floatData.Length
    let bytesPerSample = getBytesPerPixel format
    let pixelData = Array.zeroCreate (pixelCount * bytesPerSample)

    for i = 0 to pixelCount - 1 do
        let offset = i * bytesPerSample
        writePixelValue floatData.[i] pixelData offset format normalize

    pixelData

/// Get bounds attribute for output format per XISF spec
/// Float32/Float64: REQUIRED per spec, returns Some(XisfImageBounds(0.0, 1.0))
/// Integer formats: SHOULD NOT be specified per spec, returns None (uses implicit [0, 2^n-1])
let getBoundsForFormat (format: XisfSampleFormat) : XisfImageBounds option =
    match format with
    | XisfSampleFormat.Float32 | XisfSampleFormat.Float64 ->
        Some (XisfImageBounds(0.0, 1.0))  // XISF spec REQUIRES bounds for float formats
    | XisfSampleFormat.UInt8 | XisfSampleFormat.UInt16 | XisfSampleFormat.UInt32 ->
        None  // XISF spec: SHOULD NOT specify bounds for default [0, 2^n-1] ranges
    | _ ->
        None  // Unsupported/complex formats

/// Get recommended output sample format based on input format
/// By default, preserves input format
let getRecommendedOutputFormat (inputFormat: XisfSampleFormat) : XisfSampleFormat * bool =
    match inputFormat with
    | XisfSampleFormat.UInt8 -> (XisfSampleFormat.UInt8, false)
    | XisfSampleFormat.UInt16 -> (XisfSampleFormat.UInt16, false)
    | XisfSampleFormat.UInt32 -> (XisfSampleFormat.UInt32, false)
    | XisfSampleFormat.Float32 -> (XisfSampleFormat.Float32, true)   // Normalize to [0,1]
    | XisfSampleFormat.Float64 -> (XisfSampleFormat.Float64, true)   // Normalize to [0,1]
    | _ -> (XisfSampleFormat.UInt16, false)  // Fallback for unsupported formats

/// Parse output format string (for --output-format flag)
let parseOutputFormat (formatStr: string) : XisfSampleFormat option =
    match formatStr.ToLower() with
    | "uint8" | "u8" -> Some XisfSampleFormat.UInt8
    | "uint16" | "u16" -> Some XisfSampleFormat.UInt16
    | "uint32" | "u32" -> Some XisfSampleFormat.UInt32
    | "float32" | "f32" | "float" -> Some XisfSampleFormat.Float32
    | "float64" | "f64" | "double" -> Some XisfSampleFormat.Float64
    | _ -> None

/// Helper to format sample format and range info for logging
let formatSampleFormatInfo (img: XisfImage) : string =
    let normalized = if isNormalized img then " [0,1]" else ""
    sprintf "%A%s" img.SampleFormat normalized
