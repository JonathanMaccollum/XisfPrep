module Algorithms.Painting

open System
open Serilog
open XisfLib.Core

/// Create black pixel data for given dimensions and format
let createBlackPixels (width: int) (height: int) (channels: int) (format: XisfSampleFormat) : byte[] =
    let sampleCount = width * height * channels
    let bytesPerSample = PixelIO.getBytesPerPixel format
    Array.zeroCreate (sampleCount * bytesPerSample)

/// Get max value for output format
let getMaxValue (format: XisfSampleFormat) =
    match format with
    | XisfSampleFormat.UInt8 -> 255.0
    | XisfSampleFormat.UInt16 -> 65535.0
    | XisfSampleFormat.UInt32 -> float UInt32.MaxValue
    | XisfSampleFormat.Float32 | XisfSampleFormat.Float64 -> 1.0
    | _ -> 65535.0

/// Set pixel value in byte array for given format
let setPixel (pixels: byte[]) (index: int) (value: float) (format: XisfSampleFormat) =
    match format with
    | XisfSampleFormat.UInt8 ->
        pixels.[index] <- byte (min 255.0 (max 0.0 value))
    | XisfSampleFormat.UInt16 ->
        let v = uint16 (min 65535.0 (max 0.0 value))
        let bytes = BitConverter.GetBytes(v)
        pixels.[index * 2] <- bytes.[0]
        pixels.[index * 2 + 1] <- bytes.[1]
    | XisfSampleFormat.UInt32 ->
        let v = uint32 (min (float UInt32.MaxValue) (max 0.0 value))
        let bytes = BitConverter.GetBytes(v)
        for i in 0..3 do pixels.[index * 4 + i] <- bytes.[i]
    | XisfSampleFormat.Float32 ->
        let bytes = BitConverter.GetBytes(float32 value)
        for i in 0..3 do pixels.[index * 4 + i] <- bytes.[i]
    | XisfSampleFormat.Float64 ->
        let bytes = BitConverter.GetBytes(value)
        for i in 0..7 do pixels.[index * 8 + i] <- bytes.[i]
    | fmt ->
        Log.Warning($"setPixel: Unsupported sample format {fmt}")

/// Paint a circle marker
let paintCircle (pixels: byte[]) (width: int) (height: int) (channels: int) (channel: int) (cx: float) (cy: float) (radius: float) (intensity: float) (format: XisfSampleFormat) =
    let maxVal = getMaxValue format
    let value = intensity * maxVal
    let r2 = radius * radius
    let thickness = max 1.0 (radius * 0.15)
    let inner2 = (radius - thickness) * (radius - thickness)

    let minX = max 0 (int (cx - radius - 1.0))
    let maxX = min (width - 1) (int (cx + radius + 1.0))
    let minY = max 0 (int (cy - radius - 1.0))
    let maxY = min (height - 1) (int (cy + radius + 1.0))

    for y in minY .. maxY do
        for x in minX .. maxX do
            let dx = float x - cx
            let dy = float y - cy
            let d2 = dx * dx + dy * dy
            if d2 <= r2 && d2 >= inner2 then
                let idx = (y * width + x) * channels + channel
                setPixel pixels idx value format

/// Paint a crosshair (X) marker
let paintCrosshair (pixels: byte[]) (width: int) (height: int) (channels: int) (channel: int) (cx: float) (cy: float) (size: float) (intensity: float) (format: XisfSampleFormat) =
    let maxVal = getMaxValue format
    let value = intensity * maxVal
    let halfSize = int (size / 2.0)
    let icx = int cx
    let icy = int cy

    // Horizontal line
    for x in max 0 (icx - halfSize) .. min (width - 1) (icx + halfSize) do
        let idx = (icy * width + x) * channels + channel
        if icy >= 0 && icy < height then
            setPixel pixels idx value format

    // Vertical line
    for y in max 0 (icy - halfSize) .. min (height - 1) (icy + halfSize) do
        let idx = (y * width + icx) * channels + channel
        if icx >= 0 && icx < width then
            setPixel pixels idx value format

/// Paint an X marker (diagonal cross)
let paintX (pixels: byte[]) (width: int) (height: int) (channels: int) (channel: int) (cx: float) (cy: float) (size: float) (intensity: float) (format: XisfSampleFormat) =
    let maxVal = getMaxValue format
    let value = intensity * maxVal
    let halfSize = int (size / 2.0)
    let icx = int cx
    let icy = int cy

    // Diagonal lines
    for d in -halfSize .. halfSize do
        // Top-left to bottom-right
        let x1 = icx + d
        let y1 = icy + d
        if x1 >= 0 && x1 < width && y1 >= 0 && y1 < height then
            let idx = (y1 * width + x1) * channels + channel
            setPixel pixels idx value format

        // Top-right to bottom-left
        let x2 = icx + d
        let y2 = icy - d
        if x2 >= 0 && x2 < width && y2 >= 0 && y2 < height then
            let idx = (y2 * width + x2) * channels + channel
            setPixel pixels idx value format

/// Paint a Gaussian profile
let paintGaussian (pixels: byte[]) (width: int) (height: int) (channels: int) (channel: int) (cx: float) (cy: float) (fwhm: float) (intensity: float) (format: XisfSampleFormat) =
    let maxVal = getMaxValue format
    let sigma = fwhm / 2.355  // FWHM = 2.355 * sigma
    let sigma2 = sigma * sigma
    let extent = int (3.0 * sigma) + 1

    let minX = max 0 (int cx - extent)
    let maxX = min (width - 1) (int cx + extent)
    let minY = max 0 (int cy - extent)
    let maxY = min (height - 1) (int cy + extent)

    for y in minY .. maxY do
        for x in minX .. maxX do
            let dx = float x - cx
            let dy = float y - cy
            let d2 = dx * dx + dy * dy
            let gaussVal = exp (-d2 / (2.0 * sigma2))
            let value = intensity * maxVal * gaussVal
            let idx = (y * width + x) * channels + channel
            setPixel pixels idx value format
