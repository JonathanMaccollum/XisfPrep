module XisfIO

open System
open System.IO
open XisfLib.Core

// ============================================================================
// TYPES
// ============================================================================

type XisfError =
    | FileNotFound of path: string
    | ReadFailed of path: string * message: string
    | WriteFailed of path: string * message: string
    | InvalidImage of message: string

    override this.ToString() =
        match this with
        | FileNotFound path -> $"File not found: {path}"
        | ReadFailed (path, msg) -> $"Failed to read {path}: {msg}"
        | WriteFailed (path, msg) -> $"Failed to write {path}: {msg}"
        | InvalidImage msg -> $"Invalid image: {msg}"

type ImageMetadata = {
    Width: int
    Height: int
    Channels: int
    Format: XisfSampleFormat
}

// ============================================================================
// LOADING
// ============================================================================

/// Load XISF image from file
let loadImage (filePath: string) : Async<Result<XisfImage, XisfError>> =
    async {
        try
            if not (File.Exists filePath) then
                return Error (FileNotFound filePath)
            else
                let reader = new XisfReader()
                let! unit = reader.ReadAsync(filePath) |> Async.AwaitTask

                if unit.Images.Count = 0 then
                    return Error (InvalidImage "No images found in XISF file")
                else
                    return Ok unit.Images.[0]
        with ex ->
            return Error (ReadFailed (filePath, ex.Message))
    }

/// Load image with extracted metadata and float pixels
let loadImageWithPixels (filePath: string) : Async<Result<XisfImage * ImageMetadata * float[], XisfError>> =
    async {
        match! loadImage filePath with
        | Error err -> return Error err
        | Ok img ->
            try
                let metadata = {
                    Width = int img.Geometry.Width
                    Height = int img.Geometry.Height
                    Channels = int img.Geometry.ChannelCount
                    Format = img.SampleFormat
                }

                let pixels = PixelIO.readPixelsAsFloat img

                return Ok (img, metadata, pixels)
            with ex ->
                return Error (ReadFailed (filePath, $"Failed to read pixels: {ex.Message}"))
    }

// ============================================================================
// WRITING
// ============================================================================

/// Write XISF image to file
let writeImage (outputPath: string) (creatorName: string) (image: XisfImage) : Async<Result<unit, XisfError>> =
    async {
        try
            let metadata = XisfFactory.CreateMinimalMetadata(creatorName)
            let unit = XisfFactory.CreateMonolithic(metadata, image)
            let writer = new XisfWriter()
            do! writer.WriteAsync(unit, outputPath) |> Async.AwaitTask
            return Ok ()
        with ex ->
            return Error (WriteFailed (outputPath, ex.Message))
    }

// ============================================================================
// OUTPUT IMAGE CREATION
// ============================================================================

type OutputImageConfig = {
    Dimensions: (int * int * int) option
    Format: XisfSampleFormat option
    HistoryEntries: string list
    ExcludeFitsKeys: Set<string>
    AdditionalFits: XisfCoreElement[]
    AdditionalProps: XisfProperty[]
}

let defaultOutputImageConfig = {
    Dimensions = None
    Format = None
    HistoryEntries = []
    ExcludeFitsKeys = Set.empty
    AdditionalFits = [||]
    AdditionalProps = [||]
}

/// Create output XisfImage from processed pixels
let createOutputImage
    (originalImg: XisfImage)
    (pixels: byte[])
    (config: OutputImageConfig)
    : Result<XisfImage, XisfError> =

    try
        // Determine dimensions (default to original)
        let (width, height, channels) =
            match config.Dimensions with
            | Some (w, h, c) -> (w, h, c)
            | None ->
                (int originalImg.Geometry.Width,
                 int originalImg.Geometry.Height,
                 int originalImg.Geometry.ChannelCount)

        // Validate dimensions
        if width <= 0 || height <= 0 || channels <= 0 then
            Error (InvalidImage $"Invalid dimensions: {width}x{height}x{channels}")
        else

        // Determine format (default to original)
        let outputFormat =
            match config.Format with
            | Some fmt -> fmt
            | None -> originalImg.SampleFormat

        // Create geometry
        let geometry = XisfImageGeometry([| uint32 width; uint32 height |], uint32 channels)

        // Create data block
        let dataBlock = InlineDataBlock(ReadOnlyMemory(pixels), XisfEncoding.Base64)

        // Get bounds per XISF spec
        let bounds =
            match PixelIO.getBoundsForFormat outputFormat with
            | Some b -> b
            | None -> Unchecked.defaultof<XisfImageBounds>  // null for integer formats

        // Process FITS keywords
        let existingFits =
            if isNull originalImg.AssociatedElements then [||]
            else originalImg.AssociatedElements |> Seq.toArray

        // Filter out excluded keys
        let filteredFits =
            if config.ExcludeFitsKeys.IsEmpty then
                existingFits
            else
                existingFits
                |> Array.filter (fun elem ->
                    match elem with
                    | :? XisfFitsKeyword as kw -> not (config.ExcludeFitsKeys.Contains kw.Name)
                    | _ -> true)

        // Add history entries
        let historyFits =
            config.HistoryEntries
            |> List.map (fun text -> XisfFitsKeyword("HISTORY", "", text) :> XisfCoreElement)
            |> Array.ofList

        // Combine all FITS keywords
        let allFits = Array.concat [filteredFits; historyFits; config.AdditionalFits]

        // Process properties
        let existingProps =
            if isNull originalImg.Properties then [||]
            else originalImg.Properties |> Seq.toArray

        let allProps = Array.append existingProps config.AdditionalProps

        // Create output image
        let outputImage =
            XisfImage(
                geometry,
                outputFormat,
                originalImg.ColorSpace,
                dataBlock,
                bounds,
                originalImg.PixelStorage,
                originalImg.ImageType,
                originalImg.Offset,
                originalImg.Orientation,
                originalImg.ImageId,
                originalImg.Uuid,
                allProps,
                allFits
            )

        Ok outputImage

    with ex ->
        Error (InvalidImage $"Failed to create output image: {ex.Message}")

// ============================================================================
// FITS KEYWORDS MODULE
// ============================================================================

module FitsKeywords =

    /// Filter FITS keywords by name (keep only specified names)
    let filterByName (names: string list) (elements: XisfCoreElement[]) : XisfCoreElement[] =
        let nameSet = Set.ofList names
        elements
        |> Array.filter (fun elem ->
            match elem with
            | :? XisfFitsKeyword as kw -> nameSet.Contains kw.Name
            | _ -> false)

    /// Exclude FITS keywords by name (keep all except specified names)
    let excludeByName (names: string list) (elements: XisfCoreElement[]) : XisfCoreElement[] =
        let excludeSet = Set.ofList names
        elements
        |> Array.filter (fun elem ->
            match elem with
            | :? XisfFitsKeyword as kw -> not (excludeSet.Contains kw.Name)
            | _ -> true)

    /// Create a HISTORY FITS keyword
    let createHistory (text: string) : XisfCoreElement =
        XisfFitsKeyword("HISTORY", "", text) :> XisfCoreElement

    /// Preserve specific FITS keywords from original image
    let preserveKeywords (img: XisfImage) (names: string list) : XisfCoreElement[] =
        if isNull img.AssociatedElements then [||]
        else
            let nameSet = Set.ofList names
            img.AssociatedElements
            |> Seq.toArray
            |> Array.filter (fun elem ->
                match elem with
                | :? XisfFitsKeyword as kw -> nameSet.Contains kw.Name
                | _ -> false)
