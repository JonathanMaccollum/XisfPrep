module Algorithms.OutputImage

open System
open XisfLib.Core

/// Create output image preserving source geometry and headers, with additional headers
/// Replaces specified headers in-place to maintain header ordering
let createOutputImage (source: XisfImage) (pixelData: byte[]) (outputFormat: XisfSampleFormat) (additionalHeaders: XisfCoreElement[]) (inPlaceKeys: Set<string>) : XisfImage =
    let dataBlock = InlineDataBlock(ReadOnlyMemory(pixelData), XisfEncoding.Base64)

    let bounds =
        match PixelIO.getBoundsForFormat outputFormat with
        | Some b -> b
        | None -> Unchecked.defaultof<XisfImageBounds>

    // Build map of headers to replace in-place
    let replacements =
        additionalHeaders
        |> Array.choose (fun elem ->
            match elem with
            | :? XisfFitsKeyword as kw when inPlaceKeys.Contains kw.Name -> Some (kw.Name, elem)
            | _ -> None)
        |> Map.ofArray

    // Process existing headers, replacing in-place where needed
    let combinedHeaders =
        if isNull source.AssociatedElements then
            additionalHeaders
        else
            let existing = source.AssociatedElements :> seq<_> |> Seq.toArray

            // Replace headers in-place
            let updated =
                existing
                |> Array.map (fun elem ->
                    match elem with
                    | :? XisfFitsKeyword as kw when Map.containsKey kw.Name replacements ->
                        replacements.[kw.Name]
                    | _ -> elem)

            // Append headers that weren't replaced in-place
            let toAppend =
                additionalHeaders
                |> Array.filter (fun elem ->
                    match elem with
                    | :? XisfFitsKeyword as kw -> not (inPlaceKeys.Contains kw.Name)
                    | _ -> true)

            Array.append updated toAppend

    XisfImage(
        source.Geometry,
        outputFormat,
        source.ColorSpace,
        dataBlock,
        bounds,
        source.PixelStorage,
        source.ImageType,
        source.Offset,
        source.Orientation,
        source.ImageId,
        source.Uuid,
        source.Properties,
        combinedHeaders
    )

/// Write image to disk with minimal metadata
let writeOutputFile (outputPath: string) (image: XisfImage) (creatorName: string) : Async<unit> =
    async {
        let metadata = XisfFactory.CreateMinimalMetadata(creatorName)
        let outUnit = XisfFactory.CreateMonolithic(metadata, image)
        let writer = new XisfWriter()
        do! writer.WriteAsync(outUnit, outputPath) |> Async.AwaitTask
    }
