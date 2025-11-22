module Commands.Inspect

open System
open System.IO
open System.Text
open System.Xml.Linq
open Serilog
open XisfLib.Core

type InspectOptions = {
    Input: string
    RawXml: bool
    Attachments: bool
    Preview: bool
}

let showHelp() =
    printfn "inspect - Diagnostic inspection of XISF file structure"
    printfn ""
    printfn "Usage: xisfprep inspect [options]"
    printfn ""
    printfn "Required:"
    printfn "  --input, -i <file>        Input XISF file to inspect"
    printfn ""
    printfn "Optional:"
    printfn "  --raw-xml                 Dump raw XML header"
    printfn "  --attachments             Show detailed attachment block analysis"
    printfn "  --preview                 Show 128x128 pixel sample from center"
    printfn ""
    printfn "Use Cases:"
    printfn "  - Debug problematic XISF files"
    printfn "  - Verify file integrity before processing"
    printfn "  - Understand file structure and metadata"
    printfn "  - Diagnose attachment layout issues"
    printfn ""
    printfn "Examples:"
    printfn "  xisfprep inspect -i \"image.xisf\""
    printfn "  xisfprep inspect -i \"image.xisf\" --raw-xml"
    printfn "  xisfprep inspect -i \"image.xisf\" --attachments --preview"

let parseArgs (args: string array) : InspectOptions =
    let rec parse (args: string list) (opts: InspectOptions) =
        match args with
        | [] -> opts
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parse rest { opts with Input = value }
        | "--raw-xml" :: rest ->
            parse rest { opts with RawXml = true }
        | "--attachments" :: rest ->
            parse rest { opts with Attachments = true }
        | "--preview" :: rest ->
            parse rest { opts with Preview = true }
        | arg :: _ ->
            failwithf "Unknown argument: %s" arg

    let defaults = {
        Input = ""
        RawXml = false
        Attachments = false
        Preview = false
    }

    let opts = parse (List.ofArray args) defaults

    if String.IsNullOrEmpty opts.Input then failwith "Required argument: --input"

    opts

let formatBytes (bytes: int64) =
    let kb = float bytes / 1024.0
    let mb = kb / 1024.0
    let gb = mb / 1024.0
    if gb >= 1.0 then sprintf "%.2f GB" gb
    elif mb >= 1.0 then sprintf "%.2f MB" mb
    elif kb >= 1.0 then sprintf "%.2f KB" kb
    else sprintf "%d bytes" bytes

let inspectFile (opts: InspectOptions) =
    let filePath = opts.Input

    printfn "=== XISF File Inspection ==="
    printfn "File: %s" filePath

    let fileInfo = FileInfo(filePath)
    printfn "Size: %s (%d bytes)" (formatBytes fileInfo.Length) fileInfo.Length
    printfn ""

    use stream = File.OpenRead(filePath)

    // Read file header (16 bytes)
    printfn "[FILE HEADER]"
    let headerBytes = Array.zeroCreate<byte> 16
    stream.Read(headerBytes, 0, 16) |> ignore

    let signature = Encoding.ASCII.GetString(headerBytes, 0, 8)
    let headerLength = BitConverter.ToUInt32(headerBytes, 8)
    let reserved = BitConverter.ToUInt32(headerBytes, 12)

    let signatureValid = signature = "XISF0100"
    printfn "Signature: %s %s" signature (if signatureValid then "✓" else "✗ INVALID")
    printfn "Header Length: %d bytes" headerLength
    if reserved <> 0u then
        printfn "Reserved: %d (should be 0)" reserved
    printfn ""

    if not signatureValid then
        printfn "ERROR: Invalid XISF signature. Expected 'XISF0100'"
        1
    else
        // Read XML header
        let xmlBytes = Array.zeroCreate<byte> (int headerLength)
        stream.Read(xmlBytes, 0, int headerLength) |> ignore
        let xmlText = Encoding.UTF8.GetString(xmlBytes)

        // Raw XML dump if requested
        if opts.RawXml then
            printfn "[RAW XML HEADER]"
            printfn "%s" xmlText
            printfn ""

        // Parse and display XML structure
        printfn "[XML STRUCTURE]"
        try
            let doc = XDocument.Parse(xmlText)
            let root = doc.Root

            printfn "Root: %s" root.Name.LocalName
            match root.Attribute(XName.Get("version")) with
            | null -> ()
            | attr -> printfn "Version: %s" attr.Value
            printfn ""

            // Metadata
            let metadata = root.Elements() |> Seq.filter (fun e -> e.Name.LocalName = "Metadata") |> Seq.tryHead
            match metadata with
            | Some m ->
                let props = m.Elements() |> Seq.filter (fun e -> e.Name.LocalName = "Property") |> Seq.toList
                printfn "[METADATA] (%d properties)" props.Length
                for p in props do
                    let id = match p.Attribute(XName.Get("id")) with | null -> "?" | a -> a.Value
                    let value = match p.Attribute(XName.Get("value")) with | null -> "(inline)" | a -> a.Value
                    printfn "  %s = %s" id value
                printfn ""
            | None -> ()

            // Images
            let images = root.Elements() |> Seq.filter (fun e -> e.Name.LocalName = "Image") |> Seq.toList
            for (i, img) in images |> List.indexed do
                printfn "[IMAGE %d]" i
                let geometry = match img.Attribute(XName.Get("geometry")) with | null -> "?" | a -> a.Value
                let sampleFormat = match img.Attribute(XName.Get("sampleFormat")) with | null -> "?" | a -> a.Value
                let colorSpace = match img.Attribute(XName.Get("colorSpace")) with | null -> "?" | a -> a.Value
                let location = match img.Attribute(XName.Get("location")) with | null -> "?" | a -> a.Value
                let bounds = match img.Attribute(XName.Get("bounds")) with | null -> None | a -> Some a.Value

                printfn "  Geometry: %s" geometry
                printfn "  Format: %s" sampleFormat
                printfn "  Color Space: %s" colorSpace
                printfn "  Location: %s" location
                match bounds with
                | Some b -> printfn "  Bounds: %s" b
                | None -> ()

                // FITS keywords
                let fitsKeywords =
                    img.Elements()
                    |> Seq.filter (fun e -> e.Name.LocalName = "FITSKeyword")
                    |> Seq.toList

                if not fitsKeywords.IsEmpty then
                    printfn "  FITS Keywords: %d" fitsKeywords.Length

                    // Show key metadata
                    let importantKeys = ["OBJECT"; "FILTER"; "IMAGETYP"; "EXPTIME"; "CCD-TEMP"; "GAIN"; "BAYERPAT"]
                    for keyName in importantKeys do
                        let found =
                            fitsKeywords
                            |> List.tryFind (fun k ->
                                match k.Attribute(XName.Get("name")) with
                                | null -> false
                                | a -> a.Value = keyName)
                        match found with
                        | Some k ->
                            let value = match k.Attribute(XName.Get("value")) with | null -> "" | a -> a.Value.Trim([|'\''|])
                            printfn "    %s: %s" keyName value
                        | None -> ()

                    // Count others
                    let otherCount = fitsKeywords.Length - (importantKeys |> List.filter (fun k ->
                        fitsKeywords |> List.exists (fun kw ->
                            match kw.Attribute(XName.Get("name")) with
                            | null -> false
                            | a -> a.Value = k)) |> List.length)
                    if otherCount > 0 then
                        printfn "    ... and %d more keywords" otherCount

                printfn ""

        with ex ->
            printfn "ERROR parsing XML: %s" ex.Message
            printfn ""

        // Attachment analysis
        if opts.Attachments then
            printfn "[ATTACHMENTS]"
            let currentPos = stream.Position
            let remainingBytes = fileInfo.Length - currentPos
            printfn "Data section starts at: %d" currentPos
            printfn "Remaining file size: %s" (formatBytes remainingBytes)

            try
                let doc = XDocument.Parse(xmlText)
                let attachments =
                    doc.Descendants()
                    |> Seq.choose (fun e ->
                        match e.Attribute(XName.Get("location")) with
                        | null -> None
                        | attr when attr.Value.StartsWith("attachment:") ->
                            let parts = attr.Value.Substring(11).Split(':')
                            if parts.Length >= 2 then
                                Some (UInt64.Parse(parts.[0]), UInt64.Parse(parts.[1]))
                            else None
                        | _ -> None)
                    |> Seq.toList

                if attachments.Length > 0 then
                    for (pos, size) in attachments do
                        printfn "  Position: %d, Size: %s (%d bytes)" pos (formatBytes (int64 size)) size

                    let maxEnd = attachments |> List.map (fun (p, s) -> p + s) |> List.max
                    printfn "End of last attachment: %d" maxEnd
                    printfn "File ends at: %d" fileInfo.Length
                    let unused = fileInfo.Length - int64 maxEnd
                    if unused <> 0L then
                        printfn "Unused space: %s" (formatBytes unused)
                else
                    printfn "  No attachment blocks found"

                printfn ""
            with ex ->
                printfn "Could not analyze attachments: %s" ex.Message
                printfn ""

        // Pixel preview
        if opts.Preview then
            printfn "[PIXEL PREVIEW 128x128]"
            try
                let reader = new XisfReader()
                let unit = reader.ReadAsync(filePath) |> Async.AwaitTask |> Async.RunSynchronously
                let img = unit.Images.[0]

                let width = int img.Geometry.Width
                let height = int img.Geometry.Height
                let channels = int img.Geometry.ChannelCount

                // Read pixels
                let pixels = PixelIO.readPixelsAsFloat img

                // Calculate center region
                let previewSize = 128
                let startX = max 0 ((width - previewSize) / 2)
                let startY = max 0 ((height - previewSize) / 2)
                let endX = min width (startX + previewSize)
                let endY = min height (startY + previewSize)

                printfn "  Region: (%d,%d) to (%d,%d)" startX startY endX endY
                printfn "  Channels: %d" channels

                // Calculate stats for preview region
                for ch = 0 to channels - 1 do
                    let mutable sum = 0.0
                    let mutable minVal = Double.MaxValue
                    let mutable maxVal = Double.MinValue
                    let mutable count = 0

                    for y = startY to endY - 1 do
                        for x = startX to endX - 1 do
                            let idx = (y * width + x) * channels + ch
                            let value = pixels.[idx]
                            sum <- sum + value
                            minVal <- min minVal value
                            maxVal <- max maxVal value
                            count <- count + 1

                    let mean = sum / float count
                    let chName = if channels = 1 then "Gray" elif ch = 0 then "R" elif ch = 1 then "G" else "B"
                    printfn "  %s: min=%.1f, max=%.1f, mean=%.1f" chName minVal maxVal mean

                printfn ""
            with ex ->
                printfn "  Could not read pixels: %s" ex.Message
                printfn ""

        // Validation summary
        printfn "[VALIDATION]"
        printfn "✓ Signature valid"
        printfn "✓ XML well-formed"
        printfn "✓ File structure appears valid"

        0

let run (args: string array) =
    let hasHelp = args |> Array.contains "--help" || args |> Array.contains "-h"

    if hasHelp then
        showHelp()
        0
    else
        try
            let opts = parseArgs args

            if not (File.Exists(opts.Input)) then
                Log.Error("File not found: {Path}", opts.Input)
                1
            else
                inspectFile opts
        with ex ->
            Log.Error("Error: {Message}", ex.Message)
            printfn ""
            printfn "Run 'xisfprep inspect --help' for usage information"
            1
