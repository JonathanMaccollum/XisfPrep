module Commands.SharedInfra

open System
open System.IO
open Serilog
open XisfLib.Core

// ============================================================================
// BATCH PROCESSING
// ============================================================================

module BatchProcessing =

    type BatchConfig<'Config> = {
        Files: string[]
        OutputDir: string
        Suffix: string
        Overwrite: bool
        MaxParallel: int
        Config: 'Config
    }

    /// Build output path(s) for a file
    /// Returns None to skip, Some [paths] to process
    type BuildOutputPathsFunc = string -> string -> string -> string list option

    /// Process a single file with output path(s)
    type ProcessFileFunc<'Config, 'Error> =
        string -> string list -> 'Config -> Async<Result<unit, 'Error>>

    /// Process batch of files in parallel
    /// Returns exit code: 0 = success, 1 = some failures
    let processBatch
        (config: BatchConfig<'Config>)
        (buildOutputPaths: BuildOutputPathsFunc)
        (processFile: ProcessFileFunc<'Config, 'Error>)
        : Async<int> =
        async {
            // Create output directory if needed
            if not (Directory.Exists config.OutputDir) then
                Directory.CreateDirectory config.OutputDir |> ignore
                Log.Information("Created output directory: {Dir}", config.OutputDir)

            // Build tasks for each file
            let tasks =
                config.Files
                |> Array.map (fun filePath ->
                    async {
                        let fileName = Path.GetFileName filePath

                        // Build output paths
                        let baseName = Path.GetFileNameWithoutExtension filePath
                        let outputPathsOpt = buildOutputPaths baseName config.Suffix config.OutputDir

                        match outputPathsOpt with
                        | None ->
                            // Skip this file
                            return Ok ()

                        | Some [] ->
                            // No output paths
                            return Ok ()

                        | Some outputPaths ->
                            // Check if any output files exist
                            let existingPaths = outputPaths |> List.filter File.Exists

                            if not config.Overwrite && not (List.isEmpty existingPaths) then
                                // Skip if files exist and not overwriting
                                for path in existingPaths do
                                    Log.Warning("Output file exists, skipping (use --overwrite to replace): {Path}",
                                                Path.GetFileName path)
                                return Ok ()
                            else
                                // Process the file
                                let! result = processFile filePath outputPaths config.Config

                                match result with
                                | Ok () -> return Ok ()
                                | Error err ->
                                    Log.Error("Failed to process {File}: {Error}", fileName, err.ToString())
                                    return Error err
                    })

            // Execute in parallel
            let! results = Async.Parallel(tasks, maxDegreeOfParallelism = config.MaxParallel)

            // Count successes and failures
            let successCount = results |> Array.filter (function Ok _ -> true | Error _ -> false) |> Array.length
            let failCount = results.Length - successCount

            printfn ""
            let plural = if config.Files.Length = 1 then "" else "s"
            printfn "Completed: %d succeeded, %d failed out of %d file%s" successCount failCount config.Files.Length plural

            return if failCount > 0 then 1 else 0
        }

// ============================================================================
// ARGUMENT PARSING
// ============================================================================

module ArgumentParsing =

    type CommonOptions = {
        Input: string
        Output: string
        Suffix: string
        Overwrite: bool
        MaxParallel: int
        OutputFormat: XisfSampleFormat option
    }

    let defaultCommonOptions = {
        Input = ""
        Output = ""
        Suffix = ""
        Overwrite = false
        MaxParallel = Environment.ProcessorCount
        OutputFormat = None
    }

    /// Parse common flags and return remaining args + parsed options
    let rec parseCommonFlags (args: string list) (opts: CommonOptions) : (string list * CommonOptions) =
        match args with
        | [] -> ([], opts)
        | "--input" :: value :: rest | "-i" :: value :: rest ->
            parseCommonFlags rest { opts with Input = value }
        | "--output" :: value :: rest | "-o" :: value :: rest ->
            parseCommonFlags rest { opts with Output = value }
        | "--suffix" :: value :: rest ->
            parseCommonFlags rest { opts with Suffix = value }
        | "--overwrite" :: rest ->
            parseCommonFlags rest { opts with Overwrite = true }
        | "--parallel" :: value :: rest ->
            parseCommonFlags rest { opts with MaxParallel = int value }
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> parseCommonFlags rest { opts with OutputFormat = Some fmt }
            | None -> failwithf "Unknown output format: %s (supported: uint8, uint16, uint32, float32, float64)" value
        | remaining ->
            // Not a common flag, return remaining args
            (remaining, opts)

    /// Parse individual flag helpers (compositional approach)

    let parseInputFlag (args: string list) (current: string option) : (string list * string option) =
        match args with
        | "--input" :: value :: rest | "-i" :: value :: rest -> (rest, Some value)
        | _ -> (args, current)

    let parseOutputFlag (args: string list) (current: string option) : (string list * string option) =
        match args with
        | "--output" :: value :: rest | "-o" :: value :: rest -> (rest, Some value)
        | _ -> (args, current)

    let parseSuffixFlag (args: string list) (current: string) : (string list * string) =
        match args with
        | "--suffix" :: value :: rest -> (rest, value)
        | _ -> (args, current)

    let parseOverwriteFlag (args: string list) (current: bool) : (string list * bool) =
        match args with
        | "--overwrite" :: rest -> (rest, true)
        | _ -> (args, current)

    let parseParallelFlag (args: string list) (current: int) : (string list * int) =
        match args with
        | "--parallel" :: value :: rest -> (rest, int value)
        | _ -> (args, current)

    let parseOutputFormatFlag (args: string list) (current: XisfSampleFormat option) : (string list * XisfSampleFormat option) =
        match args with
        | "--output-format" :: value :: rest ->
            match PixelIO.parseOutputFormat value with
            | Some fmt -> (rest, Some fmt)
            | None -> failwithf "Unknown output format: %s" value
        | _ -> (args, current)
