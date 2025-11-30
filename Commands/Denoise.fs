module Commands.Denoise

open System
open System.IO
open Serilog
open FsToolkit.ErrorHandling
open Algorithms.Self2SelfDenoise

let showHelp() =
    printfn "denoise - Self2Self neural network denoising (experimental)"
    printfn ""
    printfn "Usage:"
    printfn "  xisfprep denoise train [options]   Train model and save"
    printfn "  xisfprep denoise apply [options]   Apply saved model to image(s)"
    printfn ""
    printfn "TRAIN MODE - Train a denoising model on a noisy image"
    printfn "============"
    printfn "Required:"
    printfn "  --input, -i <file>        Input XISF file to train on (mono only)"
    printfn "  --model <path>            Path to save trained model (.pt file)"
    printfn ""
    printfn "Training Parameters:"
    printfn "  --epochs <n>              Number of training epochs (default: 100)"
    printfn "  --layers <n>              Number of CNN layers (default: 5)"
    printfn "  --filters <n>             Number of filters per layer (default: 48)"
    printfn "  --learning-rate <lr>      Learning rate (default: 0.001)"
    printfn "  --dropout <rate>          Dropout rate for masking (default: 0.5)"
    printfn "  --cpu                     Force CPU training (default: use GPU if available)"
    printfn ""
    printfn "Model Management:"
    printfn "  --overwrite               Overwrite existing model (start fresh training)"
    printfn "  --continue                Continue training existing model (incremental)"
    printfn ""
    printfn "Examples:"
    printfn "  # Create new model"
    printfn "  xisfprep denoise train -i noisy.xisf --model my_denoiser.pt --epochs 100"
    printfn ""
    printfn "  # Continue training existing model on new image"
    printfn "  xisfprep denoise train -i another.xisf --model my_denoiser.pt --continue --epochs 50"
    printfn ""
    printfn "APPLY MODE - Apply trained model to denoise images"
    printfn "==========="
    printfn "Required:"
    printfn "  --input, -i <pattern>     Input XISF file(s) (wildcards supported)"
    printfn "  --output, -o <directory>  Output directory for denoised files"
    printfn "  --model <path>            Path to trained model (.pt file)"
    printfn ""
    printfn "Model Parameters (must match training):"
    printfn "  --layers <n>              Number of CNN layers (default: 5)"
    printfn "  --filters <n>             Number of filters per layer (default: 48)"
    printfn ""
    printfn "Output Options:"
    printfn "  --suffix <text>           Output filename suffix (default: _denoised)"
    printfn "  --overwrite               Overwrite existing output files"
    printfn "  --cpu                     Force CPU inference (default: use GPU if available)"
    printfn ""
    printfn "Example:"
    printfn "  xisfprep denoise apply -i \"lights/*.xisf\" -o denoised/ --model my_denoiser.pt"
    printfn ""
    printfn "Algorithm:"
    printfn "  Self2Self trains a CNN on a single noisy image by randomly masking pixels"
    printfn "  and learning to predict dropped pixels from visible neighbors. Since noise"
    printfn "  is pixel-independent but signal is spatially correlated, the network learns"
    printfn "  to denoise without requiring clean reference images."
    printfn ""

type DenoiseError =
    | LoadFailed of XisfIO.XisfError
    | NotMonoImage of int
    | SaveFailed of XisfIO.XisfError
    | ModelError of string

    override this.ToString() =
        match this with
        | LoadFailed err -> err.ToString()
        | NotMonoImage channels -> $"Self2Self currently only supports mono images (got {channels} channels)"
        | SaveFailed err -> err.ToString()
        | ModelError msg -> $"Model error: {msg}"

// ============================================================================
// TRAIN MODE
// ============================================================================

let trainMode (inputPath: string) (modelPath: string) (overwrite: bool) (continueTraining: bool) (config: TrainingConfig) : Async<Result<unit, DenoiseError>> =
    asyncResult {
        let fileName = Path.GetFileName inputPath

        // Check model file existence and flags
        let modelExists = File.Exists modelPath

        if modelExists && not overwrite && not continueTraining then
            Log.Error("Model file already exists: {Path}", modelPath)
            Log.Error("Use --overwrite to replace it or --continue to continue training")
            return! Error (ModelError "Model file already exists. Use --overwrite or --continue")

        if not modelExists && continueTraining then
            Log.Error("Cannot continue training - model file not found: {Path}", modelPath)
            return! Error (ModelError "Model file not found. Remove --continue to create new model")

        if overwrite && continueTraining then
            Log.Error("Cannot use both --overwrite and --continue flags")
            return! Error (ModelError "Cannot use both --overwrite and --continue")

        Log.Information("Training mode: {File}", fileName)

        // Load XISF
        let! (_, metadata, pixels) =
            XisfIO.loadImageWithPixels inputPath
            |> AsyncResult.mapError LoadFailed

        Log.Information("  Loaded: {Width}×{Height}, {Channels} channel(s), {Format}",
                       metadata.Width, metadata.Height, metadata.Channels, metadata.Format)

        // Check mono
        if metadata.Channels <> 1 then
            return! Error (NotMonoImage metadata.Channels)

        // Train model
        let model =
            if continueTraining then
                Log.Information("  Loading existing model to continue training...")
                let existingModel = Algorithms.Self2SelfDenoise.loadModel modelPath config.NumLayers config.NumFilters
                Log.Information("  Continuing Self2Self neural network training...")
                Algorithms.Self2SelfDenoise.continueTraining existingModel pixels metadata.Width metadata.Height config modelPath
            else
                if modelExists then
                    Log.Information("  Overwriting existing model with new training...")
                else
                    Log.Information("  Training new Self2Self neural network...")
                Algorithms.Self2SelfDenoise.trainModel pixels metadata.Width metadata.Height config modelPath

        // Save model
        try
            Algorithms.Self2SelfDenoise.saveModel model modelPath
            Log.Information("Training complete! Model saved to: {Path}", modelPath)
        with ex ->
            return! Error (ModelError $"Failed to save model: {ex.Message}")
    }

// ============================================================================
// APPLY MODE
// ============================================================================

let applyToFile
    (inputPath: string)
    (outputDir: string)
    (suffix: string)
    (overwrite: bool)
    (model: SimpleDenoiser)
    (useGpu: bool)
    : Async<Result<unit, DenoiseError>> =
    asyncResult {
        let fileName = Path.GetFileName inputPath
        let baseName = Path.GetFileNameWithoutExtension inputPath
        let outputFileName = baseName + suffix + ".xisf"
        let outputPath = Path.Combine(outputDir, outputFileName)

        // Skip if exists and not overwriting
        if File.Exists outputPath && not overwrite then
            Log.Information("Skipping (exists): {File}", fileName)
            return ()
        else

        Log.Information("Processing: {File}", fileName)

        // Load XISF
        let! (originalImg, metadata, pixels) =
            XisfIO.loadImageWithPixels inputPath
            |> AsyncResult.mapError LoadFailed

        Log.Verbose("  Loaded: {Width}×{Height}, {Channels} channel(s), {Format}",
                   metadata.Width, metadata.Height, metadata.Channels, metadata.Format)

        // Check mono
        if metadata.Channels <> 1 then
            return! Error (NotMonoImage metadata.Channels)

        // Calculate input stats
        let inputMin = Array.min pixels
        let inputMax = Array.max pixels
        let inputMean = Array.average pixels
        let sortedInput = Array.sort pixels
        let inputMedian = sortedInput.[sortedInput.Length / 2]
        let inputVariance = pixels |> Array.map (fun x -> (x - inputMean) ** 2.0) |> Array.average
        let inputStdDev = sqrt inputVariance

        Log.Information("  Input  Stats: Min={Min:F2}, Max={Max:F2}, Mean={Mean:F2}, Median={Median:F2}, StdDev={StdDev:F2}",
                       inputMin, inputMax, inputMean, inputMedian, inputStdDev)

        // Apply model
        Log.Information("  Applying denoising model...")
        let denoisedPixels = Algorithms.Self2SelfDenoise.applyModel model pixels metadata.Width metadata.Height useGpu

        // Calculate output stats
        let outputMin = Array.min denoisedPixels
        let outputMax = Array.max denoisedPixels
        let outputMean = Array.average denoisedPixels
        let sortedOutput = Array.sort denoisedPixels
        let outputMedian = sortedOutput.[sortedOutput.Length / 2]
        let outputVariance = denoisedPixels |> Array.map (fun x -> (x - outputMean) ** 2.0) |> Array.average
        let outputStdDev = sqrt outputVariance

        Log.Information("  Output Stats: Min={Min:F2}, Max={Max:F2}, Mean={Mean:F2}, Median={Median:F2}, StdDev={StdDev:F2}",
                       outputMin, outputMax, outputMean, outputMedian, outputStdDev)

        // Show changes
        let meanChange = ((outputMean - inputMean) / inputMean) * 100.0
        let medianChange = ((outputMedian - inputMedian) / inputMedian) * 100.0
        let stdDevChange = ((outputStdDev - inputStdDev) / inputStdDev) * 100.0
        Log.Information("  Changes: Mean={Mean:+0.0;-0.0}%, Median={Median:+0.0;-0.0}%, StdDev={StdDev:+0.0;-0.0}%",
                       meanChange, medianChange, stdDevChange)

        // Prepare output
        let (outputFormat, normalize) =
            PixelIO.getRecommendedOutputFormat metadata.Format

        let denoisedBytes = PixelIO.writePixelsFromFloat denoisedPixels outputFormat normalize

        // Create output image
        let historyEntry = "Self2Self denoising"

        let! outputImg =
            XisfIO.createOutputImage originalImg denoisedBytes {
                XisfIO.defaultOutputImageConfig with
                    Format = Some outputFormat
                    HistoryEntries = [historyEntry]
            }
            |> Result.mapError LoadFailed
            |> AsyncResult.ofResult

        // Save
        do! XisfIO.writeImage outputPath "XisfPrep Self2Self Denoise v0.1" outputImg
            |> AsyncResult.mapError SaveFailed

        Log.Information("  -> {Output}", outputFileName)
    }

let applyMode
    (inputPattern: string)
    (outputDir: string)
    (modelPath: string)
    (numLayers: int)
    (numFilters: int)
    (suffix: string)
    (overwrite: bool)
    (useGpu: bool)
    : Async<Result<unit, DenoiseError>> =
    async {
        // Load model
        let model =
            try
                Algorithms.Self2SelfDenoise.loadModel modelPath numLayers numFilters |> Ok
            with ex ->
                Error (ModelError $"Failed to load model: {ex.Message}")

        match model with
        | Error err -> return Error err
        | Ok loadedModel ->

            // Create output directory
            if not (Directory.Exists outputDir) then
                Directory.CreateDirectory outputDir |> ignore
                Log.Information("Created output directory: {Dir}", outputDir)

            // Get input files
            let inputDir = Path.GetDirectoryName inputPattern
            let pattern = Path.GetFileName inputPattern
            let files =
                if String.IsNullOrEmpty inputDir then
                    Directory.GetFiles(".", pattern)
                else
                    Directory.GetFiles(inputDir, pattern)

            Log.Information("Found {Count} file(s) matching pattern", files.Length)

            // Process each file
            let mutable successCount = 0
            let mutable errorCount = 0

            for file in files do
                let! result = applyToFile file outputDir suffix overwrite loadedModel useGpu
                match result with
                | Ok () -> successCount <- successCount + 1
                | Error err ->
                    Log.Error("Failed to process {File}: {Error}", Path.GetFileName file, err)
                    errorCount <- errorCount + 1

            Log.Information("Complete: {Success} succeeded, {Errors} failed", successCount, errorCount)

            if errorCount > 0 then
                return Error (ModelError $"{errorCount} file(s) failed to process")
            else
                return Ok ()
    }

// ============================================================================
// COMMAND LINE PARSING
// ============================================================================

let run (args: string[]) : int =

    if args.Length = 0 || args.[0] = "--help" || args.[0] = "-h" then
        showHelp()
        0
    else

    let mode = args.[0]

    match mode with
    | "train" ->
        // Parse train arguments
        let rec parseTrainArgs (args: string list) input model overwrite continueTraining (config: TrainingConfig) =
            match args with
            | [] -> (input, model, overwrite, continueTraining, config)
            | "--input" :: path :: rest | "-i" :: path :: rest ->
                parseTrainArgs rest (Some path) model overwrite continueTraining config
            | "--model" :: path :: rest ->
                parseTrainArgs rest input (Some path) overwrite continueTraining config
            | "--overwrite" :: rest ->
                parseTrainArgs rest input model true continueTraining config
            | "--continue" :: rest ->
                parseTrainArgs rest input model overwrite true config
            | "--epochs" :: n :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with NumEpochs = int n }
            | "--layers" :: n :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with NumLayers = int n }
            | "--filters" :: n :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with NumFilters = int n }
            | "--learning-rate" :: lr :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with LearningRate = float lr }
            | "--dropout" :: rate :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with DropoutRate = float rate }
            | "--cpu" :: rest ->
                parseTrainArgs rest input model overwrite continueTraining { config with UseGpu = false }
            | unknown :: _ ->
                Log.Error("Unknown argument: {Arg}", unknown)
                showHelp()
                exit 1

        let (input, model, overwrite, continueTraining, config) =
            parseTrainArgs (List.ofArray args.[1..]) None None false false defaultConfig

        match (input, model) with
        | (None, _) | (_, None) ->
            Log.Error("Missing required arguments for train mode")
            showHelp()
            1
        | (Some inputPath, Some modelPath) ->
            if not (File.Exists inputPath) then
                Log.Error("Input file not found: {Path}", inputPath)
                1
            else
                let result = trainMode inputPath modelPath overwrite continueTraining config |> Async.RunSynchronously
                match result with
                | Ok () -> 0
                | Error err ->
                    Log.Error("Training failed: {Error}", err)
                    1

    | "apply" ->
        // Parse apply arguments
        let rec parseApplyArgs (args: string list) input output model layers filters suffix overwrite useGpu =
            match args with
            | [] -> (input, output, model, layers, filters, suffix, overwrite, useGpu)
            | "--input" :: path :: rest | "-i" :: path :: rest ->
                parseApplyArgs rest (Some path) output model layers filters suffix overwrite useGpu
            | "--output" :: path :: rest | "-o" :: path :: rest ->
                parseApplyArgs rest input (Some path) model layers filters suffix overwrite useGpu
            | "--model" :: path :: rest ->
                parseApplyArgs rest input output (Some path) layers filters suffix overwrite useGpu
            | "--layers" :: n :: rest ->
                parseApplyArgs rest input output model (int n) filters suffix overwrite useGpu
            | "--filters" :: n :: rest ->
                parseApplyArgs rest input output model layers (int n) suffix overwrite useGpu
            | "--suffix" :: s :: rest ->
                parseApplyArgs rest input output model layers filters s overwrite useGpu
            | "--overwrite" :: rest ->
                parseApplyArgs rest input output model layers filters suffix true useGpu
            | "--cpu" :: rest ->
                parseApplyArgs rest input output model layers filters suffix overwrite false
            | unknown :: _ ->
                Log.Error("Unknown argument: {Arg}", unknown)
                showHelp()
                exit 1

        let (input, output, model, layers, filters, suffix, overwrite, useGpu) =
            parseApplyArgs (List.ofArray args.[1..]) None None None 5 48 "_denoised" false true

        match (input, output, model) with
        | (None, _, _) | (_, None, _) | (_, _, None) ->
            Log.Error("Missing required arguments for apply mode")
            showHelp()
            1
        | (Some inputPattern, Some outputDir, Some modelPath) ->
            if not (File.Exists modelPath) then
                Log.Error("Model file not found: {Path}", modelPath)
                1
            else
                let result = applyMode inputPattern outputDir modelPath layers filters suffix overwrite useGpu |> Async.RunSynchronously
                match result with
                | Ok () -> 0
                | Error err ->
                    Log.Error("Apply failed: {Error}", err)
                    1

    | _ ->
        Log.Error("Unknown mode: {Mode}. Use 'train' or 'apply'", mode)
        showHelp()
        1
