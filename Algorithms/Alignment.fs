module Algorithms.Alignment

open System
open FsToolkit.ErrorHandling
open XisfLib.Core
open Algorithms.StarDetection
open Algorithms.TriangleMatch
open Algorithms.SpatialMatch
open Algorithms.RBFTransform
open Algorithms.Interpolation
open Algorithms.Statistics

// ============================================================================
// TYPES - Core data structures for alignment pipeline
// ============================================================================

/// Raw image data - pixel buffer with dimensions
type ImageData = {
    Pixels: float[]
    Width: int
    Height: int
    Channels: int
}

/// Image with detected stars
type DetectedImage = {
    Image: ImageData
    Stars: DetectedStar[]
    MAD: float
}

/// Image with matched star correspondences
type MatchedImage = {
    Detected: DetectedImage
    RefStars: DetectedStar[]  // Reference stars needed for RBF
    MatchedPairs: (int * int)[]  // (refIdx, targetIdx)
    Transform: SpatialMatch.SimilarityTransform
    Diagnostics: TriangleMatch.AlignmentDiagnostics option  // Triangle matching diagnostics
}

/// RBF distortion result with full statistics
type DistortionResult = {
    Coefficients: RBFCoefficients option
    ResidualRMS: float
    MaxResidual: float
    ControlPointCount: int
}

type AlignmentError =
    | TransformEstimationFailed of matchedPairs: int
    | InsufficientAnchorPairs of found: int * needed: int
    | TransformationFailed of message: string

    override this.ToString() =
        match this with
        | TransformEstimationFailed pairs -> $"Failed to estimate transform from {pairs} matched pairs"
        | InsufficientAnchorPairs (found, needed) -> $"Expanding match failed to find sufficient anchor pairs ({found} found, need {needed})"
        | TransformationFailed msg -> $"Transform failed: {msg}"

/// Prepared reference frame for alignment
type ReferenceFrame = {
    Detected: DetectedImage
    Triangles: Triangle[]
}

// ============================================================================
// CONFIGURATION TYPES
// ============================================================================

type DetectionConfig = {
    Threshold: float
    GridSize: int
    MinFWHM: float
    MaxFWHM: float
    MaxEccentricity: float
    MaxStars: int
}

type MatchAlgorithm =
    | Triangle
    | Expanding

/// Matching algorithm parameters (without reference data)
type MatchingParams = {
    Algorithm: MatchAlgorithm
    RatioTolerance: float
    MaxStarsTriangles: int
    MinVotes: int
    AnchorStars: int
    AnchorDistribution: SpatialMatch.AnchorDistribution
}

type MatchingConfig = {
    RefStars: DetectedStar[]
    RefTriangles: Triangle[]
    Params: MatchingParams
}

type DistortionConfig = {
    Kernel: RBFKernel
    SupportFactor: float
    Regularization: float
}

type TransformConfig = {
    Interpolation: InterpolationMethod
    ApplyRBF: bool
}

/// Unified alignment configuration combining all alignment stages
type AlignmentConfig = {
    Detection: DetectionConfig
    Matching: MatchingParams
    Distortion: DistortionConfig option
    Transform: TransformConfig
}

// ============================================================================
// PIPELINE FUNCTIONS
// ============================================================================

/// Step 1: Detect stars in image
let detect (config: DetectionConfig) (img: ImageData) : DetectedImage =
    // Extract luminance channel
    let channel0 =
        if img.Channels = 1 then
            img.Pixels
        else
            Array.init (img.Width * img.Height) (fun i -> img.Pixels.[i * img.Channels])

    // Calculate MAD for threshold
    let sorted = Array.sort channel0
    let mid = sorted.Length / 2
    let median =
        if sorted.Length % 2 = 0 then
            (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else
            sorted.[mid]
    let mad = Statistics.calculateMAD channel0 median

    // Build detection params
    let detectionParams: DetectionParams = {
        Threshold = config.Threshold
        GridSize = config.GridSize
        MinFWHM = config.MinFWHM
        MaxFWHM = config.MaxFWHM
        MaxEccentricity = config.MaxEccentricity
        MaxStars = Some config.MaxStars
    }

    // Detect stars
    let starsResult =
        StarDetection.detectStarsInChannel
            channel0 img.Width img.Height mad 0 "Luminance" detectionParams

    {
        Image = img
        Stars = starsResult.Stars
        MAD = mad
    }

/// Step 4: Match detected stars to reference
let matchToReference (config: MatchingConfig) (fileName: string) (detected: DetectedImage) : Result<MatchedImage, AlignmentError> =
    let targetStars = detected.Stars
    let width = detected.Image.Width
    let height = detected.Image.Height

    match config.Params.Algorithm with
    | Triangle ->
        // Triangle matching algorithm
        let targetTriangles = TriangleMatch.formTriangles targetStars config.Params.MaxStarsTriangles
        let matches = TriangleMatch.matchTriangles config.RefTriangles targetTriangles config.Params.RatioTolerance
        let correspondences = TriangleMatch.voteForCorrespondences matches
        let sortedVotes = correspondences |> Map.toArray |> Array.sortByDescending snd

        // Build pairs for transform estimation
        let aboveThreshold = sortedVotes |> Array.filter (fun (_, v) -> v >= config.Params.MinVotes)
        let bestPerRef =
            aboveThreshold
            |> Array.groupBy (fun ((refIdx, _), _) -> refIdx)
            |> Array.map (fun (_, group) -> group |> Array.maxBy snd)
        let bijective =
            bestPerRef
            |> Array.groupBy (fun ((_, targetIdx), _) -> targetIdx)
            |> Array.map (fun (_, group) -> group |> Array.maxBy snd)

        let matchedPairs = bijective |> Array.map (fun ((refIdx, targetIdx), _) -> (refIdx, targetIdx))

        let pairs =
            matchedPairs
            |> Array.map (fun (refIdx, targetIdx) ->
                let r = config.RefStars.[refIdx]
                let t = targetStars.[targetIdx]
                (r.X, r.Y, t.X, t.Y))

        match SpatialMatch.estimateTransformFromPairs pairs with
        | Some transform ->
            // Compute diagnostics for triangle matching
            let matchPercentage =
                if config.RefTriangles.Length > 0 then
                    (float matches.Length / float config.RefTriangles.Length) * 100.0
                else 0.0

            let topVoteCount =
                if sortedVotes.Length > 0 then snd sortedVotes.[0]
                else 0

            let diagnostics = {
                FileName = fileName
                DetectedStars = targetStars.Length
                TrianglesFormed = targetTriangles.Length
                TrianglesMatched = matches.Length
                MatchPercentage = matchPercentage
                TopVoteCount = topVoteCount
                EstimatedTransform = Some (transform.Dx, transform.Dy, transform.Rotation, transform.Scale)
            }

            Ok {
                Detected = detected
                RefStars = config.RefStars
                MatchedPairs = matchedPairs
                Transform = transform
                Diagnostics = Some diagnostics
            }
        | None -> Error (TransformEstimationFailed matchedPairs.Length)

    | Expanding ->
        // Expanding match algorithm
        let expandConfig = {
            SpatialMatch.defaultExpandingConfig with
                AnchorStars = config.Params.AnchorStars
                AnchorDistribution = config.Params.AnchorDistribution
                RatioTolerance = config.Params.RatioTolerance
                MinAnchorVotes = config.Params.MinVotes
        }

        let result = SpatialMatch.matchExpanding config.RefStars targetStars width height expandConfig

        match result.Transform with
        | Some transform ->
            // Extract matched pairs via inverse transform
            let inverse = SpatialMatch.invertTransform transform
            let targetTree =
                targetStars
                |> Array.indexed
                |> Array.map (fun (i, s) -> (s, i))
                |> SpatialMatch.buildKdTree

            let matchedPairs = ResizeArray<int * int>()
            for refIdx in 0 .. config.RefStars.Length - 1 do
                let refStar = config.RefStars.[refIdx]
                let (predX, predY) = SpatialMatch.applyTransform inverse refStar.X refStar.Y
                match SpatialMatch.findNearest targetTree predX predY with
                | Some (_, targetIdx, dist) when dist < 15.0 ->
                    matchedPairs.Add((refIdx, targetIdx))
                | _ -> ()

            // Create diagnostics for expanding match (converted to similar format)
            let diagnostics = {
                FileName = fileName
                DetectedStars = targetStars.Length
                TrianglesFormed = 0  // Not applicable for expanding
                TrianglesMatched = result.AnchorPairs
                MatchPercentage = 0.0  // Not applicable
                TopVoteCount = result.TotalInliers
                EstimatedTransform = Some (transform.Dx, transform.Dy, transform.Rotation, transform.Scale)
            }

            Ok {
                Detected = detected
                RefStars = config.RefStars
                MatchedPairs = matchedPairs.ToArray()
                Transform = transform
                Diagnostics = Some diagnostics
            }
        | None -> Error (InsufficientAnchorPairs (result.AnchorPairs, config.Params.AnchorStars))

/// Step 5: Compute RBF distortion correction (or skip)
let computeDistortion (config: DistortionConfig option) (matched: MatchedImage) : (MatchedImage * DistortionResult option) =
    match config with
    | None ->
        (matched, None)  // Skip track - no distortion correction
    | Some distConfig ->
        // Main track - compute RBF
        let rbfConfig: RBFTransform.RBFConfig = {
            Kernel = distConfig.Kernel
            SupportRadiusFactor = distConfig.SupportFactor
            ShapeFactor = 0.5
            Regularization = distConfig.Regularization
            MinControlPoints = 20
            MaxControlPoints = 500
        }

        let refStars = matched.RefStars
        let targetStars = matched.Detected.Stars
        let width = matched.Detected.Image.Width
        let height = matched.Detected.Image.Height

        let rbfResult =
            RBFTransform.setupRBF
                refStars targetStars matched.MatchedPairs
                matched.Transform rbfConfig width height

        // Package full result with statistics
        let distResult: DistortionResult = {
            Coefficients = rbfResult.Coefficients
            ResidualRMS = rbfResult.ResidualRMS
            MaxResidual = rbfResult.MaxResidual
            ControlPointCount = rbfResult.ControlPointCount
        }

        (matched, Some distResult)

/// Step 6: Apply geometric transformation
let transform (config: TransformConfig) (matched: MatchedImage, rbfCoeffs: RBFCoefficients option) : Result<ImageData, AlignmentError> =
    try
        // Apply transformation
        let img = matched.Detected.Image
        let pixelCount = img.Width * img.Height

        // Pre-extract channels for efficient sampling
        let channelArrays =
            if img.Channels = 1 then
                [| img.Pixels |]
            else
                [| for ch in 0 .. img.Channels - 1 ->
                    Array.init pixelCount (fun i -> img.Pixels.[i * img.Channels + ch]) |]

        // Transform in parallel - each output pixel is independent
        let outputPixels = Array.zeroCreate (pixelCount * img.Channels)

        let rbfToUse = if config.ApplyRBF then rbfCoeffs else None

        Array.Parallel.iter (fun pixIdx ->
            let ox = pixIdx % img.Width
            let oy = pixIdx / img.Width

            // Transform output coords to target coords (with RBF if available)
            let (tx, ty) =
                RBFTransform.applyFullInverseTransform
                    matched.Transform rbfToUse (float ox) (float oy) 5 0.01

            // Sample each channel
            for ch in 0 .. img.Channels - 1 do
                let value = Interpolation.sample config.Interpolation channelArrays.[ch] img.Width img.Height tx ty
                outputPixels.[pixIdx * img.Channels + ch] <- value
        ) [| 0 .. pixelCount - 1 |]

        Ok {
            Pixels = outputPixels
            Width = img.Width
            Height = img.Height
            Channels = img.Channels
        }
    with ex ->
        Error (TransformationFailed ex.Message)

// ============================================================================
// HIGH-LEVEL API
// ============================================================================

/// Prepare reference frame for alignment
let prepareReference (config: DetectionConfig) (maxStarsTriangles: int) (refImage: ImageData) : ReferenceFrame =
    let detected = detect config refImage
    let triangles = TriangleMatch.formTriangles detected.Stars maxStarsTriangles
    {
        Detected = detected
        Triangles = triangles
    }

/// Full alignment pipeline orchestrating all stages
let align (reference: ReferenceFrame) (target: ImageData) (config: AlignmentConfig) (fileName: string) : Result<ImageData, AlignmentError> =
    result {
        // DETECT
        let detected = detect config.Detection target
        if detected.Stars.Length < 10 then
            return! Error (TransformationFailed "Insufficient stars detected in target image")

        // MATCH
        let matchConfig: MatchingConfig = {
            RefStars = reference.Detected.Stars
            RefTriangles = reference.Triangles
            Params = config.Matching
        }
        let! matched = matchToReference matchConfig fileName detected

        // DISTORT
        let (matched', rbfResult) = computeDistortion config.Distortion matched
        let rbfCoeffs = rbfResult |> Option.bind (fun r -> r.Coefficients)

        // TRANSFORM
        return! transform config.Transform (matched', rbfCoeffs)
    }
