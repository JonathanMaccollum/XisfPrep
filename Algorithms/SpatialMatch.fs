module Algorithms.SpatialMatch

open System

// Re-use star type from StarDetection
type DetectedStar = StarDetection.DetectedStar

// --- KD-Tree for 2D Spatial Lookups ---

type KdNode =
    | Leaf of DetectedStar * int  // Star and its original index
    | Branch of axis: int * median: float * left: KdNode * right: KdNode
    | Empty

/// Build a 2D KD-tree from stars with their original indices
let buildKdTree (stars: (DetectedStar * int)[]) : KdNode =
    let rec build (points: (DetectedStar * int)[]) (depth: int) : KdNode =
        if points.Length = 0 then Empty
        elif points.Length = 1 then
            let (star, idx) = points.[0]
            Leaf (star, idx)
        else
            let axis = depth % 2
            let sorted =
                if axis = 0 then points |> Array.sortBy (fun (s, _) -> s.X)
                else points |> Array.sortBy (fun (s, _) -> s.Y)

            let mid = sorted.Length / 2
            let (medianStar, _) = sorted.[mid]
            let median = if axis = 0 then medianStar.X else medianStar.Y

            let left = sorted.[0 .. mid - 1]
            let right = sorted.[mid ..]

            Branch (axis, median, build left (depth + 1), build right (depth + 1))

    build stars 0

/// Find all stars within radius of point
let findWithinRadius (tree: KdNode) (x: float) (y: float) (radius: float) : (DetectedStar * int * float)[] =
    let results = ResizeArray<DetectedStar * int * float>()
    let radiusSq = radius * radius

    let rec search node depth =
        match node with
        | Empty -> ()
        | Leaf (star, idx) ->
            let dx = star.X - x
            let dy = star.Y - y
            let distSq = dx * dx + dy * dy
            if distSq <= radiusSq then
                results.Add(star, idx, sqrt distSq)
        | Branch (axis, median, left, right) ->
            let coord = if axis = 0 then x else y
            let diff = coord - median

            // Search the side containing the query point first
            if diff <= 0.0 then
                search left (depth + 1)
                if diff + radius >= 0.0 then search right (depth + 1)
            else
                search right (depth + 1)
                if diff - radius <= 0.0 then search left (depth + 1)

    search tree 0
    results.ToArray()

/// Find nearest star to point
let findNearest (tree: KdNode) (x: float) (y: float) : (DetectedStar * int * float) option =
    let mutable best: (DetectedStar * int * float) option = None
    let mutable bestDistSq = Double.MaxValue

    let rec search node depth =
        match node with
        | Empty -> ()
        | Leaf (star, idx) ->
            let dx = star.X - x
            let dy = star.Y - y
            let distSq = dx * dx + dy * dy
            if distSq < bestDistSq then
                bestDistSq <- distSq
                best <- Some (star, idx, sqrt distSq)
        | Branch (axis, median, left, right) ->
            let coord = if axis = 0 then x else y
            let diff = coord - median

            // Search the closer side first
            let (first, second) = if diff <= 0.0 then (left, right) else (right, left)
            search first (depth + 1)

            // Only search the other side if it could contain a closer point
            if diff * diff < bestDistSq then
                search second (depth + 1)

    search tree 0
    best

// --- Similarity Transform ---

type SimilarityTransform = {
    A: float      // cos(θ) * scale
    B: float      // sin(θ) * scale
    Dx: float     // X translation
    Dy: float     // Y translation
    Scale: float
    Rotation: float  // degrees
}

/// Apply transform: Reference = Transform(Target)
let applyTransform (t: SimilarityTransform) (tx: float) (ty: float) : float * float =
    let rx = t.A * tx - t.B * ty + t.Dx
    let ry = t.B * tx + t.A * ty + t.Dy
    (rx, ry)

/// Invert transform: Target = InverseTransform(Reference)
let invertTransform (t: SimilarityTransform) : SimilarityTransform =
    let det = t.A * t.A + t.B * t.B
    let aInv = t.A / det
    let bInv = -t.B / det
    let dxInv = -(aInv * t.Dx - bInv * t.Dy)
    let dyInv = -(bInv * t.Dx + aInv * t.Dy)
    {
        A = aInv
        B = bInv
        Dx = dxInv
        Dy = dyInv
        Scale = 1.0 / t.Scale
        Rotation = -t.Rotation
    }

/// Estimate similarity transform from correspondence pairs
/// Each pair is (refX, refY, targetX, targetY)
let estimateTransformFromPairs (pairs: (float * float * float * float)[]) : SimilarityTransform option =
    if pairs.Length < 2 then None
    else
        let n = float pairs.Length

        let sumTx = pairs |> Array.sumBy (fun (_, _, tx, _) -> tx)
        let sumTy = pairs |> Array.sumBy (fun (_, _, _, ty) -> ty)
        let sumRx = pairs |> Array.sumBy (fun (rx, _, _, _) -> rx)
        let sumRy = pairs |> Array.sumBy (fun (_, ry, _, _) -> ry)

        let sumTxRx = pairs |> Array.sumBy (fun (rx, _, tx, _) -> tx * rx)
        let sumTyRy = pairs |> Array.sumBy (fun (_, ry, _, ty) -> ty * ry)
        let sumTxRy = pairs |> Array.sumBy (fun (_, ry, tx, _) -> tx * ry)
        let sumTyRx = pairs |> Array.sumBy (fun (rx, _, _, ty) -> ty * rx)

        let sumTx2 = pairs |> Array.sumBy (fun (_, _, tx, _) -> tx * tx)
        let sumTy2 = pairs |> Array.sumBy (fun (_, _, _, ty) -> ty * ty)

        let denom = n * (sumTx2 + sumTy2) - sumTx * sumTx - sumTy * sumTy

        if abs denom < 1e-10 then None
        else
            let a = (n * (sumTxRx + sumTyRy) - sumTx * sumRx - sumTy * sumRy) / denom
            let b = (n * (sumTxRy - sumTyRx) - sumTx * sumRy + sumTy * sumRx) / denom

            let dx = (sumRx - a * sumTx + b * sumTy) / n
            let dy = (sumRy - b * sumTx - a * sumTy) / n

            let scale = sqrt (a * a + b * b)
            let rotation = atan2 b a * 180.0 / Math.PI

            Some {
                A = a
                B = b
                Dx = dx
                Dy = dy
                Scale = scale
                Rotation = rotation
            }

// --- RANSAC Outlier Rejection ---

type RansacConfig = {
    /// Number of RANSAC iterations
    Iterations: int
    /// Minimum pairs to estimate transform
    SampleSize: int
    /// Distance threshold for inliers (pixels)
    InlierThreshold: float
    /// Minimum ratio of inliers required
    MinInlierRatio: float
}

let defaultRansacConfig = {
    Iterations = 500
    SampleSize = 3
    InlierThreshold = 2.0
    MinInlierRatio = 0.3
}

/// Apply transform and compute error for a pair
let private computePairError (t: SimilarityTransform) (rx: float, ry: float, tx: float, ty: float) =
    let (px, py) = applyTransform t tx ty
    sqrt ((rx - px) ** 2.0 + (ry - py) ** 2.0)

/// RANSAC refinement for robust transform estimation
let ransacRefineTransform (pairs: (float * float * float * float)[]) (config: RansacConfig) : (SimilarityTransform * int) option =
    if pairs.Length < config.SampleSize then None
    else
        let rng = System.Random()
        let mutable bestTransform: SimilarityTransform option = None
        let mutable bestInlierCount = 0
        let mutable bestInlierIndices: int[] = [||]

        for _ in 1 .. config.Iterations do
            // 1. Randomly sample pairs
            let indices =
                [| for _ in 1 .. config.SampleSize -> rng.Next(pairs.Length) |]
                |> Array.distinct

            if indices.Length >= config.SampleSize then
                let sample = indices |> Array.map (fun i -> pairs.[i])

                // 2. Estimate transform from sample
                match estimateTransformFromPairs sample with
                | None -> ()
                | Some t ->
                    // 3. Count inliers
                    let inlierIndices =
                        pairs
                        |> Array.indexed
                        |> Array.filter (fun (_, p) -> computePairError t p < config.InlierThreshold)
                        |> Array.map fst

                    // 4. Keep if best so far
                    if inlierIndices.Length > bestInlierCount then
                        bestInlierCount <- inlierIndices.Length
                        bestInlierIndices <- inlierIndices
                        bestTransform <- Some t

        // 5. Re-estimate from all inliers if we found enough
        let minRequired = int (float pairs.Length * config.MinInlierRatio)
        if bestInlierCount >= minRequired && bestInlierCount >= config.SampleSize then
            let inlierPairs = bestInlierIndices |> Array.map (fun i -> pairs.[i])
            match estimateTransformFromPairs inlierPairs with
            | Some refined -> Some (refined, bestInlierCount)
            | None -> bestTransform |> Option.map (fun t -> (t, bestInlierCount))
        else
            bestTransform |> Option.map (fun t -> (t, bestInlierCount))

// --- Center-Seeded Anchor Finding ---

type AnchorDistribution =
    | Center  // Select from central region only
    | Grid    // Distribute across grid cells

/// Select brightest stars within central region of image
let selectCenterBright (stars: DetectedStar[]) (imageWidth: int) (imageHeight: int) (maxStars: int) (centerFraction: float) : (DetectedStar * int)[] =
    let cx = float imageWidth / 2.0
    let cy = float imageHeight / 2.0
    let halfW = float imageWidth * centerFraction / 2.0
    let halfH = float imageHeight * centerFraction / 2.0

    stars
    |> Array.indexed
    |> Array.filter (fun (_, s) ->
        abs (s.X - cx) <= halfW && abs (s.Y - cy) <= halfH)
    |> Array.sortByDescending (fun (_, s) -> s.Flux)
    |> Array.truncate maxStars
    |> Array.map (fun (i, s) -> (s, i))

/// Select brightest stars distributed across a grid pattern
let selectGridDistributed (stars: DetectedStar[]) (imageWidth: int) (imageHeight: int) (maxStars: int) (gridSize: int) : (DetectedStar * int)[] =
    let cellWidth = float imageWidth / float gridSize
    let cellHeight = float imageHeight / float gridSize
    let starsPerCell = max 1 (maxStars / (gridSize * gridSize))

    // Index stars with original indices
    let indexed = stars |> Array.indexed

    // Group stars by grid cell
    let cellStars =
        indexed
        |> Array.groupBy (fun (_, s) ->
            let cellX = min (gridSize - 1) (int (s.X / cellWidth))
            let cellY = min (gridSize - 1) (int (s.Y / cellHeight))
            (cellX, cellY))

    // Take brightest N from each cell
    let selected =
        cellStars
        |> Array.collect (fun (_, group) ->
            group
            |> Array.sortByDescending (fun (_, s) -> s.Flux)
            |> Array.truncate starsPerCell)

    // Sort all by brightness and take top maxStars
    selected
    |> Array.sortByDescending (fun (_, s) -> s.Flux)
    |> Array.truncate maxStars
    |> Array.map (fun (i, s) -> (s, i))

/// Select anchors using specified distribution method
let selectAnchors (stars: DetectedStar[]) (imageWidth: int) (imageHeight: int) (maxStars: int) (distribution: AnchorDistribution) : (DetectedStar * int)[] =
    match distribution with
    | Center -> selectCenterBright stars imageWidth imageHeight maxStars 0.6
    | Grid -> selectGridDistributed stars imageWidth imageHeight maxStars 3

/// Triangle descriptor for quick matching
type TriangleDescriptor = {
    Vertices: int[]  // Original star indices
    RatioAB: float
    RatioBC: float
}

/// Form triangles from indexed stars
let formTrianglesFromIndexed (starsWithIdx: (DetectedStar * int)[]) : TriangleDescriptor[] =
    let n = starsWithIdx.Length
    if n < 3 then [||]
    else
        let triangles = ResizeArray<TriangleDescriptor>()

        for i in 0 .. n - 3 do
            for j in i + 1 .. n - 2 do
                for k in j + 1 .. n - 1 do
                    let (s1, idx1) = starsWithIdx.[i]
                    let (s2, idx2) = starsWithIdx.[j]
                    let (s3, idx3) = starsWithIdx.[k]

                    let d12 = sqrt ((s1.X - s2.X) ** 2.0 + (s1.Y - s2.Y) ** 2.0)
                    let d23 = sqrt ((s2.X - s3.X) ** 2.0 + (s2.Y - s3.Y) ** 2.0)
                    let d31 = sqrt ((s3.X - s1.X) ** 2.0 + (s3.Y - s1.Y) ** 2.0)

                    let sides = [| (d12, idx3); (d23, idx1); (d31, idx2) |] |> Array.sortBy fst

                    let sideA = fst sides.[0]
                    let sideB = fst sides.[1]
                    let sideC = fst sides.[2]

                    if sideA > 5.0 && sideC > 0.0 then
                        triangles.Add {
                            Vertices = [| snd sides.[0]; snd sides.[1]; snd sides.[2] |]
                            RatioAB = sideA / sideB
                            RatioBC = sideB / sideC
                        }

        triangles.ToArray()

/// Match anchor triangles and return correspondence votes
let matchAnchorTriangles (refTris: TriangleDescriptor[]) (targetTris: TriangleDescriptor[]) (tolerance: float) : Map<(int * int), int> =
    let votes = System.Collections.Generic.Dictionary<(int * int), int>()

    for refTri in refTris do
        for targetTri in targetTris do
            let errorAB = abs (refTri.RatioAB - targetTri.RatioAB)
            let errorBC = abs (refTri.RatioBC - targetTri.RatioBC)

            if errorAB + errorBC < tolerance then
                for i in 0 .. 2 do
                    let pair = (refTri.Vertices.[i], targetTri.Vertices.[i])
                    if votes.ContainsKey pair then
                        votes.[pair] <- votes.[pair] + 1
                    else
                        votes.[pair] <- 1

    votes |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq

/// Extract best bijective correspondences from votes
let extractBijectiveCorrespondences (votes: Map<(int * int), int>) (minVotes: int) : ((int * int) * int)[] =
    let aboveThreshold =
        votes |> Map.toArray |> Array.filter (fun (_, v) -> v >= minVotes)

    // Best target per reference
    let bestPerRef =
        aboveThreshold
        |> Array.groupBy (fun ((refIdx, _), _) -> refIdx)
        |> Array.map (fun (_, group) -> group |> Array.maxBy snd)

    // Ensure bijective: best reference per target
    bestPerRef
    |> Array.groupBy (fun ((_, targetIdx), _) -> targetIdx)
    |> Array.map (fun (_, group) -> group |> Array.maxBy snd)

// --- Expanding Match Algorithm ---

type ExpandingMatchConfig = {
    /// Number of anchor stars from image center
    AnchorStars: int
    /// Anchor distribution method
    AnchorDistribution: AnchorDistribution
    /// Fraction of image considered "center" (0.5 = central 50%)
    CenterFraction: float
    /// Tolerance for triangle ratio matching
    RatioTolerance: float
    /// Minimum votes to consider a valid anchor correspondence
    MinAnchorVotes: int
    /// Search radius for propagation (pixels)
    PropagationRadius: float
    /// Minimum inliers for valid transform
    MinInliers: int
    /// RANSAC configuration (None to disable)
    Ransac: RansacConfig option
    /// Number of refinement iterations (0 = no refinement)
    RefinementIterations: int
    /// Tighter radius for refinement phase (multiplier of PropagationRadius)
    RefinementRadiusFactor: float
}

let defaultExpandingConfig = {
    AnchorStars = 12
    AnchorDistribution = Center
    CenterFraction = 0.6
    RatioTolerance = 0.05
    MinAnchorVotes = 2
    PropagationRadius = 30.0
    MinInliers = 6
    Ransac = Some defaultRansacConfig
    RefinementIterations = 1
    RefinementRadiusFactor = 0.5
}

/// Result of expanding match algorithm
type ExpandingMatchResult = {
    Transform: SimilarityTransform option
    AnchorPairs: int
    PropagatedPairs: int
    TotalInliers: int
    MatchedTargetIndices: Set<int>
}

/// Run expanding match algorithm
let matchExpanding
    (refStars: DetectedStar[])
    (targetStars: DetectedStar[])
    (imageWidth: int)
    (imageHeight: int)
    (config: ExpandingMatchConfig)
    : ExpandingMatchResult =

    // Phase 1: Find anchor correspondences using selected distribution
    let centerRef = selectAnchors refStars imageWidth imageHeight config.AnchorStars config.AnchorDistribution
    let centerTarget = selectAnchors targetStars imageWidth imageHeight config.AnchorStars config.AnchorDistribution

    let refTris = formTrianglesFromIndexed centerRef
    let targetTris = formTrianglesFromIndexed centerTarget

    let votes = matchAnchorTriangles refTris targetTris config.RatioTolerance
    let anchors = extractBijectiveCorrespondences votes config.MinAnchorVotes

    if anchors.Length < 2 then
        { Transform = None; AnchorPairs = 0; PropagatedPairs = 0; TotalInliers = 0; MatchedTargetIndices = Set.empty }
    else
        // Estimate initial transform from anchors
        let anchorPairs =
            anchors |> Array.map (fun ((refIdx, targetIdx), _) ->
                let r = refStars.[refIdx]
                let t = targetStars.[targetIdx]
                (r.X, r.Y, t.X, t.Y))

        match estimateTransformFromPairs anchorPairs with
        | None ->
            let anchorTargetIndices = anchors |> Array.map (fun ((_, ti), _) -> ti) |> Set.ofArray
            { Transform = None; AnchorPairs = anchors.Length; PropagatedPairs = 0; TotalInliers = 0; MatchedTargetIndices = anchorTargetIndices }
        | Some initialTransform ->
            // Phase 2: Propagate using KD-tree lookups
            let inverse = invertTransform initialTransform
            let targetTree =
                targetStars
                |> Array.indexed
                |> Array.map (fun (i, s) -> (s, i))
                |> buildKdTree

            let propagated = ResizeArray<float * float * float * float>()
            let propagatedIndices = ResizeArray<int>()
            let anchorTargetIndices = anchors |> Array.map (fun ((_, ti), _) -> ti) |> Set.ofArray

            for refIdx in 0 .. refStars.Length - 1 do
                let refStar = refStars.[refIdx]
                // Predict target position
                let (predX, predY) = applyTransform inverse refStar.X refStar.Y

                // Search for candidate
                match findNearest targetTree predX predY with
                | Some (targetStar, targetIdx, dist) when dist < config.PropagationRadius ->
                    // Don't duplicate anchor pairs
                    if not (anchorTargetIndices.Contains targetIdx) then
                        propagated.Add(refStar.X, refStar.Y, targetStar.X, targetStar.Y)
                        propagatedIndices.Add(targetIdx)
                | _ -> ()

            // Phase 3: Re-estimate transform with all pairs (optionally using RANSAC)
            let allPairs = Array.append anchorPairs (propagated.ToArray())
            let allMatchedIndices = Set.union anchorTargetIndices (Set.ofSeq propagatedIndices)

            let (phase3Transform, phase3InlierCount) =
                match config.Ransac with
                | Some ransacConfig ->
                    // Use RANSAC for robust estimation
                    match ransacRefineTransform allPairs ransacConfig with
                    | Some (t, count) -> (Some t, count)
                    | None -> (Some initialTransform, anchors.Length)
                | None ->
                    // Simple estimation without RANSAC
                    match estimateTransformFromPairs allPairs with
                    | Some t -> (Some t, allPairs.Length)
                    | None -> (Some initialTransform, anchors.Length)

            // Phase 4: Refinement iterations (re-propagate with tighter radius)
            if config.RefinementIterations <= 0 || phase3Transform.IsNone then
                { Transform = phase3Transform
                  AnchorPairs = anchors.Length
                  PropagatedPairs = propagated.Count
                  TotalInliers = phase3InlierCount
                  MatchedTargetIndices = allMatchedIndices }
            else
                let mutable currentTransform = phase3Transform.Value
                let mutable currentInlierCount = phase3InlierCount
                let mutable currentMatchedIndices = allMatchedIndices
                let mutable currentPropagatedCount = propagated.Count

                for _ in 1 .. config.RefinementIterations do
                    // Re-propagate with refined transform and tighter radius
                    let refinedInverse = invertTransform currentTransform
                    let refinedRadius = config.PropagationRadius * config.RefinementRadiusFactor

                    let refinedPairs = ResizeArray<float * float * float * float>()
                    let refinedIndices = ResizeArray<int>()

                    for refIdx in 0 .. refStars.Length - 1 do
                        let refStar = refStars.[refIdx]
                        let (predX, predY) = applyTransform refinedInverse refStar.X refStar.Y

                        match findNearest targetTree predX predY with
                        | Some (targetStar, targetIdx, dist) when dist < refinedRadius ->
                            refinedPairs.Add(refStar.X, refStar.Y, targetStar.X, targetStar.Y)
                            refinedIndices.Add(targetIdx)
                        | _ -> ()

                    // Re-estimate with RANSAC if configured
                    let refinedPairsArray = refinedPairs.ToArray()
                    if refinedPairsArray.Length >= config.MinInliers then
                        match config.Ransac with
                        | Some ransacConfig ->
                            match ransacRefineTransform refinedPairsArray ransacConfig with
                            | Some (t, count) ->
                                currentTransform <- t
                                currentInlierCount <- count
                                currentMatchedIndices <- Set.ofSeq refinedIndices
                                currentPropagatedCount <- refinedPairs.Count
                            | None -> ()
                        | None ->
                            match estimateTransformFromPairs refinedPairsArray with
                            | Some t ->
                                currentTransform <- t
                                currentInlierCount <- refinedPairsArray.Length
                                currentMatchedIndices <- Set.ofSeq refinedIndices
                                currentPropagatedCount <- refinedPairs.Count
                            | None -> ()

                { Transform = Some currentTransform
                  AnchorPairs = anchors.Length
                  PropagatedPairs = currentPropagatedCount
                  TotalInliers = currentInlierCount
                  MatchedTargetIndices = currentMatchedIndices }
