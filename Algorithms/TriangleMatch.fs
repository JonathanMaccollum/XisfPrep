module Algorithms.TriangleMatch

open System
open Serilog

type DetectedStar = StarDetection.DetectedStar

// --- Triangle Matching Types ---

type Triangle = {
    /// Indices into the ORIGINAL star array, sorted geometrically:
    /// [0] = Vertex opposite Side A (Shortest side)
    /// [1] = Vertex opposite Side B (Middle side)
    /// [2] = Vertex opposite Side C (Longest side)
    Vertices: int[]

    /// Side lengths sorted ascending (scale-invariant descriptor)
    SideA: float  // shortest
    SideB: float  // middle
    SideC: float  // longest
    /// Ratios for matching (rotation and scale invariant)
    RatioAB: float  // SideA / SideB
    RatioBC: float  // SideB / SideC
}

type TriangleMatchResult = {
    RefTriangle: Triangle
    TargetTriangle: Triangle
    RatioError: float  // Combined error in ratios
}

type AlignmentDiagnostics = {
    FileName: string
    DetectedStars: int
    TrianglesFormed: int
    TrianglesMatched: int
    MatchPercentage: float
    TopVoteCount: int
    EstimatedTransform: (float * float * float * float) option  // dx, dy, rotation, scale
}

// --- Triangle Matching Algorithm ---

/// Form triangles from the brightest N stars
/// Returns triangles with scale/rotation invariant descriptors
let formTriangles (stars: DetectedStar[]) (maxStars: int) : Triangle[] =
    // 1. Preserve ORIGINAL indices by pairing them before sorting/filtering
    let topStarsWithIndices =
        stars
        |> Array.indexed // Creates (originalIndex, Star) tuple
        |> Array.sortByDescending (fun (_, s) -> s.Flux)
        |> Array.truncate maxStars

    let n = topStarsWithIndices.Length
    if n < 3 then [||]
    else
        let triangles = ResizeArray<Triangle>()

        // Form all possible triangles (n choose 3)
        for i in 0 .. n - 3 do
            for j in i + 1 .. n - 2 do
                for k in j + 1 .. n - 1 do
                    // Extract original indices and star objects
                    let (idx1, s1) = topStarsWithIndices.[i]
                    let (idx2, s2) = topStarsWithIndices.[j]
                    let (idx3, s3) = topStarsWithIndices.[k]

                    // Calculate side lengths
                    let d12 = sqrt ((s1.X - s2.X) ** 2.0 + (s1.Y - s2.Y) ** 2.0)
                    let d23 = sqrt ((s2.X - s3.X) ** 2.0 + (s2.Y - s3.Y) ** 2.0)
                    let d31 = sqrt ((s3.X - s1.X) ** 2.0 + (s3.Y - s1.Y) ** 2.0)

                    // Create array of (Length, OppositeVertexOriginalIndex)
                    // d12 connects 1-2, so opposite is 3
                    // d23 connects 2-3, so opposite is 1
                    // d31 connects 3-1, so opposite is 2
                    let rawSides : (float * int)[] =
                        [| (d12, idx3); (d23, idx1); (d31, idx2) |]

                    // Sort by length (ascending) to establish geometry
                    let sortedSides = rawSides |> Array.sortBy fst

                    let sideA = fst sortedSides.[0] // Shortest
                    let sideB = fst sortedSides.[1] // Middle
                    let sideC = fst sortedSides.[2] // Longest

                    // Extract the vertices in geometric order (Opposite Shortest, Opposite Middle, Opposite Longest)
                    let orderedVertices =
                        [| snd sortedSides.[0]; snd sortedSides.[1]; snd sortedSides.[2] |]

                    // Skip degenerate triangles
                    if sideA > 5.0 && sideC > 0.0 then
                        let ratioAB = sideA / sideB
                        let ratioBC = sideB / sideC

                        triangles.Add {
                            Vertices = orderedVertices
                            SideA = sideA
                            SideB = sideB
                            SideC = sideC
                            RatioAB = ratioAB
                            RatioBC = ratioBC
                        }

        triangles.ToArray()

/// Binary search for lower bound index where RatioAB >= value
let private binarySearchLower (sorted: Triangle[]) (value: float) : int =
    let mutable lo = 0
    let mutable hi = sorted.Length
    while lo < hi do
        let mid = lo + (hi - lo) / 2
        if sorted.[mid].RatioAB < value then
            lo <- mid + 1
        else
            hi <- mid
    lo

/// Binary search for upper bound index where RatioAB <= value
let private binarySearchUpper (sorted: Triangle[]) (value: float) : int =
    let mutable lo = 0
    let mutable hi = sorted.Length
    while lo < hi do
        let mid = lo + (hi - lo) / 2
        if sorted.[mid].RatioAB <= value then
            lo <- mid + 1
        else
            hi <- mid
    lo

/// Match triangles between reference and target by ratio similarity
/// Uses sorted array + binary search for O(T log R) instead of O(T × R)
let matchTriangles (refTriangles: Triangle[]) (targetTriangles: Triangle[]) (tolerance: float) : TriangleMatchResult[] =
    if refTriangles.Length = 0 || targetTriangles.Length = 0 then [||]
    else
        // Sort reference triangles by RatioAB for binary search
        let sortedRef = refTriangles |> Array.sortBy (fun t -> t.RatioAB)
        let matches = ResizeArray<TriangleMatchResult>()

        for targetTri in targetTriangles do
            // Binary search for RatioAB range
            let lo = binarySearchLower sortedRef (targetTri.RatioAB - tolerance)
            let hi = binarySearchUpper sortedRef (targetTri.RatioAB + tolerance)

            // Check candidates within range
            for i in lo .. hi - 1 do
                let refTri = sortedRef.[i]
                let errorAB = abs (refTri.RatioAB - targetTri.RatioAB)
                let errorBC = abs (refTri.RatioBC - targetTri.RatioBC)
                let totalError = errorAB + errorBC

                if totalError < tolerance then
                    matches.Add {
                        RefTriangle = refTri
                        TargetTriangle = targetTri
                        RatioError = totalError
                    }

        matches.ToArray()

/// Vote for star correspondences from triangle matches
/// Returns map of (refStarIdx, targetStarIdx) -> vote count
let voteForCorrespondences (matches: TriangleMatchResult[]) : Map<(int * int), int> =
    let votes = System.Collections.Generic.Dictionary<(int * int), int>()

    for m in matches do
        // Since Vertices are sorted geometrically (Opposite Shortest, Opposite Middle, Opposite Longest),
        // we can map them directly index-to-index.
        for i in 0 .. 2 do
            let refIdx = m.RefTriangle.Vertices.[i]
            let targetIdx = m.TargetTriangle.Vertices.[i]
            let pair = (refIdx, targetIdx)

            if votes.ContainsKey pair then
                votes.[pair] <- votes.[pair] + 1
            else
                votes.[pair] <- 1

    votes |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq

/// Estimate similarity transform from matched star pairs using least squares
/// Returns (dx, dy, rotation in degrees, scale)
let estimateTransform
    (refStars: DetectedStar[])
    (targetStars: DetectedStar[])
    (correspondences: ((int * int) * int)[])
    (minVotes: int)
    : (float * float * float * float) option =

    // Filter by vote count
    let aboveThreshold =
        correspondences |> Array.filter (fun (_, votes) -> votes >= minVotes)

    // Select best target for each reference star (highest votes)
    let bestPerRef =
        aboveThreshold
        |> Array.groupBy (fun ((refIdx, _), _) -> refIdx)
        |> Array.map (fun (_, group) ->
            group |> Array.maxBy (fun (_, votes) -> votes))

    // Also ensure each target is used only once (full bijective mapping)
    let filteredCorrespondences =
        bestPerRef
        |> Array.groupBy (fun ((_, targetIdx), _) -> targetIdx)
        |> Array.map (fun (_, group) ->
            group |> Array.maxBy (fun (_, votes) -> votes))

    // Debug: log first few correspondences
    for i in 0 .. min 4 (filteredCorrespondences.Length - 1) do
        let ((refIdx, targetIdx), votes) = filteredCorrespondences.[i]
        let r = refStars.[refIdx]
        let t = targetStars.[targetIdx]
        Log.Verbose("Pair {I}: Ref[{RefIdx}]=({RefX:F1},{RefY:F1}) -> Target[{TargetIdx}]=({TargetX:F1},{TargetY:F1}) votes={Votes}",
            i, refIdx, r.X, r.Y, targetIdx, t.X, t.Y, votes)

    let pairs =
        filteredCorrespondences
        |> Array.map (fun ((refIdx, targetIdx), _) ->
            let r = refStars.[refIdx]
            let t = targetStars.[targetIdx]
            (r.X, r.Y, t.X, t.Y))

    // Debug: count pairs with matching vs mismatching coordinates
    let matchingCount = pairs |> Array.filter (fun (rx, ry, tx, ty) ->
        abs(rx - tx) < 0.1 && abs(ry - ty) < 0.1) |> Array.length
    Log.Verbose("Total pairs for transform: {Count} ({Matching} with matching coords)",
        pairs.Length, matchingCount)

    if pairs.Length < 2 then None
    else
        // Solve for similarity transform:
        // Rx = a*Tx - b*Ty + dx
        // Ry = b*Tx + a*Ty + dy
        // Where scale = sqrt(a² + b²), rotation = atan2(b, a)

        let n = float pairs.Length

        // T = Target (x, y), R = Reference (rx, ry)
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

        // Denominator
        let denom = n * (sumTx2 + sumTy2) - sumTx * sumTx - sumTy * sumTy

        Log.Verbose("Sums: n={N} sumTx={SumTx:F1} sumTy={SumTy:F1} sumRx={SumRx:F1} sumRy={SumRy:F1}",
            n, sumTx, sumTy, sumRx, sumRy)
        Log.Verbose("Cross: sumTxRx={TxRx:F1} sumTyRy={TyRy:F1} sumTxRy={TxRy:F1} sumTyRx={TyRx:F1}",
            sumTxRx, sumTyRy, sumTxRy, sumTyRx)
        Log.Verbose("Denom={Denom:F1}", denom)

        if abs denom < 1e-10 then None
        else
            let a = (n * (sumTxRx + sumTyRy) - sumTx * sumRx - sumTy * sumRy) / denom
            let b = (n * (sumTxRy - sumTyRx) - sumTx * sumRy + sumTy * sumRx) / denom // Corrected sign

            let dx = (sumRx - a * sumTx + b * sumTy) / n
            let dy = (sumRy - b * sumTx - a * sumTy) / n

            let scale = sqrt (a * a + b * b)
            let rotation = atan2 b a * 180.0 / Math.PI

            Log.Verbose("Transform: a={A:F4} b={B:F4} dx={Dx:F2} dy={Dy:F2} rot={Rot:F2}° scale={Scale:F4}",
                a, b, dx, dy, rotation, scale)

            Some (dx, dy, rotation, scale)

/// Run diagnostic analysis on a target image against reference
let runDiagnostic
    (refStars: DetectedStar[])
    (refTriangles: Triangle[])
    (targetStars: DetectedStar[])
    (fileName: string)
    (ratioTolerance: float)
    (maxStarsForTriangles: int)
    (minVotes: int)
    : AlignmentDiagnostics =

    let targetTriangles = formTriangles targetStars maxStarsForTriangles
    let matches = matchTriangles refTriangles targetTriangles ratioTolerance

    let correspondences = voteForCorrespondences matches
    let sortedVotes =
        correspondences
        |> Map.toArray
        |> Array.sortByDescending snd

    let topVoteCount =
        if sortedVotes.Length > 0 then snd sortedVotes.[0]
        else 0

    let transform = estimateTransform refStars targetStars sortedVotes minVotes

    let matchPercentage =
        if refTriangles.Length > 0 then
            (float matches.Length / float refTriangles.Length) * 100.0
        else 0.0

    {
        FileName = fileName
        DetectedStars = targetStars.Length
        TrianglesFormed = targetTriangles.Length
        TrianglesMatched = matches.Length
        MatchPercentage = matchPercentage
        TopVoteCount = topVoteCount
        EstimatedTransform = transform
    }
