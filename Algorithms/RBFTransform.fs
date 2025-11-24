module Algorithms.RBFTransform

open System

// --- Types ---

/// RBF kernel types
type RBFKernel =
    | ThinPlateSpline
    | Wendland of smoothness: int
    | InverseMultiquadric

/// RBF transform configuration
type RBFConfig = {
    Kernel: RBFKernel
    SupportRadiusFactor: float
    ShapeFactor: float
    Regularization: float
    MinControlPoints: int
    MaxControlPoints: int
}

/// Precomputed RBF coefficients
type RBFCoefficients = {
    ControlPoints: (float * float)[]
    WeightsX: float[]
    WeightsY: float[]
    PolyX: float[]
    PolyY: float[]
    Kernel: RBFKernel
    SupportRadius: float option
    ShapeParam: float option
    /// KD-tree for fast Wendland evaluation
    KdTree: SpatialMatch.KdNode option
}

/// Result of RBF setup
type RBFSetupResult = {
    Coefficients: RBFCoefficients option
    ResidualRMS: float
    MaxResidual: float
    ControlPointCount: int
}

let defaultRBFConfig = {
    Kernel = Wendland 1
    SupportRadiusFactor = 3.0
    ShapeFactor = 0.5
    Regularization = 1e-6
    MinControlPoints = 20
    MaxControlPoints = 500
}

// --- LU Decomposition Solver ---

/// LU decomposition with partial pivoting
/// Returns (L, U, P) where PA = LU
let private luDecompose (a: float[,]) : float[,] * float[,] * int[] =
    let n = Array2D.length1 a

    // Copy matrix
    let lu = Array2D.copy a
    let perm = [| 0 .. n - 1 |]

    for k in 0 .. n - 2 do
        // Find pivot
        let mutable maxVal = abs lu.[k, k]
        let mutable maxRow = k
        for i in k + 1 .. n - 1 do
            let v = abs lu.[i, k]
            if v > maxVal then
                maxVal <- v
                maxRow <- i

        // Swap rows
        if maxRow <> k then
            for j in 0 .. n - 1 do
                let temp = lu.[k, j]
                lu.[k, j] <- lu.[maxRow, j]
                lu.[maxRow, j] <- temp
            let tempP = perm.[k]
            perm.[k] <- perm.[maxRow]
            perm.[maxRow] <- tempP

        // Eliminate
        if abs lu.[k, k] > 1e-12 then
            for i in k + 1 .. n - 1 do
                lu.[i, k] <- lu.[i, k] / lu.[k, k]
                for j in k + 1 .. n - 1 do
                    lu.[i, j] <- lu.[i, j] - lu.[i, k] * lu.[k, j]

    // Extract L and U
    let l = Array2D.zeroCreate n n
    let u = Array2D.zeroCreate n n

    for i in 0 .. n - 1 do
        l.[i, i] <- 1.0
        for j in 0 .. i - 1 do
            l.[i, j] <- lu.[i, j]
        for j in i .. n - 1 do
            u.[i, j] <- lu.[i, j]

    (l, u, perm)

/// Forward substitution: solve Ly = b
let private forwardSubstitute (l: float[,]) (b: float[]) : float[] =
    let n = b.Length
    let y = Array.zeroCreate n

    for i in 0 .. n - 1 do
        let mutable sum = b.[i]
        for j in 0 .. i - 1 do
            sum <- sum - l.[i, j] * y.[j]
        y.[i] <- sum / l.[i, i]

    y

/// Back substitution: solve Ux = y
let private backSubstitute (u: float[,]) (y: float[]) : float[] =
    let n = y.Length
    let x = Array.zeroCreate n

    for i in n - 1 .. -1 .. 0 do
        let mutable sum = y.[i]
        for j in i + 1 .. n - 1 do
            sum <- sum - u.[i, j] * x.[j]
        if abs u.[i, i] > 1e-12 then
            x.[i] <- sum / u.[i, i]
        else
            x.[i] <- 0.0

    x

/// Solve linear system Ax = b using LU decomposition
let luSolve (a: float[,]) (b: float[]) : float[] =
    let n = Array2D.length1 a
    if n <> b.Length then
        failwith "Matrix and vector dimensions must match"

    let (l, u, perm) = luDecompose a

    // Apply permutation to b
    let pb = Array.init n (fun i -> b.[perm.[i]])

    // Solve LUx = Pb
    let y = forwardSubstitute l pb
    let x = backSubstitute u y

    x

/// Solve augmented RBF system for one coordinate
/// Returns (weights, polyCoeffs)
let solveAugmentedSystem
    (phi: float[,])
    (p: float[,])
    (d: float[])
    (lambda: float)
    : float[] * float[] =

    let n = Array2D.length1 phi
    let m = Array2D.length2 p
    let size = n + m

    // Build augmented matrix
    let augmented = Array2D.zeroCreate size size

    // Upper left: Φ + λI
    for i in 0 .. n - 1 do
        for j in 0 .. n - 1 do
            augmented.[i, j] <- phi.[i, j] + (if i = j then lambda else 0.0)

    // Upper right: P
    for i in 0 .. n - 1 do
        for j in 0 .. m - 1 do
            augmented.[i, n + j] <- p.[i, j]

    // Lower left: Pᵀ
    for i in 0 .. m - 1 do
        for j in 0 .. n - 1 do
            augmented.[n + i, j] <- p.[j, i]

    // Lower right: 0 (already zero)

    // Build RHS
    let rhs = Array.zeroCreate size
    for i in 0 .. n - 1 do
        rhs.[i] <- d.[i]

    // Solve
    let solution = luSolve augmented rhs

    // Extract weights and polynomial coefficients
    let w = solution.[0 .. n - 1]
    let c = solution.[n .. size - 1]
    (w, c)

// --- Kernel Functions ---

/// Evaluate TPS kernel: r² ln(r)
let private tpsKernel (r: float) : float =
    if r < 1e-10 then 0.0
    else r * r * log r

/// Evaluate Wendland ψ₃,₁ kernel (C²): (1-r)⁴(4r+1)
let private wendland1Kernel (r: float) (support: float) : float =
    let rNorm = r / support
    if rNorm >= 1.0 then 0.0
    else
        let t = 1.0 - rNorm
        t * t * t * t * (4.0 * rNorm + 1.0)

/// Evaluate Wendland ψ₃,₂ kernel (C⁴): (1-r)⁶(35r²+18r+3)
let private wendland2Kernel (r: float) (support: float) : float =
    let rNorm = r / support
    if rNorm >= 1.0 then 0.0
    else
        let t = 1.0 - rNorm
        let t2 = t * t
        let t6 = t2 * t2 * t2
        t6 * (35.0 * rNorm * rNorm + 18.0 * rNorm + 3.0)

/// Evaluate IMQ kernel: 1/√(r² + ε²)
let private imqKernel (r: float) (epsilon: float) : float =
    1.0 / sqrt (r * r + epsilon * epsilon)

/// Evaluate kernel based on type
let evaluateKernel (kernel: RBFKernel) (r: float) (support: float option) (epsilon: float option) : float =
    match kernel with
    | ThinPlateSpline -> tpsKernel r
    | Wendland smoothness ->
        let s = support |> Option.defaultValue 1.0
        match smoothness with
        | 1 -> wendland1Kernel r s
        | 2 -> wendland2Kernel r s
        | _ -> wendland1Kernel r s
    | InverseMultiquadric ->
        let e = epsilon |> Option.defaultValue 1.0
        imqKernel r e

// --- RBF System Setup ---

/// Compute median of array
let private median (values: float[]) : float =
    if values.Length = 0 then 0.0
    else
        let sorted = Array.sort values
        let mid = sorted.Length / 2
        if sorted.Length % 2 = 0 then
            (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else
            sorted.[mid]

/// Compute average spacing between control points
let private computeMedianSpacing (points: (float * float)[]) : float =
    if points.Length < 2 then 100.0
    else
        // Sample distances to nearest neighbors
        let distances = ResizeArray<float>()
        for i in 0 .. points.Length - 1 do
            let (xi, yi) = points.[i]
            let mutable minDist = Double.MaxValue
            for j in 0 .. points.Length - 1 do
                if i <> j then
                    let (xj, yj) = points.[j]
                    let d = sqrt ((xi - xj) ** 2.0 + (yi - yj) ** 2.0)
                    if d < minDist then minDist <- d
            if minDist < Double.MaxValue then
                distances.Add(minDist)

        if distances.Count > 0 then
            median (distances.ToArray())
        else
            100.0

/// Spatially subsample control points using grid-based selection
/// Ensures even distribution across the field
let private spatialSubsample
    (points: (float * float)[])
    (indices: int[])
    (maxPoints: int)
    (width: float)
    (height: float)
    : (float * float)[] * int[] =

    if points.Length <= maxPoints then
        (points, indices)
    else
        // Determine grid size to get approximately maxPoints
        let aspectRatio = width / height
        let gridY = int (sqrt (float maxPoints / aspectRatio))
        let gridX = int (float gridY * aspectRatio)
        let cellWidth = width / float gridX
        let cellHeight = height / float gridY

        // Group points by grid cell
        let cells = System.Collections.Generic.Dictionary<int * int, ResizeArray<int>>()

        for i in 0 .. points.Length - 1 do
            let (x, y) = points.[i]
            let cx = min (gridX - 1) (int (x / cellWidth))
            let cy = min (gridY - 1) (int (y / cellHeight))
            let key = (cx, cy)

            if not (cells.ContainsKey key) then
                cells.[key] <- ResizeArray<int>()
            cells.[key].Add(i)

        // Select one point from each cell (first one, which tends to be brighter due to earlier matching)
        let selected = ResizeArray<int>()
        for kvp in cells do
            if kvp.Value.Count > 0 then
                selected.Add(kvp.Value.[0])

        // If we still have too many, take evenly spaced subset
        let finalIndices =
            if selected.Count <= maxPoints then
                selected.ToArray()
            else
                let step = float selected.Count / float maxPoints
                [| for i in 0 .. maxPoints - 1 -> selected.[int (float i * step)] |]

        let sampledPoints = finalIndices |> Array.map (fun i -> points.[i])
        let sampledIndices = finalIndices |> Array.map (fun i -> indices.[i])

        (sampledPoints, sampledIndices)

/// Build RBF system matrices
let private buildSystem
    (controlPoints: (float * float)[])
    (kernel: RBFKernel)
    (support: float option)
    (epsilon: float option)
    : float[,] * float[,] =

    let n = controlPoints.Length

    // Build Φ matrix
    let phi = Array2D.init n n (fun i j ->
        let (xi, yi) = controlPoints.[i]
        let (xj, yj) = controlPoints.[j]
        let r = sqrt ((xi - xj) ** 2.0 + (yi - yj) ** 2.0)
        evaluateKernel kernel r support epsilon
    )

    // Build P matrix (affine: 1, x, y)
    let p = Array2D.init n 3 (fun i j ->
        let (x, y) = controlPoints.[i]
        match j with
        | 0 -> 1.0
        | 1 -> x
        | 2 -> y
        | _ -> 0.0
    )

    (phi, p)

/// Setup RBF from matched star pairs
let setupRBF
    (refStars: StarDetection.DetectedStar[])
    (targetStars: StarDetection.DetectedStar[])
    (matchedPairs: (int * int)[])
    (similarity: SpatialMatch.SimilarityTransform)
    (config: RBFConfig)
    (imageWidth: int)
    (imageHeight: int)
    : RBFSetupResult =

    if matchedPairs.Length < config.MinControlPoints then
        { Coefficients = None
          ResidualRMS = 0.0
          MaxResidual = 0.0
          ControlPointCount = matchedPairs.Length }
    else
        // Compute residuals after similarity transform
        let controlData =
            matchedPairs |> Array.map (fun (refIdx, targetIdx) ->
                let ref = refStars.[refIdx]
                let target = targetStars.[targetIdx]

                // Apply forward similarity transform to target
                let (tx, ty) = SpatialMatch.applyTransform similarity target.X target.Y

                // Residual = Reference - Transformed Target
                let dx = ref.X - tx
                let dy = ref.Y - ty

                // Control point is in transformed target space
                ((tx, ty), (dx, dy))
            )

        let allControlPoints = controlData |> Array.map fst
        let allDisplacements = controlData |> Array.map snd

        // Spatially subsample control points for performance
        let (controlPoints, sampledIndices) =
            spatialSubsample
                allControlPoints
                [| 0 .. allControlPoints.Length - 1 |]
                config.MaxControlPoints
                (float imageWidth)
                (float imageHeight)

        let displacements = sampledIndices |> Array.map (fun i -> allDisplacements.[i])

        // Compute kernel parameters
        let medianSpacing = computeMedianSpacing controlPoints
        let support =
            match config.Kernel with
            | Wendland _ -> Some (medianSpacing * config.SupportRadiusFactor)
            | _ -> None
        let epsilon =
            match config.Kernel with
            | InverseMultiquadric -> Some (medianSpacing * config.ShapeFactor)
            | _ -> None

        // Build system
        let (phi, p) = buildSystem controlPoints config.Kernel support epsilon

        // Solve for x and y displacements
        let dx = displacements |> Array.map fst
        let dy = displacements |> Array.map snd

        let (wx, cx) = solveAugmentedSystem phi p dx config.Regularization
        let (wy, cy) = solveAugmentedSystem phi p dy config.Regularization

        // Compute residual statistics on ALL points (not just sampled)
        // This shows how well the sampled RBF represents the full distortion
        let residuals =
            Array.init allControlPoints.Length (fun i ->
                let (x, y) = allControlPoints.[i]

                // Evaluate RBF at this point
                let mutable sumX = cx.[0] + cx.[1] * x + cx.[2] * y
                let mutable sumY = cy.[0] + cy.[1] * x + cy.[2] * y

                for j in 0 .. controlPoints.Length - 1 do
                    let (xj, yj) = controlPoints.[j]
                    let r = sqrt ((x - xj) ** 2.0 + (y - yj) ** 2.0)
                    let k = evaluateKernel config.Kernel r support epsilon
                    sumX <- sumX + wx.[j] * k
                    sumY <- sumY + wy.[j] * k

                // Error vs actual displacement
                let errX = (fst allDisplacements.[i]) - sumX
                let errY = (snd allDisplacements.[i]) - sumY
                sqrt (errX * errX + errY * errY)
            )

        let rms = sqrt (Array.averageBy (fun r -> r * r) residuals)
        let maxRes = Array.max residuals

        // Build KD-tree for Wendland optimization
        let kdTree =
            match config.Kernel with
            | Wendland _ ->
                // Create dummy stars for KD-tree (we just need positions)
                let starsForTree =
                    controlPoints
                    |> Array.indexed
                    |> Array.map (fun (i, (x, y)) ->
                        let dummyStar: StarDetection.DetectedStar = {
                            X = x; Y = y; FWHM = 0.0; HFR = 0.0
                            Peak = 0.0; Flux = 0.0; Background = 0.0
                            SNR = 0.0; Eccentricity = 0.0; Saturated = false
                        }
                        (dummyStar, i)
                    )
                Some (SpatialMatch.buildKdTree starsForTree)
            | _ -> None

        let coeffs = {
            ControlPoints = controlPoints
            WeightsX = wx
            WeightsY = wy
            PolyX = cx
            PolyY = cy
            Kernel = config.Kernel
            SupportRadius = support
            ShapeParam = epsilon
            KdTree = kdTree
        }

        { Coefficients = Some coeffs
          ResidualRMS = rms
          MaxResidual = maxRes
          ControlPointCount = controlPoints.Length }

// --- RBF Evaluation ---

/// Evaluate RBF displacement at a point
let evaluateRBF (coeffs: RBFCoefficients) (x: float) (y: float) : float * float =
    let mutable sumX = coeffs.PolyX.[0] + coeffs.PolyX.[1] * x + coeffs.PolyX.[2] * y
    let mutable sumY = coeffs.PolyY.[0] + coeffs.PolyY.[1] * x + coeffs.PolyY.[2] * y

    // Use KD-tree optimization for Wendland kernels
    match coeffs.KdTree, coeffs.SupportRadius with
    | Some tree, Some radius ->
        // Only sum contributions within support radius
        let neighbors = SpatialMatch.findWithinRadius tree x y radius
        for (_, idx, dist) in neighbors do
            let k = evaluateKernel coeffs.Kernel dist coeffs.SupportRadius coeffs.ShapeParam
            sumX <- sumX + coeffs.WeightsX.[idx] * k
            sumY <- sumY + coeffs.WeightsY.[idx] * k
    | _ ->
        // Fall back to full summation for TPS/IMQ
        for i in 0 .. coeffs.ControlPoints.Length - 1 do
            let (xi, yi) = coeffs.ControlPoints.[i]
            let r = sqrt ((x - xi) ** 2.0 + (y - yi) ** 2.0)
            let k = evaluateKernel coeffs.Kernel r coeffs.SupportRadius coeffs.ShapeParam
            sumX <- sumX + coeffs.WeightsX.[i] * k
            sumY <- sumY + coeffs.WeightsY.[i] * k

    (sumX, sumY)

/// Apply full inverse transform with Newton iteration
/// Maps reference coordinates to target coordinates
let applyFullInverseTransform
    (similarity: SpatialMatch.SimilarityTransform)
    (rbf: RBFCoefficients option)
    (refX: float)
    (refY: float)
    (maxIterations: int)
    (tolerance: float)
    : float * float =

    // Initial estimate: inverse similarity only
    let invSim = SpatialMatch.invertTransform similarity
    let (tx0, ty0) = SpatialMatch.applyTransform invSim refX refY

    match rbf with
    | None -> (tx0, ty0)
    | Some coeffs ->
        // Newton iteration to find t such that T(t) = ref
        let mutable tx = tx0
        let mutable ty = ty0
        let mutable converged = false

        for _ in 1 .. maxIterations do
            if not converged then
                // Forward transform
                let (simX, simY) = SpatialMatch.applyTransform similarity tx ty
                let (dx, dy) = evaluateRBF coeffs simX simY
                let fwdX = simX + dx
                let fwdY = simY + dy

                // Error
                let errX = refX - fwdX
                let errY = refY - fwdY

                if abs errX < tolerance && abs errY < tolerance then
                    converged <- true
                else
                    // Update estimate using inverse Jacobian (rotation + scale only, NO translation)
                    // The error is a displacement vector, not a position
                    let det = similarity.A * similarity.A + similarity.B * similarity.B
                    let aInv = similarity.A / det
                    let bInv = -similarity.B / det
                    let corrX = aInv * errX - bInv * errY
                    let corrY = bInv * errX + aInv * errY
                    tx <- tx + corrX
                    ty <- ty + corrY

        (tx, ty)
