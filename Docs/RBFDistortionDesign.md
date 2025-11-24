# RBF Distortion Correction Design

Technical design for adding RBF-based distortion correction to the alignment pipeline.

## Architecture Decision

### Hybrid Approach: Similarity + Residual RBF

Rather than replacing the similarity transform, use a two-stage model:

```
T(x) = T_similarity(x) + T_rbf(x)
```

**Rationale**:
1. Similarity transform handles bulk rotation, scale, translation (4 DOF)
2. RBF corrects residual field distortion after global alignment
3. Better conditioning - RBF only models small residuals
4. Fallback gracefully - if RBF fails, similarity transform still works

### Data Flow

```
Reference Stars ─┐
                 ├─→ Match ─→ Similarity Transform ─→ Apply to Target Positions
Target Stars ────┘                                            │
                                                              ▼
                                           Compute Residuals (Ref - Transformed Target)
                                                              │
                                                              ▼
                                                     Build RBF Model
                                                              │
                                                              ▼
                                              Combined Transform for Resampling
```

## Module Design

### New File: `Algorithms/RBFTransform.fs`

```fsharp
module Algorithms.RBFTransform

open System

/// RBF kernel types
type RBFKernel =
    | ThinPlateSpline           // r² ln(r) - classic, global influence
    | Wendland of smoothness:int // Compact support, local influence
    | InverseMultiquadric       // 1/√(r² + ε²) - good middle ground

/// RBF transform configuration
type RBFConfig = {
    Kernel: RBFKernel
    /// Support radius for Wendland (relative to median spacing)
    SupportRadiusFactor: float
    /// Shape parameter for IMQ (relative to median spacing)
    ShapeFactor: float
    /// Regularization parameter (0 = exact interpolation)
    Regularization: float
    /// Minimum matched pairs required
    MinControlPoints: int
}

/// Precomputed RBF coefficients for fast evaluation
type RBFCoefficients = {
    /// Control point positions (in target space after similarity)
    ControlPoints: (float * float)[]
    /// Weight vectors for x and y displacement
    WeightsX: float[]
    WeightsY: float[]
    /// Polynomial coefficients [c0, c1x, c2y] for x and y
    PolyX: float[]
    PolyY: float[]
    /// Kernel parameters
    Kernel: RBFKernel
    SupportRadius: float option  // For Wendland
    ShapeParam: float option     // For IMQ
}

/// Result of RBF setup
type RBFSetupResult = {
    Coefficients: RBFCoefficients option
    ResidualRMS: float
    MaxResidual: float
    ControlPointCount: int
}
```

### Core Functions

```fsharp
/// Compute RBF coefficients from matched star pairs and similarity transform
val setupRBF :
    refStars: DetectedStar[] ->
    targetStars: DetectedStar[] ->
    matchedPairs: (int * int)[] ->
    similarityTransform: SimilarityTransform ->
    config: RBFConfig ->
    RBFSetupResult

/// Evaluate RBF displacement at a point
val evaluateRBF :
    coefficients: RBFCoefficients ->
    x: float ->
    y: float ->
    (float * float)

/// Combined transform: inverse similarity + RBF correction
val applyFullTransform :
    similarity: SimilarityTransform ->
    rbf: RBFCoefficients option ->
    x: float ->
    y: float ->
    (float * float)
```

## Implementation Details

### 1. Residual Computation

After matching and similarity transform estimation:

```fsharp
let computeResiduals
    (refStars: DetectedStar[])
    (targetStars: DetectedStar[])
    (matchedPairs: (int * int)[])
    (transform: SimilarityTransform) =

    matchedPairs |> Array.map (fun (refIdx, targetIdx) ->
        let ref = refStars.[refIdx]
        let target = targetStars.[targetIdx]

        // Apply forward transform to target
        let (tx, ty) = applyTransform transform target.X target.Y

        // Residual = Reference - Transformed Target
        let dx = ref.X - tx
        let dy = ref.Y - ty

        // Control point is in target space (after similarity transform)
        ((tx, ty), (dx, dy))
    )
```

### 2. RBF System Setup

Build and solve the augmented linear system:

```fsharp
let buildRBFSystem (controlPoints: (float * float)[]) (displacements: (float * float)[]) kernel =
    let n = controlPoints.Length

    // Build Φ matrix (n × n)
    let phi = Array2D.init n n (fun i j ->
        let (xi, yi) = controlPoints.[i]
        let (xj, yj) = controlPoints.[j]
        let r = sqrt ((xi - xj) ** 2.0 + (yi - yj) ** 2.0)
        evaluateKernel kernel r
    )

    // Build P matrix (n × 3) for affine polynomial
    let p = Array2D.init n 3 (fun i j ->
        let (x, y) = controlPoints.[i]
        match j with
        | 0 -> 1.0
        | 1 -> x
        | 2 -> y
        | _ -> 0.0
    )

    // Augmented system:
    // [Φ + λI  P ] [w] = [d]
    // [Pᵀ      0 ] [c]   [0]

    // Solve for x and y displacements separately
    let (wx, cx) = solveAugmentedSystem phi p (Array.map fst displacements) regularization
    let (wy, cy) = solveAugmentedSystem phi p (Array.map snd displacements) regularization

    (wx, wy, cx, cy)
```

### 3. Kernel Implementations

```fsharp
let evaluateKernel kernel r =
    match kernel with
    | ThinPlateSpline ->
        if r < 1e-10 then 0.0
        else r * r * log r

    | Wendland smoothness ->
        let rNorm = r / supportRadius
        if rNorm >= 1.0 then 0.0
        else
            match smoothness with
            | 1 -> // C² - ψ₃,₁
                let t = 1.0 - rNorm
                t ** 4.0 * (4.0 * rNorm + 1.0)
            | 2 -> // C⁴ - ψ₃,₂
                let t = 1.0 - rNorm
                t ** 6.0 * (35.0 * rNorm * rNorm + 18.0 * rNorm + 3.0)
            | _ -> failwith "Unsupported Wendland smoothness"

    | InverseMultiquadric ->
        1.0 / sqrt (r * r + epsilon * epsilon)
```

### 4. Linear System Solver

For the augmented system, use LU decomposition:

```fsharp
let solveAugmentedSystem phi p d lambda =
    let n = Array2D.length1 phi
    let m = Array2D.length2 p  // 3 for affine
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
    // Last m entries are 0

    // Solve using LU decomposition
    let solution = luSolve augmented rhs

    // Extract weights and polynomial coefficients
    let w = solution.[0 .. n - 1]
    let c = solution.[n .. size - 1]
    (w, c)
```

### 5. Combined Transform Evaluation

For resampling, we need inverse mapping (reference → target):

```fsharp
let applyFullInverseTransform
    (similarity: SimilarityTransform)
    (rbf: RBFCoefficients option)
    (refX: float)
    (refY: float) =

    // First: inverse similarity transform
    let invSim = invertTransform similarity
    let (tx, ty) = applyTransform invSim refX refY

    // Then: apply RBF correction (if available)
    match rbf with
    | None -> (tx, ty)
    | Some coeffs ->
        // Note: RBF was built in forward direction (target→ref)
        // For inverse, we need to find point that maps to (refX, refY)
        // Use iterative refinement
        let (dx, dy) = evaluateRBF coeffs tx ty
        (tx - dx, ty - dy)  // Approximate inverse
```

**Important**: The RBF inverse is approximate. For small distortions this works well. For accuracy, could add Newton iteration.

## Integration with Align.fs

### New Options

```fsharp
type DistortionCorrection =
    | None          // Similarity only (current behavior)
    | TPS           // Thin-plate spline
    | Wendland      // Compact support (recommended)
    | IMQ           // Inverse multiquadric

type AlignOptions = {
    // ... existing fields ...

    // Distortion correction
    DistortionCorrection: DistortionCorrection
    RBFRegularization: float
    WendlandSupportFactor: float  // Default 3.0
}
```

### Modified processAlignFile

```fsharp
let processAlignFile ... =
    async {
        // ... existing: read, detect stars, match ...

        // Get similarity transform
        let transform = result.Transform.Value

        // Optionally compute RBF correction
        let rbfCoeffs =
            match opts.DistortionCorrection with
            | None -> None
            | _ ->
                let config = {
                    Kernel = match opts.DistortionCorrection with
                             | TPS -> ThinPlateSpline
                             | Wendland -> Wendland 1
                             | IMQ -> InverseMultiquadric
                             | None -> failwith "unreachable"
                    SupportRadiusFactor = opts.WendlandSupportFactor
                    Regularization = opts.RBFRegularization
                    MinControlPoints = 10
                }

                let rbfResult = setupRBF refStars targetStars matchedPairs transform config
                rbfResult.Coefficients

        // Transform with combined model
        let transformedPixels =
            transformImageWithDistortion
                pixelFloats width height channels
                transform rbfCoeffs
                opts.Interpolation

        // ... write output ...
    }
```

## Performance Considerations

### Wendland Optimization

For Wendland kernels, use spatial indexing to skip zero contributions:

```fsharp
let evaluateRBFWendland (coeffs: RBFCoefficients) (kdTree: KdNode) (x: float) (y: float) =
    let supportRadius = coeffs.SupportRadius.Value

    // Only sum contributions from points within support
    let neighbors = findWithinRadius kdTree x y supportRadius

    let mutable sumX = coeffs.PolyX.[0] + coeffs.PolyX.[1] * x + coeffs.PolyX.[2] * y
    let mutable sumY = coeffs.PolyY.[0] + coeffs.PolyY.[1] * x + coeffs.PolyY.[2] * y

    for (_, idx, dist) in neighbors do
        let phi = wendlandKernel dist supportRadius
        sumX <- sumX + coeffs.WeightsX.[idx] * phi
        sumY <- sumY + coeffs.WeightsY.[idx] * phi

    (sumX, sumY)
```

This reduces per-pixel evaluation from O(n) to O(k) where k << n.

### Sparse Matrix for Wendland

For system setup, only compute non-zero entries:

```fsharp
// Use Dictionary for sparse representation
let sparsePhiRow i =
    let (xi, yi) = controlPoints.[i]
    let entries = Dictionary<int, float>()

    for j in 0 .. n - 1 do
        let (xj, yj) = controlPoints.[j]
        let r = sqrt ((xi - xj) ** 2.0 + (yi - yj) ** 2.0)
        if r < supportRadius then
            entries.[j] <- wendlandKernel r supportRadius

    entries
```

For 1000 stars with k=50 neighbors average: 50,000 entries vs 1,000,000 for dense.

## Quality Metrics

### Residual Statistics

After RBF setup, report:
- RMS residual (should be near zero for exact interpolation)
- Maximum residual
- Number of control points used

### Jacobian Monitoring

Sample Jacobian determinant across image to verify topology preservation:

```fsharp
let checkTopology (transform: SimilarityTransform) (rbf: RBFCoefficients option) width height =
    let samples = 100
    let mutable minDet = Double.MaxValue
    let mutable violations = 0

    for i in 0 .. samples - 1 do
        for j in 0 .. samples - 1 do
            let x = float i / float samples * float width
            let y = float j / float samples * float height

            // Numerical Jacobian
            let h = 0.5
            let (x1, y1) = applyFullInverse transform rbf (x - h) y
            let (x2, y2) = applyFullInverse transform rbf (x + h) y
            let (x3, y3) = applyFullInverse transform rbf x (y - h)
            let (x4, y4) = applyFullInverse transform rbf x (y + h)

            let dxdx = (x2 - x1) / (2.0 * h)
            let dxdy = (x4 - x3) / (2.0 * h)
            let dydx = (y2 - y1) / (2.0 * h)
            let dydy = (y4 - y3) / (2.0 * h)

            let det = dxdx * dydy - dxdy * dydx
            minDet <- min minDet det
            if det <= 0.0 then violations <- violations + 1

    (minDet, violations)
```

## Default Configuration

Based on your research recommendations:

```fsharp
let defaultRBFConfig = {
    Kernel = Wendland 1           // ψ₃,₁ for C² smoothness
    SupportRadiusFactor = 3.0     // 3× median star spacing
    ShapeFactor = 0.5             // For IMQ fallback
    Regularization = 1e-6         // Small regularization for stability
    MinControlPoints = 20
}
```

## CLI Arguments

```
Distortion Correction:
  --distortion <type>       Correction method (default: none)
                              none     - Similarity transform only
                              wendland - Local RBF correction (recommended)
                              tps      - Thin-plate spline (global)
                              imq      - Inverse multiquadric
  --rbf-support <factor>    Wendland support radius factor (default: 3.0)
  --rbf-regularization <v>  Regularization parameter (default: 1e-6)
```

## Output Headers

```fsharp
let createDistortionHeaders (rbfResult: RBFSetupResult) =
    [|
        XisfFitsKeyword("DISTCORR", "WENDLAND", "Distortion correction method")
        XisfFitsKeyword("RBFPTS", rbfResult.ControlPointCount.ToString(), "RBF control points")
        XisfFitsKeyword("RBFRMS", sprintf "%.3f" rbfResult.ResidualRMS, "RBF residual RMS (px)")
        XisfFitsKeyword("RBFMAX", sprintf "%.3f" rbfResult.MaxResidual, "RBF max residual (px)")
    |]
```

## Implementation Order

1. **RBF core** (`Algorithms/RBFTransform.fs`)
   - Kernel functions
   - System builder
   - LU solver (or use external library)
   - Coefficient evaluation

2. **Wendland optimization**
   - Leverage existing KD-tree from SpatialMatch.fs
   - Sparse evaluation

3. **Integration with Align.fs**
   - New options
   - Modified transform pipeline
   - Headers

4. **Testing**
   - Verify residuals go to zero
   - Check topology preservation
   - Compare TPS vs Wendland vs IMQ quality

## Design Decisions

1. **LU Solver**: Native F# implementation for now. Sufficient for typical star counts (< 1000). Can optimize later if needed.

2. **Inverse accuracy**: Use Newton iteration for accurate RBF inverse - important for larger distortions.

3. **Visualization mode**: Add `--output-mode distortion` to visualize the distortion field.

4. **Fallback behavior**: Fall back to similarity-only with warning if RBF setup fails.

---

## Newton Iteration for RBF Inverse

The RBF maps target → reference. For resampling we need reference → target. Use iterative refinement:

```fsharp
/// Accurate inverse via Newton iteration
let applyFullInverseTransform
    (similarity: SimilarityTransform)
    (rbf: RBFCoefficients option)
    (refX: float)
    (refY: float)
    (maxIterations: int)
    (tolerance: float) =

    // Initial estimate: inverse similarity only
    let invSim = invertTransform similarity
    let (tx0, ty0) = applyTransform invSim refX refY

    match rbf with
    | None -> (tx0, ty0)
    | Some coeffs ->
        // Newton iteration to find t such that T(t) = ref
        let mutable tx = tx0
        let mutable ty = ty0

        for _ in 1 .. maxIterations do
            // Forward transform
            let (simX, simY) = applyTransform similarity tx ty
            let (dx, dy) = evaluateRBF coeffs simX simY
            let (fwdX, fwdY) = (simX + dx, simY + dy)

            // Error
            let errX = refX - fwdX
            let errY = refY - fwdY

            if abs errX < tolerance && abs errY < tolerance then
                ()  // Converged
            else
                // Update (simplified - assumes Jacobian ≈ I for small distortions)
                let (corrX, corrY) = applyTransform invSim errX errY
                tx <- tx + corrX
                ty <- ty + corrY

        (tx, ty)
```

Default: `maxIterations = 5`, `tolerance = 0.01` pixels.

---

## Distortion Visualization Mode

New output mode to visualize the distortion field:

```fsharp
type OutputMode =
    | Detect      // Show detected stars only
    | Match       // Show matched star correspondences
    | Align       // Apply transformation (default)
    | Distortion  // Visualize distortion field
```

### Visualization Options

**Vector field**: Arrows showing displacement direction/magnitude
- Draw arrows at grid points (e.g., every 100 pixels)
- Arrow length proportional to displacement
- Color-coded by magnitude

**Magnitude map**: Heatmap of distortion magnitude
- Compute |displacement| at each pixel
- Map to color scale (blue → red)
- Overlay control points as markers

### Implementation

```fsharp
let processDistortionFile
    (inputPath: string) (outputDir: string) ...
    (transform: SimilarityTransform) (rbf: RBFCoefficients option) =
    async {
        // Create output image
        let pixels = createBlackPixels width height 3 XisfSampleFormat.UInt16

        // Draw magnitude heatmap
        for y in 0 .. height - 1 do
            for x in 0 .. width - 1 do
                let (tx, ty) = applyFullInverseTransform transform rbf (float x) (float y) 5 0.01
                let dx = float x - tx  // Displacement in reference frame
                let dy = float y - ty
                let mag = sqrt (dx * dx + dy * dy)

                // Map magnitude to color (0-10 pixels → blue-red)
                let (r, g, b) = magnitudeToColor mag 10.0
                let idx = (y * width + x) * 3
                pixels.[idx] <- r
                pixels.[idx + 1] <- g
                pixels.[idx + 2] <- b

        // Overlay control points as white dots
        for (cx, cy) in controlPoints do
            paintCircle pixels width height 3 0 cx cy 3.0 1.0 format
            paintCircle pixels width height 3 1 cx cy 3.0 1.0 format
            paintCircle pixels width height 3 2 cx cy 3.0 1.0 format

        // Draw vector arrows at grid
        let gridSpacing = 100
        for gy in 0 .. height / gridSpacing do
            for gx in 0 .. width / gridSpacing do
                let x = float (gx * gridSpacing)
                let y = float (gy * gridSpacing)
                let (tx, ty) = applyFullInverseTransform transform rbf x y 5 0.01
                let dx = x - tx
                let dy = y - ty
                drawArrow pixels width height x y (x + dx * 10.0) (y + dy * 10.0)

        // ... write output ...
    }

let magnitudeToColor (mag: float) (maxMag: float) =
    let t = min 1.0 (mag / maxMag)
    // Blue (0,0,1) → Cyan (0,1,1) → Green (0,1,0) → Yellow (1,1,0) → Red (1,0,0)
    let (r, g, b) =
        if t < 0.25 then
            (0.0, t * 4.0, 1.0)
        elif t < 0.5 then
            (0.0, 1.0, 1.0 - (t - 0.25) * 4.0)
        elif t < 0.75 then
            ((t - 0.5) * 4.0, 1.0, 0.0)
        else
            (1.0, 1.0 - (t - 0.75) * 4.0, 0.0)
    (uint16 (r * 65535.0), uint16 (g * 65535.0), uint16 (b * 65535.0))
```

### Output Headers

```fsharp
let createDistortionVisHeaders (transform: SimilarityTransform) (rbfResult: RBFSetupResult) =
    [|
        XisfFitsKeyword("IMAGETYP", "DISTMAP", "Distortion visualization")
        XisfFitsKeyword("DISTCORR", kernelName, "Correction method")
        XisfFitsKeyword("RBFPTS", rbfResult.ControlPointCount.ToString(), "Control points")
        XisfFitsKeyword("MAXDIST", sprintf "%.2f" maxDistortion, "Max distortion (px)")
        XisfFitsKeyword("AVGDIST", sprintf "%.2f" avgDistortion, "Avg distortion (px)")
    |]
```

---

## Updated CLI Arguments

```
Output Mode:
  --output-mode <mode>      Output type (default: align)
                              detect     - Show detected stars only
                              match      - Show matched star correspondences
                              align      - Apply transformation (default)
                              distortion - Visualize distortion field

Distortion Correction:
  --distortion <type>       Correction method (default: none)
                              none     - Similarity transform only
                              wendland - Local RBF correction (recommended)
                              tps      - Thin-plate spline (global)
                              imq      - Inverse multiquadric
  --rbf-support <factor>    Wendland support radius factor (default: 3.0)
  --rbf-regularization <v>  Regularization parameter (default: 1e-6)
```

---

## Updated Implementation Order

1. **Native LU solver** - Basic implementation for augmented system
2. **RBF core** (`Algorithms/RBFTransform.fs`)
   - Kernel functions (TPS, Wendland, IMQ)
   - System builder with polynomial constraints
   - Coefficient evaluation with Newton iteration
3. **Wendland optimization** - KD-tree sparse evaluation
4. **Integration with Align.fs**
   - New options and enums
   - Modified transform pipeline with fallback
   - Updated headers
5. **Distortion visualization mode**
   - Magnitude heatmap
   - Vector field overlay
   - Control point markers
6. **Testing**
   - Verify residuals → 0
   - Newton convergence
   - Topology preservation
   - Visual inspection of distortion maps
