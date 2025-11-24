# RBF Distortion Correction Implementation

Implementation decisions and architecture for the RBF-based distortion correction feature.

## Architecture

### Hybrid Transform Model

We use a two-stage transformation model rather than replacing the similarity transform:

```
T(x) = T_similarity(x) + T_rbf(x)
```

**Rationale**:
- Similarity transform handles bulk rotation, scale, translation (4 DOF)
- RBF corrects only residual local distortion after global alignment
- Better numerical conditioning since RBF models small residuals
- Graceful fallback - if RBF fails, similarity transform still works

### Module Structure

- `Algorithms/RBFTransform.fs` - Core RBF implementation
- `Commands/Align.fs` - Integration with alignment pipeline

## Key Implementation Decisions

### 1. Kernel Selection

**Default: Wendland ψ₃,₁** (compact support)

Based on research in `Docs/RBF.md`, Wendland was chosen for:
- Local influence prevents distant distortions
- Sparse matrices for computational efficiency
- Good topology preservation
- Scales well to dense star fields

Other supported kernels:
- TPS (Thin-Plate Spline) - global, classic
- IMQ (Inverse Multiquadric) - global with decay

### 2. Native LU Solver

Implemented native F# LU decomposition with partial pivoting rather than using external libraries.

**Rationale**:
- No external dependencies
- Sufficient for typical control point counts (< 1000)
- Can optimize later if needed

Location: `RBFTransform.fs` lines 56-191

### 3. Newton Iteration for Inverse

The RBF maps target → reference. For resampling we need reference → target. We use Newton iteration rather than analytical inverse.

**Parameters**:
- Max iterations: 5
- Tolerance: 0.01 pixels

**Rationale**:
- RBF inverse is not analytical
- Newton converges quickly for small distortions
- Single-step approximation would lose accuracy for larger distortions

Location: `RBFTransform.fs` function `applyFullInverseTransform`

### 4. Spatial Subsampling of Control Points

**Default: 500 control points maximum**

With ~5000 matched stars, per-pixel RBF evaluation was 19x slower than baseline. We subsample using grid-based spatial distribution.

**Algorithm**:
1. Divide image into grid cells (aspect-ratio aware)
2. Select one point from each cell
3. Ensures even coverage across field

**Rationale**:
- Maintains spatial representation of distortion field
- Reduces evaluation from O(5000) to O(500) per pixel
- Grid-based selection prevents clustering in bright regions

Location: `RBFTransform.fs` function `spatialSubsample`

### 5. KD-Tree Optimization for Wendland

For Wendland kernels (compact support), we use a KD-tree to find only nearby control points within the support radius.

**Rationale**:
- Wendland kernel is zero outside support radius
- No need to sum all 500 control points
- Reduces evaluation from O(n) to O(k) where k ≈ 20-50

Location: Uses existing `SpatialMatch.buildKdTree` and `findWithinRadius`

### 6. Boundary Handling

**Decision: Black fill (0) for out-of-bounds pixels**

When the inverse transform maps to coordinates outside the source image, we return 0.

**Rationale**:
- Standard for astronomical imaging
- Edge extension would introduce artifacts
- Clear visual indication of transformed boundaries

### 7. Output Format

**Decision: Float32 output for aligned images**

Interpolated pixel values are stored as Float32 regardless of input format.

**Rationale**:
- Preserves interpolation precision
- Avoids quantization artifacts
- Normalized to [0, 1] range

### 8. Fallback Behavior

**Decision: Fall back to similarity-only with warning**

If RBF setup fails (too few points, singular matrix), we proceed with just the similarity transform.

**Rationale**:
- Better to produce output than fail completely
- User is warned about the fallback
- Similarity transform alone is often sufficient

## Configuration Parameters

### RBFConfig Record

| Field | Default | Description |
|-------|---------|-------------|
| Kernel | Wendland 1 | RBF kernel type |
| SupportRadiusFactor | 3.0 | Wendland support = factor × median spacing |
| ShapeFactor | 0.5 | IMQ shape parameter factor |
| Regularization | 1e-6 | Small regularization for stability |
| MinControlPoints | 20 | Minimum required for RBF |
| MaxControlPoints | 500 | Maximum (spatially sampled) |

### CLI Arguments

```
--distortion <type>       none, wendland, tps, imq
--rbf-support <factor>    Wendland support radius factor
--rbf-regularization <v>  Regularization parameter
```

## Distortion Visualization Mode

Added `--output-mode distortion` to visualize the distortion field.

**Output**:
- RGB heatmap of distortion magnitude
- Color scale: blue (0px) → cyan → green → yellow → red (10px)
- White circles marking control point locations

**Use cases**:
- Verify RBF is modeling actual distortion
- Compare different kernels
- Diagnose alignment issues

Location: `Align.fs` function `processDistortionFile`

## Performance Characteristics

### Baseline (similarity only)
- 25 images × 26 megapixels: ~108 seconds

### With RBF (500 control points)
- Additional overhead for RBF evaluation
- Newton iteration (5 iterations per pixel)
- Expected: 2-3x baseline

### Optimizations Applied
1. Spatial subsampling (5000 → 500 points)
2. KD-tree for Wendland evaluation
3. Parallel pixel processing via `Array.Parallel`

## Future Considerations

### Potential Optimizations

1. **Grid-based caching**: Precompute transform on coarse grid, interpolate for pixels
2. **Reduced Newton iterations**: 2-3 may suffice for typical distortions
3. **SIMD**: Vectorize kernel evaluation

### Possible Extensions

1. **CLI for MaxControlPoints**: Allow user to tune performance/quality tradeoff
2. **Quality metrics**: Report Jacobian statistics for topology verification
3. **Alternative subsampling**: Farthest-point sampling for more uniform distribution

## References

- Mathematical foundations: `Docs/RBF.md`
- Original design: `Docs/RBFDistortionDesign.md`
- Test commands: `Docs/QuickRef.md`
