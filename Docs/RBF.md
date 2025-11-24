# Radial Basis Function Image Warping: Mathematical Foundation

## Overview

This document details the mathematical foundations for implementing RBF-based image warping for astrophotography registration, specifically for correcting arbitrary geometric distortions using star positions detected via triangle similarity matching.

**Target Implementation**: F#  
**Use Case**: Astrophotography image alignment with detected star control points  
**Goal**: Smooth, artifact-free geometric transformation preserving topology

## Fundamental Concept

Radial Basis Functions provide a method for interpolating scattered data in multidimensional space. For image warping, we use RBFs to construct smooth transformation functions that map source image coordinates to target image coordinates based on sparse control point correspondences.

### Problem Statement

Given:
- Source control points: **s** = {**s**₁, **s**₂, ..., **sₙ} ∈ ℝ²
- Target control points: **t** = {**t**₁, **t**₂, ..., **tₙ} ∈ ℝ²
- Displacement vectors: **d**ᵢ = **t**ᵢ - **s**ᵢ

Find: Smooth transformation function **T**: ℝ² → ℝ² such that **T**(**s**ᵢ) = **t**ᵢ for all i

### Solution Form

The transformation is represented as:

**T**(**x**) = **P**(**x**) + Σᵢ₌₁ⁿ **w**ᵢ · φ(‖**x** - **s**ᵢ‖)

Where:
- φ(r) is the radial basis function
- **w**ᵢ are weight vectors in ℝ² (computed during setup)
- **P**(**x**) is a low-degree polynomial term (typically affine)
- r = ‖**x** - **s**ᵢ‖ is the Euclidean distance

## The Four Basis Functions

### 1. Multiquadric (MQ)

**Function Definition**:
```
φ(r) = √(r² + ε²)
```

**Properties**:
- Conditionally positive definite of order 1
- Global support (non-zero everywhere)
- Increases monotonically with distance
- **ε** is the shape parameter (typically 0.5 to 5.0 times average point spacing)

**Characteristics**:
- Excellent for capturing global deformation trends
- Smooth interpolation with good extrapolation behavior
- Higher accuracy with fewer control points than TPS
- Requires polynomial term of at least degree 0 (constant)

**Optimal For**: Large-scale field distortions, global geometric corrections

**Implementation Notes**:
- Guard against overflow with large r values
- Shape parameter ε critically affects smoothness vs. fidelity
- Full matrix solution required (O(n³) complexity)

**References**:
- Hardy, R.L. (1971). "Multiquadric equations of topography and other irregular surfaces"
- Carlson & Foley (1991). "The parameter R² in multiquadric interpolation"
- https://www.researchgate.net/publication/263732164_Image_warping_using_radial_basis_functions

---

### 2. Inverse Multiquadric (IMQ)

**Function Definition**:
```
φ(r) = 1 / √(r² + ε²)
```

**Properties**:
- Strictly positive definite
- Global support with rapid decay
- Decreases monotonically with distance
- Well-conditioned interpolation matrices
- **ε** controls locality (smaller = more local influence)

**Characteristics**:
- Pronounced local effect - nearby points dominate
- More stable numerically than MQ
- Smoother results with less overshoot
- No polynomial term required (but often beneficial)

**Optimal For**: Localized corrections, dense control point distributions

**Implementation Notes**:
- Never singular (ε² prevents division by zero at control points)
- More forgiving shape parameter selection than MQ
- Better conditioning than MQ for large systems

**References**:
- Franke, R. (1982). "Scattered data interpolation: Tests of some methods"
- https://www.tutorialspoint.com/scipy/scipy_rbf_multi_dimensional_interpolation.htm

---

### 3. Thin-Plate Spline (TPS)

**Function Definition** (2D):
```
φ(r) = r² · ln(r)    for r > 0
φ(0) = 0             at control points
```

**Properties**:
- Conditionally positive definite of order 2
- Minimizes bending energy integral: ∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dxdy
- Global support with slow decay (O(r² log r))
- Requires polynomial term of degree 1 (affine: a₀ + a₁x + a₂y)

**Characteristics**:
- "Natural" interpolant - minimal curvature solution
- Each control point affects entire domain
- Well-established theoretical foundation
- Can produce large displacements far from control points

**Optimal For**: Standard image registration tasks, well-distributed control points

**Drawbacks**:
- Global influence can cause unwanted distant distortions
- Computational cost O(n³) for n control points
- May introduce undesired oscillations between sparse points

**Implementation Notes**:
- Special handling at r=0 (use limit: lim[r→0] r² ln(r) = 0)
- Augmented system with polynomial constraints:
  ```
  Σᵢ wᵢ = 0
  Σᵢ wᵢ·xᵢ = 0
  Σᵢ wᵢ·yᵢ = 0
  ```
- System size: (n+3) × (n+3) for 2D

**References**:
- Bookstein, F.L. (1989). "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations"
- Duchon, J. (1977). "Splines minimizing rotation-invariant semi-norms in Sobolev spaces"
- https://en.wikipedia.org/wiki/Thin_plate_spline

---

### 4. Wendland Functions (Compactly Supported)

**Function Family** (various smoothness levels):

**Wendland ψ₃,₁** (C² continuous):
```
For r̃ = r/c where c is support radius:
φ(r) = (1 - r̃)⁴₊ · (4r̃ + 1)    for r ≤ c
φ(r) = 0                         for r > c

Where (·)₊ = max(·, 0)
```

**Wendland ψ₃,₂** (C⁴ continuous):
```
φ(r) = (1 - r̃)⁶₊ · (35r̃² + 18r̃ + 3)    for r ≤ c
φ(r) = 0                                  for r > c
```

**Properties**:
- Compactly supported: φ(r) = 0 for r > c
- Positive definite in ℝ³ (ψ₃,ₖ)
- Prescribed smoothness levels
- Sparse interpolation matrices

**Characteristics**:
- **Computational Efficiency**: Sparse matrices reduce complexity from O(n³) to O(n·k²) where k << n
- **Locality Control**: Support radius c directly controls influence region
- **Numerical Stability**: Better conditioning than global RBFs
- **Scalability**: Handles large control point sets efficiently

**Optimal For**: Dense star fields, large-scale astrophotography registration, real-time applications

**Implementation Notes**:
- Support radius c typically 2-4× average point spacing
- Use spatial indexing (k-d trees) for efficient neighbor queries
- Matrix assembly only includes points within support radius
- Multiple smoothness levels available (trade-off: smoothness vs. locality)

**References**:
- Wendland, H. (1995). "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree"
- Fornefett, M. et al. (2001). "Radial basis functions with compact support for elastic registration of medical images"
- https://link.springer.com/article/10.1023/A:1018916029914

---

## Mathematical Framework

### System Construction

For n control point pairs, construct the interpolation system:

**Matrix Form**:
```
[Φ  P] [w] = [d]
[Pᵀ 0] [c]   [0]
```

Where:
- **Φ** is n×n matrix: Φᵢⱼ = φ(‖**s**ᵢ - **s**ⱼ‖)
- **P** is n×m matrix of polynomial basis evaluated at control points
- **w** are n×2 weight vectors (one for x, one for y displacement)
- **c** are m×2 polynomial coefficients
- **d** are n×2 displacement vectors

### Polynomial Terms

**For 2D (affine transformation)**:
```
P = [1  x  y]    (m = 3 basis functions)
```

**Polynomial constraints** (ensure unique solution):
```
Pᵀw = 0
```

This forces the RBF component to be orthogonal to the polynomial space.

### Solution Process

1. **Compute distance matrix**: Calculate all pairwise distances between control points
2. **Evaluate basis function**: Apply φ(·) to create Φ matrix
3. **Augment with polynomials**: Add polynomial terms and constraints
4. **Solve linear system**: Use LU decomposition or iterative solver
5. **Store coefficients**: Keep w and c for transformation evaluation

### Transformation Evaluation

For arbitrary image coordinate **x** = (x, y):

1. **Compute distances** to all control points: rᵢ = ‖**x** - **s**ᵢ‖
2. **Evaluate RBF terms**: φ(rᵢ) for each control point
3. **Compute displacement**: 
   ```
   Δx = c₀ + c₁x + c₂y + Σᵢ wᵢₓ·φ(rᵢ)
   Δy = c₃ + c₄x + c₅y + Σᵢ wᵢᵧ·φ(rᵢ)
   ```
4. **Apply transformation**: **T**(**x**) = **x** + (Δx, Δy)

---

## Shape Parameter Selection

The shape parameter ε (for MQ/IMQ) or support radius c (for Wendland) critically affects interpolation quality.

### Guidelines

**Multiquadric ε**:
- Start with: ε = 2 × d_avg (average control point spacing)
- Smaller ε → sharper, more localized deformation
- Larger ε → smoother, more global deformation
- Optimal range: 0.5 to 5.0 × d_avg

**Inverse Multiquadric ε**:
- Start with: ε = 0.5 × d_avg
- Smaller ε → more peaked, local influence
- Larger ε → broader, smoother influence
- More forgiving than MQ

**Wendland Support Radius c**:
- Typical: c = 3 × d_avg (ensures overlap between influence regions)
- Must be large enough that each point has neighbors within support
- Trade-off: larger c = more computational cost but smoother result
- Minimum: c ≥ 2 × d_avg

### Optimization Approaches

**Leave-One-Out Cross-Validation (LOOCV)**:
```
LOOCV(ε) = Σᵢ (f(xᵢ) - fᵢ)² / n
```
Where f(xᵢ) is computed using all points except i.

**Generalized Cross-Validation (GCV)**:
More efficient approximation to LOOCV for large datasets.

---

## Computational Complexity

| Method | Matrix Size | Setup | Evaluation per Point |
|--------|-------------|-------|---------------------|
| MQ/IMQ | n×n dense | O(n³) | O(n) |
| TPS | (n+3)×(n+3) dense | O(n³) | O(n) |
| Wendland | n×n sparse | O(n·k²) | O(k) |

Where k = average neighbors within support radius (typically 10-50 for Wendland)

**For astrophotography** with 1000-5000 stars:
- Global methods (MQ/IMQ/TPS): ~seconds for setup
- Wendland: ~milliseconds for setup, suitable for real-time preview

---

## Topology Preservation

For image warping, we require diffeomorphic transformations (smooth, invertible, topology-preserving).

### Jacobian Determinant

At each point **x**, the Jacobian matrix:
```
J = [∂Tₓ/∂x  ∂Tₓ/∂y]
    [∂Tᵧ/∂x  ∂Tᵧ/∂y]

det(J) = (∂Tₓ/∂x)(∂Tᵧ/∂y) - (∂Tₓ/∂y)(∂Tᵧ/∂x)
```

**Requirement**: det(J) > 0 everywhere (orientation-preserving)

### Monitoring

After computing weights:
1. Sample grid of points across image domain
2. Compute Jacobian at each point
3. Verify det(J) > 0 for all samples
4. If violations occur: adjust shape parameter or add regularization

### Comparative Topology Preservation

**Best to Worst** (based on research):
1. Wendland ψ₃,₁ - excellent topology preservation with proper c
2. Inverse Multiquadric - local influence aids stability
3. Thin-Plate Spline - global nature can cause issues with sparse/irregular points
4. Multiquadric - requires careful ε selection for dense landmarks

**Reference**: 
- https://www.sciencedirect.com/science/article/abs/pii/S0167865508003449

---

## Implementation Recommendations

### For Astrophotography Registration

**Control Points**: 
- Use detected star positions (100-5000 points typical)
- Triangle similarity provides correspondence
- Consider outlier rejection before RBF setup

**Recommended Approach**:

1. **Start with Wendland ψ₃,₁**:
   - Best computational efficiency for large star counts
   - Good topology preservation
   - Local corrections preserve distant regions
   - Support radius c = 3 × median_star_spacing

2. **Fall back to Inverse Multiquadric**:
   - If Wendland doesn't provide enough smoothness
   - Better for sparser control points
   - ε = 0.5 × median_star_spacing

3. **Reserve Thin-Plate Spline**:
   - Gold standard for well-distributed points
   - Good theoretical properties
   - Use when computational cost acceptable

4. **Avoid pure Multiquadric**:
   - More sensitive to parameter tuning
   - IMQ generally more stable

### Regularization

For noisy control points (star centroid uncertainty):

Add regularization term to system:
```
[Φ + λI  P] [w] = [d]
[Pᵀ      0] [c]   [0]
```

- λ = 10⁻⁴ to 10⁻² (scaled by problem)
- Prevents overfitting to noise
- Improves numerical conditioning
- Still passes through control points approximately

---

## Key References

### Foundational Papers

1. Hardy, R.L. (1971). "Multiquadric equations of topography and other irregular surfaces". Journal of Geophysical Research, 76(8), 1905-1915.

2. Bookstein, F.L. (1989). "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations". IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(6), 567-585.

3. Wendland, H. (1995). "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree". Advances in Computational Mathematics, 4, 389-396.

4. Franke, R. (1982). "Scattered data interpolation: Tests of some methods". Mathematics of Computation, 38(157), 181-200.

### Image Registration Applications

5. Fornefett, M., Rohr, K., Stiehl, H.S. (2001). "Radial basis functions with compact support for elastic registration of medical images". Image and Vision Computing, 19, 87-96.

6. Goshtasby, A. (2012). "Image Registration: Principles, Tools and Methods". Springer. Chapter 9: Transformation Functions.

7. Masood et al. (2008). "A locally constrained radial basis function for registration and warping of images". Pattern Recognition Letters, 29(16), 2206-2222.

### Mathematical Foundations

8. Buhmann, M.D. (2003). "Radial Basis Functions: Theory and Implementations". Cambridge University Press.

9. Fasshauer, G.E. (2007). "Meshfree Approximation Methods with MATLAB". World Scientific.

10. Carlson, R.E., Foley, T.A. (1991). "The parameter R² in multiquadric interpolation". Computers & Mathematics with Applications, 21(9), 29-42.

### Online Resources

- ALGLIB RBF Documentation: https://www.alglib.net/interpolation/fastrbf.php
- Scipy RBF Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
- Image Warping Using RBF: https://www.researchgate.net/publication/263732164_Image_warping_using_radial_basis_functions

---

## Comparison Summary

| Feature | Multiquadric | Inverse MQ | Thin-Plate | Wendland |
|---------|-------------|------------|------------|----------|
| Support | Global | Global | Global | Compact |
| Matrix | Dense | Dense | Dense | Sparse |
| Smoothness | C^∞ | C^∞ | C² | C²-C⁴ |
| Complexity | O(n³) | O(n³) | O(n³) | O(n·k²) |
| Locality | Low | Medium | Low | High |
| Conditioning | Moderate | Good | Moderate | Excellent |
| Accuracy | Excellent | Very Good | Excellent | Very Good |
| Scalability | Poor (>1K pts) | Poor | Poor | Excellent |
| Best For | Global trends | Localized | Standard | Dense fields |

---

## Next Steps

After implementing RBF transformation computation:

1. **Spatial Indexing**: Build k-d tree for efficient neighbor queries (Wendland)
2. **Image Resampling**: Apply transformation to image pixels (separate interpolation stage)
3. **Quality Metrics**: Compute Jacobian determinant distribution
4. **Validation**: Compare transformed star positions to expected positions
5. **Optimization**: Parameter tuning via LOOCV or visual assessment