# Deep Learning Noise Reduction: Noise2Noise and Self-Supervised Methods

## Executive Summary

This document explores self-supervised deep learning methods for image denoising, specifically Noise2Noise and Self2Self approaches. These methods enable training denoising neural networks on user-provided noisy data **without requiring clean reference images**, making them ideal for astrophotography where clean ground truth is often unavailable.

## 1. Noise2Noise: Learning Without Clean Data

### 1.1 Core Principle

Noise2Noise (N2N) demonstrates that neural networks can learn to denoise images by training exclusively on pairs of independently corrupted versions of the same image, without ever seeing clean examples. The fundamental insight: a network can learn to predict the true signal even when trained only on noisy observations.

### 1.2 Mathematical Foundation

**Training Data Structure:**
- Traditional (Noise2Clean): pairs `(s + n, s)` where `s` is clean signal, `n` is noise
- Noise2Noise: pairs `(s + n, s + n')` where `n` and `n'` are independent noise realizations

**Key Insight:**
The neural network cannot learn to perfectly predict one noisy image from another, but networks trained on this task converge to the same predictions as traditionally trained networks with access to ground truth.

**Loss Function Behavior:**
- **L2 Loss (MSE):** Recovers the mean of targets `E[y]`, where `y = s + n`
  - Since `E[s + n] = E[s] + E[n] = s` (for zero-mean noise)
  - Minimizing `E[(f(x) - y)²]` drives `f(x) → E[y] = s`

- **L1 Loss (MAE):** Recovers the median of targets
  - Useful when up to 50% of pixels are outliers
  - More robust to impulsive noise

**Convergence Properties:**
- In theory, N2N training reaches the same performance as Noise2Clean (N2C) with infinite training data
- In practice, with finite datasets, N2N falls slightly short of N2C but remains highly effective

### 1.3 Applications

Successfully demonstrated on:
- Photographic noise removal (Gaussian, Poisson)
- Monte Carlo rendering denoising
- MRI reconstruction
- **Astrophotography** (Noise2Astro project)

## 2. Self2Self: Single Image Self-Supervised Denoising

### 2.1 Overview

Self2Self advances beyond Noise2Noise by enabling training from a **single noisy image**, making it truly unsupervised. Published at CVPR 2020 by Quan et al.

### 2.2 Technical Approach

**Dropout-Based Training:**
- Uses Bernoulli sampling (dropout) to create multiple instances of the input image
- Defines self-prediction loss on pairs of dropout-sampled instances
- Dropout ensemble provides regularization by making neuron activations stochastic

**Blind-Spot Strategy:**
- Random masks cover parts of the input image
- Network predicts covered regions from uncovered regions
- Prevents learning identity mapping (critical for avoiding trivial solutions)

**Self-Supervision Mechanism:**
```
Input: Single noisy image I
1. Apply random dropout mask M₁ → I₁ = I ⊙ M₁
2. Apply random dropout mask M₂ → I₂ = I ⊙ M₂
3. Train network f to predict I₂ from I₁
4. Loss: L = ||f(I₁) - I₂||² (only on overlapping regions)
```

### 2.3 Performance

Self2Self significantly outperforms:
- Existing single-image learning methods
- Traditional non-learning denoising (BM3D, NLM)
- Competitive with networks trained on external datasets

### 2.4 Experimental Results: XisfPrep Implementation (2025-11-29)

**Objective:** Validate Self2Self denoising on uncalibrated single-exposure astrophotography data with high thermal noise.

**Test Data:**
- Single 720s Ha exposure (Sh2-108 region)
- Sensor temperature: 0°C (intentionally warm to maximize thermal noise)
- Uncalibrated (no dark/bias/flat subtraction)
- 6252×4176 pixels, UInt16, mono
- Original stats: Min=364, Max=65534, Mean=570.78, Median=551.00, StdDev=500.57

**Implementation Details:**
- Architecture: 5-layer CNN, 48 filters, residual learning → direct prediction → residual with normalization
- Training: Random 512×512 patches per epoch
- Dropout rate: 0.5 → 0.3 (reduced to preserve detail)
- Normalization: Input/output scaled by ÷65535 to [0,1] range
- Loss function: MSE on dropped pixels + soft range penalty
- Optimizer: AdamW, learning rate 0.001

**Training Progression (5000 epochs):**
```
Epoch    1: Loss = 134,122,624
Epoch   10: Loss =   5,941,454
Epoch  100: Loss =     187,466
Epoch 1000: Loss =      67,822
Epoch 5000: Loss =      34,307

Late training oscillation:
Epoch 4950: Loss = 16,373
Epoch 4960: Loss = 24,942
Epoch 4970: Loss = 16,350
Epoch 4980: Loss = 35,394
Epoch 4990: Loss = 77,592
Epoch 5000: Loss = 34,307
```

**Critical Observation:** Loss oscillated wildly (16k → 77k) even after 5000 epochs, indicating non-convergence.

**Output Results:**
```
Denoised output: Min=0, Max=65535, Mean=463.24, Median=435.00, StdDev=430.28
```

**Quantitative Analysis:**
| Metric | Input | Output | Change | Assessment |
|--------|-------|--------|--------|------------|
| Mean | 570.78 | 463.24 | -18.8% | ❌ Systematic darkening |
| Median | 551.00 | 435.00 | -21.1% | ❌ Brightness loss |
| StdDev | 500.57 | 430.28 | -14.0% | ⚠️ Minimal denoising |
| Min | 364 | 0 | Created blacks | ❌ Invalid pixels |
| Max | 65534 | 65535 | Clipping | ❌ Lost bright data |

**Qualitative Assessment:**
- Visual appearance: "Mushed" rather than denoised
- Nebula detail: Lost, not preserved
- Stars: Cores intact but faint signal degraded
- Noise: Appeared amplified/spread rather than reduced

**Fundamental Issues Identified:**

1. **Non-Convergent Training**
   - Each epoch trains on a single random 512×512 patch with random dropout
   - Extreme variance in loss values (5× oscillation after 5000 epochs)
   - Model never sees the same data twice → online learning on infinite stream
   - Insufficient samples to learn stable denoising function

2. **Training-Inference Distribution Shift**
   - **Training:** Input contains ~30% zeros (dropout masking)
   - **Inference:** Input contains 0% zeros (full image)
   - Model trained on degraded inputs but tested on clean inputs
   - Distribution mismatch causes unpredictable behavior

3. **Target Ambiguity**
   - Training target: Predict noisy pixel values from noisy neighbors
   - Never shown clean ground truth
   - Theory assumes spatial correlation separates signal from noise
   - Practice: Unclear if model learns denoising or just interpolation

4. **Median Shift Paradox**
   - Mathematically: MSE loss on dropped pixels should preserve median
   - Reality: 20% darkening despite loss function design
   - Suggests model learning incorrect function or bias introduced by:
     * Normalization artifacts
     * Dropout masking confusion (dark pixels vs zeros)
     * Insufficient constraint on output range

5. **Architecture Experiments**
   - Residual learning (predict noise, subtract): Massive histogram shift (+90% brightening)
   - Direct prediction (predict denoised): Most pixels → 0, only bright cores survived
   - Residual with zero-mean constraint: Noise amplified (StdDev increased!)
   - Normalized direct prediction: Systematic darkening (final attempt)

**Hypothesis: Why Self2Self Failed Here**

**Core Assumption Violations:**
1. **Single image insufficient:** Self2Self theory requires learning noise statistics from spatial correlation alone. With only 5000 random patches from one image, insufficient samples to separate:
   - Faint nebulosity (real signal, spatially smooth)
   - Thermal noise (scattered hot pixels, somewhat spatially structured)

2. **Noise characteristics:** Thermal electrons at 0°C may have weak spatial correlation (thermal gradients), violating pixel-independence assumption

3. **SNR too low:** Uncalibrated single exposure has signal-to-noise ratio where noise dominates faint regions. Model may mistake signal for noise.

**Alternative Hypothesis: Better Data Needed**

Self2Self may work significantly better on:
- **Stacked images** with higher SNR (10-20 integrated frames)
- **Calibrated data** (dark/bias/flat subtracted) to remove systematic patterns
- **Cooler sensor** (-15°C vs 0°C) with less thermal noise
- **Multiple exposures** for training (more diverse samples)

**Rationale:**
- Higher SNR → stronger signal vs noise separation
- Calibrated data → purer random noise (Poisson + read noise only)
- Stacked frames → noise statistics better match theoretical assumptions
- Model might learn "polish" rather than "rescue" when signal already dominant

**Lessons Learned:**

1. **Self-supervised ≠ Magic:** Single-image training cannot create information that isn't present. With SNR ~1.1 (Mean/StdDev = 570/500), signal barely exceeds noise.

2. **Distribution shift is critical:** Training and inference must see similar data distributions. Dropout masking creates artificial zeros that don't exist in real images.

3. **Convergence matters:** Wild loss oscillation after 5000 epochs indicates fundamental training instability, not just "needs more epochs."

4. **Validation essential:** Without clean ground truth for validation, no way to detect when model diverges from desired behavior.

**Recommendation:**

**Do not use Self2Self on uncalibrated single exposures with low SNR.**

Better approaches for this scenario:
- Traditional calibration (dark/bias/flat frames)
- Sigma clipping for hot pixels
- Median filtering for scattered noise
- Multi-frame stacking (increases SNR by √N)

**Future Investigation:**

Test Self2Self on:
- Stacked master light (20+ frames integrated)
- Calibrated data (after dark/flat subtraction)
- Higher SNR targets (bright targets, longer total integration)

This would validate whether Self2Self works on "polish" tasks (already-good data) vs "rescue" tasks (very noisy single frames).

**Status:** Experimental implementation complete. Algorithm unsuitable for intended use case. Archived for future reference with better data conditions.

## 3. Related Self-Supervised Methods

### 3.1 Noise2Void (N2V)

**Key Innovation:** Blind-spot network architecture
- Masks the central pixel of each receptive field
- Predicts center pixel from surrounding pixels only
- No paired images needed—trains on single noisy images
- Published at CVPR 2019

**Mathematical Formulation:**
```
For pixel at position i:
- Receptive field: R(i)
- Blind spot: pixel i excluded from R(i)
- Prediction: f(x)[i] = g(x[R(i) \ {i}])
```

### 3.2 Neighbor2Neighbor (N2N)

**Key Innovation:** Random neighbor sub-sampling
- Generates training pairs by sub-sampling the same noisy image
- Paired pixels are spatial neighbors with similar appearance
- Avoids heavy dependence on noise distribution assumptions
- Published at CVPR 2021

**Training Procedure:**
1. Sub-sample noisy image into two sub-images (e.g., checkerboard pattern)
2. Train network to predict one sub-image from the other
3. Neighboring pixels share similar signal but independent noise

**Advantages:**
- Very simple yet effective
- No noise distribution assumptions required
- Fast training and inference

### 3.3 Comparison Matrix

| Method | Data Required | Noise Assumptions | Performance | Speed |
|--------|---------------|-------------------|-------------|-------|
| Noise2Clean | Paired clean+noisy | None | Best | Fast |
| Noise2Noise | Paired noisy only | Independent noise | Near N2C | Fast |
| Self2Self | Single noisy image | General | Very good | Medium |
| Noise2Void | Single noisy image | Zero-mean | Good | Fast |
| Neighbor2Neighbor | Single noisy image | Minimal | Very good | Fast |

## 4. Astrophotography Applications

### 4.1 Noise2Astro Project

Research specifically targeting astronomical image denoising using self-supervised methods:

**Performance Metrics:**
- Poisson noise flux recovery: **98.13%**
- Gaussian noise flux recovery: **96.45%** (smooth signal profiles)

**Advantages for Astrophotography:**
- No synthetic noise generation required
- Learns from actual telescope/camera noise characteristics
- Preserves astronomical features while removing noise
- Applicable to any telescope, instrumentation, or processing pipeline

## 5. Implementation Options

**Focus:** All methods below support training from scratch on user-provided data. The goal is to learn noise characteristics specific to the user's equipment and imaging setup, without relying on pre-trained models or commercial solutions.

### 5.1 Python Implementations

#### 5.1.1 Official Noise2Noise (TensorFlow)
**Repository:** [NVlabs/noise2noise](https://github.com/NVlabs/noise2noise)
- Official NVIDIA implementation (ICML 2018 paper)
- Tested with Python 3.6, TensorFlow
- Complete training pipeline for custom datasets
- Supports multiple noise types and restoration tasks

#### 5.1.2 PyTorch Noise2Noise Implementations

**joeylitalien/noise2noise-pytorch:**
- Unofficial PyTorch implementation
- Python 3.6.5+, tested on macOS and Ubuntu
- Clean, readable code structure

**hanyoseob/pytorch-noise2noise:**
- Includes training/testing scripts
- TensorBoard support for monitoring
- Well-documented

**sashrika15/noise2noise:**
- Supports additive Gaussian noise and text removal
- Note: Limited to specific noise types

#### 5.1.3 Self2Self Implementations

**yangpuPKU/Self2Self_pytorch_implementation:**
- PyTorch reimplementation of CVPR 2020 paper
- Custom Pconv2d structure matching TensorFlow version
- Finding: AdamW optimizer outperforms Adam

**JK-the-Ko/Self2SelfPlus:**
- Enhanced version with gated convolution
- Requires Python 3.8.10, PyTorch >= 1.12.1
- Official implementation of improved method

**JinYize/self2self_pytorch:**
- Clean PyTorch implementation
- Follows original paper closely

#### 5.1.4 Neighbor2Neighbor Implementation

**TaoHuang2018/Neighbor2Neighbor:**
- Official PyTorch implementation (CVPR 2021)
- Simple and effective
- Good starting point for single-image denoising

### 5.2 .NET Integration Options

#### 5.2.1 ONNX Runtime for .NET

**Overview:**
ONNX Runtime provides high-performance inference for deploying ONNX models in .NET applications.

**Key Features:**
- C# API for model inference
- Hardware acceleration (TensorRT, DirectML, OpenVINO)
- Cross-platform (Windows, Linux, macOS)
- Integrates with ML.NET
- **2x average performance gain on CPU** (Microsoft internal benchmarks)

**Workflow:**
1. Train model in Python (PyTorch/TensorFlow)
2. Export to ONNX format
3. Load in .NET via ONNX Runtime
4. Perform inference on XISF image data

**Example Architecture:**
```fsharp
// Pseudo-code for F# integration
module DenoiseModel =
    open Microsoft.ML.OnnxRuntime

    type DenoiserConfig = {
        ModelPath: string
        UseGPU: bool
    }

    let loadModel (config: DenoiserConfig) =
        let options = SessionOptions()
        if config.UseGPU then
            options.AppendExecutionProvider_CUDA(0)
        new InferenceSession(config.ModelPath, options)

    let denoise (session: InferenceSession) (image: float32[,]) =
        // Prepare input tensor
        // Run inference
        // Extract output
        // Return denoised image
```

**Resources:**
- [ONNX Runtime Official Site](https://onnxruntime.ai/)
- [.NET Blog: Generate images with AI using Stable Diffusion, C#, and ONNX Runtime](https://devblogs.microsoft.com/dotnet/generate-ai-images-stable-diffusion-csharp-onnx-runtime/)
- NuGet: `Microsoft.ML.OnnxRuntime`
- NuGet: `Microsoft.ML.OnnxRuntime.Gpu` (for CUDA support)

#### 5.2.2 ML.NET

**Status:** ML.NET focuses primarily on traditional ML and lacks built-in support for custom CNN architectures like U-Net or DnCNN commonly used in denoising.

**Recommendation:** Use ONNX Runtime instead for deploying pre-trained deep learning models in .NET.

### 5.3 Azure Integration Options

#### 5.3.1 Azure Machine Learning

**Training Workflow:**
1. Develop/train model in Python (Azure ML Compute)
2. Export to ONNX format
3. Deploy as REST endpoint (Azure Container Instance or AKS)
4. Call from XisfPrep via HTTP API

**Deployment Options:**
- Azure Container Instance (ACI): Low-cost, serverless inference
- Azure Kubernetes Service (AKS): Production-scale, high-throughput
- Azure ML Real-Time Endpoints: Managed inference with auto-scaling

**Resources:**
- [ONNX Runtime and Models - Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-onnx?view=azureml-api-2)
- [Deploy on AzureML](https://onnxruntime.ai/docs/tutorials/azureml.html)
- [Local inference using ONNX for AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?view=azureml-api-2)

#### 5.3.2 Azure SQL Edge (Edge Computing)

For edge deployment scenarios, Azure SQL Edge supports ONNX Runtime natively, enabling on-device inference without cloud connectivity.

### 5.4 Recommended Architecture for XisfPrep

**Hybrid Python/F# Approach:**

1. **Training Phase (Python):**
   - Use PyTorch with Neighbor2Neighbor or Self2Self
   - Train on user's raw XISF data
   - Export trained model to ONNX format

2. **Inference Phase (F#/.NET):**
   - Load ONNX model via ONNX Runtime in XisfPrep
   - Process XISF images using existing image I/O infrastructure
   - Batch processing support using existing patterns

**Alternative: Pure Python Pipeline:**
- Standalone Python module for training and inference
- Called from F# via process execution
- Simpler initially but less integrated

## 6. Mathematical Deep Dive

### 6.1 Noise2Noise Loss Function Analysis

**Objective:** Minimize expected loss over noisy observations

Let:
- `s` = true clean signal
- `n₁, n₂` = independent noise realizations
- `y₁ = s + n₁`, `y₂ = s + n₂` = noisy observations
- `f(x)` = neural network denoiser

**Noise2Clean (traditional):**
```
L_N2C = E[(f(s + n) - s)²]
```

**Noise2Noise:**
```
L_N2N = E[(f(s + n₁) - (s + n₂))²]
```

**Expanding L_N2N:**
```
L_N2N = E[(f(s + n₁) - s - n₂)²]
      = E[(f(s + n₁) - s)²] + E[n₂²] + 2E[(f(s + n₁) - s)(-n₂)]
      = E[(f(s + n₁) - s)²] + E[n₂²]  (assuming E[n₂] = 0)
      = L_N2C + constant
```

**Conclusion:** Minimizing L_N2N is equivalent to minimizing L_N2C (up to a constant). The network learns the same denoising function.

### 6.2 Self2Self Masked Prediction

**Objective:** Learn from single image by predicting dropped pixels

Let:
- `I` = noisy input image
- `M` = binary dropout mask (Bernoulli distribution)
- `I_M = I ⊙ M` = masked image

**Loss Function:**
```
L = E_M [ Σᵢ (f(I_M)[i] - I[i])² · (1 - M[i]) ]
```

Where `(1 - M[i])` ensures loss computed only on dropped pixels.

**Key Property:**
- Network cannot copy input (dropped pixels not visible)
- Must learn to predict from context (neighboring pixels)
- Averaging over many dropout masks approximates true denoising function

### 6.3 Noise2Void Blind-Spot Network

**Receptive Field Masking:**

For pixel `i` with receptive field `R(i)`:
```
f(x)[i] = g(x[R(i) \ {i}])
```

**Loss Function:**
```
L = E[ Σᵢ (f(x)[i] - x[i])² ]
```

**Why it works:**
- If noise is pixel-independent: `E[n[i] | x[R(i) \ {i}]] = 0`
- Network learns: `f(x)[i] → E[s[i] | observed neighbors]`
- Approximates true signal from local context

## 7. Implementation Considerations for XisfPrep

### 7.1 Data Requirements

**Training Data:**
- User's raw XISF frames (lights only, no calibration needed)
- Minimum: 20-50 frames recommended for Noise2Noise
- Single frame sufficient for Self2Self/Neighbor2Neighbor
- No dark frames or bias frames required (noise characteristics learned implicitly)

**Image Preprocessing:**
- Normalize pixel values to [0, 1] or [-1, 1]
- Handle Bayer pattern (train on raw CFA data or post-debayer)
- Patch extraction for training (e.g., 128×128 or 256×256 patches)

### 7.2 Network Architecture Recommendations

**U-Net:**
- Encoder-decoder with skip connections
- Excellent for image-to-image tasks
- Standard for denoising applications

**DnCNN (Denoising CNN):**
- Residual learning: predicts noise, subtracts from input
- Batch normalization for faster convergence
- Lighter weight than U-Net

**Typical Architecture (U-Net):**
```
Input: [H, W, C] (1-3 channels for mono/RGB)
Encoder: 4-5 levels, doubling filters (64→128→256→512)
Bottleneck: Highest feature count
Decoder: 4-5 levels, halving filters with skip connections
Output: [H, W, C] denoised image
```

### 7.3 Training Hyperparameters

**Learning Rate:**
- Initial: 1e-4 to 1e-3
- Scheduler: Cosine annealing or step decay
- Optimizer: AdamW (better than Adam for Self2Self)

**Batch Size:**
- 16-32 patches (depending on GPU memory)
- Larger batches more stable for Noise2Noise

**Training Duration:**
- Noise2Noise: 50-100 epochs
- Self2Self: 100-200 epochs (single image requires more iterations)
- Monitor validation loss for convergence

### 7.4 Inference Integration

**Processing Pipeline:**
```
1. Load user's raw XISF frames
2. Optional: Apply calibration (or defer to post-denoising)
3. Denoise each frame using trained ONNX model
4. Continue with existing preprocessing (alignment, stacking)
```

**Batch Processing:**
- Process multiple frames efficiently
- GPU acceleration via ONNX Runtime
- Maintain existing progress reporting

### 7.5 Model Storage

**ONNX Model Location:**
- Store in user profile: `~/.xisfprep/models/`
- Or alongside XISF data: `<project_dir>/denoising_model.onnx`
- Model size: Typically 10-100 MB depending on architecture

## 8. Proposed XisfPrep Command Design

### 8.1 Training Command

```bash
# Train a denoising model from user's raw data
xisfprep denoise train \
  --input "path/to/light_frames/*.xisf" \
  --output-model "my_denoiser.onnx" \
  --method neighbor2neighbor \
  --epochs 100 \
  --gpu
```

**Options:**
- `--method`: `noise2noise | self2self | neighbor2neighbor`
- `--architecture`: `unet | dncnn`
- `--patch-size`: Training patch size (default: 128)
- `--gpu`: Enable GPU acceleration for training

### 8.2 Inference Command

```bash
# Apply trained model to denoise images
xisfprep denoise apply \
  --input "path/to/noisy_frames/*.xisf" \
  --output "path/to/denoised/" \
  --model "my_denoiser.onnx"
```

### 8.3 Integrated Workflow Command

```bash
# Train on subset, apply to all frames
xisfprep denoise auto \
  --train-on "light_frames/[0-9].xisf" \
  --apply-to "light_frames/*.xisf" \
  --output "denoised_frames/" \
  --method neighbor2neighbor
```

## 9. Development Roadmap

### Phase 1: Research and Prototyping (Python)
- [ ] Implement Neighbor2Neighbor in PyTorch
- [ ] Test on sample XISF astrophotography data
- [ ] Export to ONNX format
- [ ] Validate ONNX inference matches PyTorch

### Phase 2: .NET Integration
- [ ] Add ONNX Runtime NuGet package to XisfPrep
- [ ] Implement F# wrapper for ONNX inference
- [ ] Integrate with existing XISF I/O infrastructure
- [ ] Add batch processing support

### Phase 3: Training Pipeline
- [ ] Create Python training script (standalone)
- [ ] Implement data loading from XISF format
- [ ] Add training progress reporting
- [ ] Export trained models automatically

### Phase 4: CLI Integration
- [ ] Add `denoise train` command
- [ ] Add `denoise apply` command
- [ ] Add configuration for model storage
- [ ] Document usage and best practices

### Phase 5: Optimization
- [ ] GPU acceleration testing
- [ ] Performance benchmarking
- [ ] Memory optimization for large XISF files
- [ ] Multi-threaded batch processing

## 10. References

### Core Papers

1. **Noise2Noise: Learning Image Restoration without Clean Data**
   - Lehtinen et al., ICML 2018
   - [arXiv:1803.04189](https://arxiv.org/abs/1803.04189)
   - [PDF](https://arxiv.org/pdf/1803.04189)

2. **Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image**
   - Quan et al., CVPR 2020
   - [CVPR 2020 Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)

3. **Noise2Void - Learning Denoising from Single Noisy Images**
   - Krull et al., CVPR 2019
   - [CVPR 2019 Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)

4. **Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images**
   - Huang et al., CVPR 2021
   - [arXiv:2101.02824](https://ar5iv.labs.arxiv.org/html/2101.02824)
   - [CVPR 2021 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf)
   - [IEEE Transactions on Image Processing](https://dl.acm.org/doi/abs/10.1109/TIP.2022.3176533)

### Astrophotography Applications

5. **Noise2Astro: Astronomical Image Denoising With Self-Supervised Neural Networks**
   - [arXiv:2209.07071](https://arxiv.org/abs/2209.07071)
   - [ResearchGate PDF](https://www.researchgate.net/publication/363584613_Noise2Astro_Astronomical_Image_Denoising_With_Self-Supervised_NeuralNetworks)
   - [NASA ADS](https://ui.adsabs.harvard.edu/abs/2022RNAAS...6..187Z/abstract)

6. **Astronomical Image Denoising by Self-Supervised Deep Learning**
   - [arXiv:2502.16807](https://arxiv.org/html/2502.16807)

7. **Solar Image Denoising with Convolutional Neural Networks**
   - [Astronomy & Astrophysics](https://www.aanda.org/articles/aa/full_html/2019/09/aa36069-19/aa36069-19.html)

8. **Image Denoising in Astrophotography – British Astronomical Association**
   - [BAA Journal](https://britastro.org/journal_contents_ite/image-denoising-in-astrophotography-an-approach-using-recent-network-denoising-models)

### Community Discussions

9. **Deep Learning Denoising for Astrophotography - Cloudy Nights**
   - [Forum Thread](https://www.cloudynights.com/forums/topic/789780-deep-learning-denoising-for-astrophotography/)

10. **Machine Learning / AI Denoising - Cloudy Nights**
    - [Forum Thread](https://www.cloudynights.com/topic/644848-machine-learning-ai-denoising/)

### Implementation Repositories

11. **Official TensorFlow Implementation (NVIDIA)**
    - [NVlabs/noise2noise](https://github.com/NVlabs/noise2noise)

12. **PyTorch Implementations**
    - [joeylitalien/noise2noise-pytorch](https://github.com/joeylitalien/noise2noise-pytorch)
    - [hanyoseob/pytorch-noise2noise](https://github.com/hanyoseob/pytorch-noise2noise)
    - [sashrika15/noise2noise](https://github.com/sashrika15/noise2noise)

13. **Self2Self PyTorch**
    - [yangpuPKU/Self2Self_pytorch_implementation](https://github.com/yangpuPKU/Self2Self_pytorch_implementation)
    - [JinYize/self2self_pytorch](https://github.com/JinYize/self2self_pytorch)
    - [JK-the-Ko/Self2SelfPlus](https://github.com/JK-the-Ko/Self2SelfPlus)

14. **Neighbor2Neighbor Official PyTorch**
    - [TaoHuang2018/Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor)

### .NET and ONNX Runtime

15. **ONNX Runtime Official**
    - [ONNX Runtime Homepage](https://onnxruntime.ai/)
    - [ONNX Runtime Inference](https://onnxruntime.ai/inference)

16. **ONNX Runtime .NET Integration**
    - [Generate AI Images with C# and ONNX Runtime - .NET Blog](https://devblogs.microsoft.com/dotnet/generate-ai-images-stable-diffusion-csharp-onnx-runtime/)
    - [ONNX Homepage](https://onnx.ai/)

17. **ONNX Model Repository**
    - [onnx/models - GitHub](https://github.com/onnx/models)

### Azure Machine Learning

18. **ONNX Runtime and Azure ML**
    - [ONNX Runtime and Models - Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-onnx?view=azureml-api-2)
    - [Deploy on AzureML](https://onnxruntime.ai/docs/tutorials/azureml.html)

19. **Azure ML ONNX Deployment**
    - [Local Inference using ONNX for AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?view=azureml-api-2)
    - [Make Predictions with AutoML ONNX Model in .NET](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automl-onnx-model-dotnet?view=azureml-api-2)
    - [Deploying Neural Network Models to Azure ML Service with Keras and ONNX](https://benalexkeen.com/deploying-neural-network-models-to-azure-ml-service-with-keras-and-onnx/)

20. **Azure ML Blog Posts**
    - [ONNX Runtime for Inferencing Machine Learning Models](https://azure.microsoft.com/en-us/blog/onnx-runtime-for-inferencing-machine-learning-models-now-in-preview/)
    - [ONNX Runtime is Now Open Source](https://azure.microsoft.com/en-us/blog/onnx-runtime-is-now-open-source/)

### Additional Resources

21. **Papers with Code**
    - [Noise2Noise: Learning Image Restoration without Clean Data](https://paperswithcode.com/paper/noise2noise-learning-image-restoration)
    - [Image Denoising Task Overview](https://paperswithcode.com/task/image-denoising)

22. **Related Methods**
    - [A Fast Blind Zero-Shot Denoiser - Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00547-8)
    - [A Fast Blind Zero-Shot Denoiser - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9674521/)
    - [Improved Noise2Noise Denoising with Limited Data](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Calvarons_Improved_Noise2Noise_Denoising_With_Limited_Data_CVPRW_2021_paper.pdf)

---

**Document Version:** 1.1
**Last Updated:** 2025-11-29
**Author:** Research compilation for XisfPrep noise reduction feature
**Change Log:**
- v1.1 (2025-11-29): Added Section 2.4 - Experimental results of Self2Self implementation on uncalibrated single-exposure data
- v1.0 (2025-11-28): Initial research compilation
