module Algorithms.Statistics

/// Calculate median of an array
let calculateMedian (values: float[]) : float =
    if Array.isEmpty values then 0.0
    else
        let sorted = Array.sort values
        let mid = sorted.Length / 2
        if sorted.Length % 2 = 0 then
            (sorted.[mid - 1] + sorted.[mid]) / 2.0
        else
            sorted.[mid]

/// Calculate median absolute deviation from a given median
let calculateMAD (values: float[]) (median: float) : float =
    if Array.isEmpty values then 0.0
    else
        let deviations = values |> Array.map (fun v -> abs (v - median))
        calculateMedian deviations

/// Convert MAD to sigma (standard deviation estimate)
/// Uses the standard conversion factor for normal distributions
let madToSigma (mad: float) : float =
    mad / 0.6745

/// K-sigma clipped noise estimator
/// Iteratively rejects outliers to estimate robust noise level
let estimateNoiseKSigma (pixels: float[]) : float =
    if Array.length pixels < 2 then 0.0
    else
        let mutable working = Array.copy pixels
        let mutable converged = false
        let maxIterations = 10
        let mutable iteration = 0
        let k = 3.0

        while not converged && iteration < maxIterations do
            let median = calculateMedian working
            let mad = calculateMAD working median
            let sigma = madToSigma mad

            if sigma <= 0.0 then
                converged <- true
            else
                let loThreshold = median - k * sigma
                let hiThreshold = median + k * sigma
                let filtered = working |> Array.filter (fun p -> p >= loThreshold && p <= hiThreshold)

                if filtered.Length = working.Length || filtered.Length < 10 then
                    converged <- true
                else
                    working <- filtered
                    iteration <- iteration + 1

        let finalMedian = calculateMedian working
        let finalMAD = calculateMAD working finalMedian
        madToSigma finalMAD

/// Calculate signal-to-noise ratio
let calculateSNR (mean: float) (stdDev: float) : float =
    if stdDev > 0.0 then mean / stdDev
    else 0.0

/// Calculate basic descriptive statistics
/// Returns (min, max, mean, median, stdDev)
let calculateBasicStats (values: float[]) : float * float * float * float * float =
    if Array.isEmpty values then
        (0.0, 0.0, 0.0, 0.0, 0.0)
    else
        let min = Array.min values
        let max = Array.max values
        let mean = Array.average values
        let median = calculateMedian values
        let variance = values |> Array.sumBy (fun x -> (x - mean) * (x - mean))
        let stdDev = sqrt (variance / float values.Length)
        (min, max, mean, median, stdDev)

/// Calculate histogram with specified number of bins
let calculateHistogram (values: float[]) (bins: int) : int[] =
    if Array.isEmpty values then
        Array.zeroCreate bins
    else
        let minVal = Array.min values
        let maxVal = Array.max values
        let range = maxVal - minVal
        let histogram = Array.zeroCreate bins

        if range > 0.0 then
            for value in values do
                let binIndex = int ((value - minVal) / range * float (bins - 1))
                let clampedIndex = min (bins - 1) (max 0 binIndex)
                histogram.[clampedIndex] <- histogram.[clampedIndex] + 1

        histogram
