/**
 * Statistical quality tests for random number generation
 */

#include "test_common.h"
#include <math.h>

#define LARGE_SAMPLE 100000
#define TOLERANCE_MEAN 0.01
#define TOLERANCE_STD 0.05

/**
 * Calculate mean of array
 */
static double calc_mean(double *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

/**
 * Calculate standard deviation
 */
static double calc_std(double *data, size_t n, double mean) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / n);
}

/**
 * Calculate skewness
 */
static double calc_skewness(double *data, size_t n, double mean, double std) {
    double sum_cube = 0.0;
    for (size_t i = 0; i < n; i++) {
        double z = (data[i] - mean) / std;
        sum_cube += z * z * z;
    }
    return sum_cube / n;
}

/**
 * Calculate kurtosis
 */
static double calc_kurtosis(double *data, size_t n, double mean, double std) {
    double sum_quad = 0.0;
    for (size_t i = 0; i < n; i++) {
        double z = (data[i] - mean) / std;
        sum_quad += z * z * z * z;
    }
    return sum_quad / n;
}

/**
 * Error function approximation for CDF calculation
 */
static double erf_approx(double x) {
    // Abramowitz and Stegun approximation
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    int sign = (x < 0) ? -1 : 1;
    x = fabs(x);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}

/**
 * Normal CDF for Kolmogorov-Smirnov test
 */
static double normal_cdf(double x, double mean, double std) {
    return 0.5 * (1.0 + erf_approx((x - mean) / (std * sqrt(2.0))));
}

/**
 * Compare function for qsort
 */
static int compare_doubles(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

/**
 * Kolmogorov-Smirnov test statistic
 * Returns the maximum distance between empirical and theoretical CDF
 */
static double ks_statistic(double *data, size_t n, double mean, double std) {
    // Sort data
    double *sorted = malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double empirical_cdf = (i + 1.0) / n;
        double theoretical_cdf = normal_cdf(sorted[i], mean, std);
        double diff = fabs(empirical_cdf - theoretical_cdf);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    free(sorted);
    return max_diff;
}

/**
 * Chi-squared goodness-of-fit test
 * Divides data into bins and compares observed vs expected frequencies
 */
static double chi_squared_test(double *data, size_t n, double mean, double std) {
    const int num_bins = 10;
    int observed[10] = {0};
    // Bin boundaries from -3σ to +3σ
    double bin_edges[11];
    for (int i = 0; i <= num_bins; i++) {
        bin_edges[i] = mean + std * (-3.0 + 6.0 * i / num_bins);
    }
    // Count observations in each bin
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < num_bins; j++) {
            if (data[i] >= bin_edges[j] && data[i] < bin_edges[j + 1]) {
                observed[j]++;
                break;
            }
        }
    }
    // Calculate expected frequencies and chi-squared
    double chi_sq = 0.0;
    for (int i = 0; i < num_bins; i++) {
        double p_lower = normal_cdf(bin_edges[i], mean, std);
        double p_upper = normal_cdf(bin_edges[i + 1], mean, std);
        double expected = n * (p_upper - p_lower);
        if (expected > 5.0) {  // Valid chi-squared test requirement
            double diff = observed[i] - expected;
            chi_sq += (diff * diff) / expected;
        }
    }
    return chi_sq;
}

/**
 * Runs test for randomness/independence
 * Counts runs above/below median
 */
static double runs_test(double *data, size_t n) {
    double median = calc_mean(data, n);  // Approximate with mean
    int runs = 1;
    for (size_t i = 1; i < n; i++) {
        if ((data[i] > median) != (data[i-1] > median)) {
            runs++;
        }
    }
    // Expected runs and standard deviation under null hypothesis
    int n_above = 0, n_below = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] > median) n_above++;
        else n_below++;
    }
    double expected_runs = (2.0 * n_above * n_below) / n + 1.0;
    double var_runs = (2.0 * n_above * n_below * (2.0 * n_above * n_below - n)) / 
                      (n * n * (n - 1.0));
    double std_runs = sqrt(var_runs);
    return fabs((runs - expected_runs) / std_runs);  // Z score
}

/**
 * Lag-1 autocorrelation test
 * Should be near 0 for independent samples
 */
static double autocorrelation_lag1(double *data, size_t n, double mean) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < n - 1; i++) {
        numerator += (data[i] - mean) * (data[i + 1] - mean);
    }
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        denominator += diff * diff;
    }
    return numerator / denominator;
}

/**
 * Poisson CDF using direct summation
 */
static double poisson_cdf(int k, double lambda) {
    if (k < 0) return 0.0;
    double sum = 0.0;
    double term = exp(-lambda);
    sum += term;
    for (int i = 1; i <= k; i++) {
        term *= lambda / i;
        sum += term;
    }
    return sum;
}

/**
 * Chi-squared goodness-of-fit test for Poisson distribution
 */
static double chi_squared_poisson(double *data, size_t n, double lambda) {
    // Find range of observed values
    int min_val = (int)data[0];
    int max_val = (int)data[0];
    for (size_t i = 1; i < n; i++) {
        int val = (int)data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    // Create bins for observed frequencies
    int range = max_val - min_val + 1;
    int *observed = calloc(range, sizeof(int));
    for (size_t i = 0; i < n; i++) {
        int bin = (int)data[i] - min_val;
        if (bin >= 0 && bin < range) {
            observed[bin]++;
        }
    }
    // Calculate chi-squared statistic
    double chi_sq = 0.0;
    int bins_used = 0;
    for (int i = 0; i < range; i++) {
        int k = min_val + i;
        double prob;
        if (k >= 0) {
            // P(X = k) = P(X <= k) - P(X <= k-1)
            prob = poisson_cdf(k, lambda) - poisson_cdf(k - 1, lambda);
        } else {
            prob = 0.0;
        }
        double expected = n * prob;
        // Only use bins with expected frequency >= 5
        if (expected >= 5.0) {
            double diff = observed[i] - expected;
            chi_sq += (diff * diff) / expected;
            bins_used++;
        }
    }
    free(observed);
    return chi_sq;
}

void test_randnorm_mean_variance(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double expected_mean = 5.0;
    double expected_std = 2.0;
    NDArray arr = ndarray_new_randnorm(dims, expected_mean, expected_std);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double actual_mean = calc_mean(arr->data, LARGE_SAMPLE);
    double actual_std = calc_std(arr->data, LARGE_SAMPLE, actual_mean);
    CU_ASSERT_DOUBLE_EQUAL(actual_mean, expected_mean, expected_std * TOLERANCE_MEAN * 3);
    CU_ASSERT_DOUBLE_EQUAL(actual_std, expected_std, expected_std * TOLERANCE_STD);
    printf("\n  Mean: %.4f (expected: %.4f)", actual_mean, expected_mean);
    printf("\n  Std:  %.4f (expected: %.4f)", actual_std, expected_std);
    ndarray_free(arr);
}

void test_randnorm_standard_normal(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    NDArray arr = ndarray_new_randnorm(dims, 0.0, 1.0);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double std = calc_std(arr->data, LARGE_SAMPLE, mean);
    CU_ASSERT_DOUBLE_EQUAL(mean, 0.0, 0.03);
    CU_ASSERT_DOUBLE_EQUAL(std, 1.0, 0.05);
    printf("\n  Standard Normal - Mean: %.4f, Std: %.4f", mean, std);
    ndarray_free(arr);
}

void test_randnorm_higher_moments(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    NDArray arr = ndarray_new_randnorm(dims, 0.0, 1.0);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double std = calc_std(arr->data, LARGE_SAMPLE, mean);
    double skewness = calc_skewness(arr->data, LARGE_SAMPLE, mean, std);
    double kurtosis = calc_kurtosis(arr->data, LARGE_SAMPLE, mean, std);
    CU_ASSERT(fabs(skewness) < 0.1);
    CU_ASSERT(fabs(kurtosis - 3.0) < 0.2);
    printf("\n  Skewness: %.4f (expected: ~0.0)", skewness);
    printf("\n  Kurtosis: %.4f (expected: ~3.0)", kurtosis);
    ndarray_free(arr);
}

void test_randnorm_kolmogorov_smirnov(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double mean = 0.0, std = 1.0;
    NDArray arr = ndarray_new_randnorm(dims, mean, std);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double ks_stat = ks_statistic(arr->data, LARGE_SAMPLE, mean, std);
    // Critical value at α=0.01 for large samples: ~1.63/sqrt(n)
    double critical_value = 1.63 / sqrt(LARGE_SAMPLE);
    CU_ASSERT(ks_stat < critical_value);
    printf("\n  K-S statistic: %.6f (critical: %.6f)", ks_stat, critical_value);
    ndarray_free(arr);
}

void test_randnorm_chi_squared(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    NDArray arr = ndarray_new_randnorm(dims, 0.0, 1.0);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double std = calc_std(arr->data, LARGE_SAMPLE, mean);
    double chi_sq = chi_squared_test(arr->data, LARGE_SAMPLE, mean, std);
    // Chi-squared with 9 df (10 bins - 1), critical value at α=0.01 is ~21.67
    CU_ASSERT(chi_sq < 21.67);
    printf("\n  Chi-squared: %.4f (critical: 21.67 at α=0.01)", chi_sq);
    ndarray_free(arr);
}

void test_randnorm_runs_test(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    NDArray arr = ndarray_new_randnorm(dims, 0.0, 1.0);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double z_score = runs_test(arr->data, LARGE_SAMPLE);
    // Z-score should be < 2.58 for 99% confidence
    CU_ASSERT(z_score < 2.58);
    printf("\n  Runs test Z-score: %.4f (should be < 2.58)", z_score);
    ndarray_free(arr);
}

void test_randnorm_autocorrelation(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    NDArray arr = ndarray_new_randnorm(dims, 0.0, 1.0);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double acf1 = autocorrelation_lag1(arr->data, LARGE_SAMPLE, mean);
    // Autocorrelation should be near 0 for independent samples
    CU_ASSERT(fabs(acf1) < 0.05);
    printf("\n  Lag-1 autocorrelation: %.6f (expected: ~0.0)", acf1);
    ndarray_free(arr);
}

void test_randunif_mean_variance(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double low = 0.0, high = 10.0;
    NDArray arr = ndarray_new_randunif(dims, low, high);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double expected_mean = (low + high) / 2.0;
    double expected_std = (high - low) / sqrt(12.0);
    double actual_mean = calc_mean(arr->data, LARGE_SAMPLE);
    double actual_std = calc_std(arr->data, LARGE_SAMPLE, actual_mean);
    CU_ASSERT_DOUBLE_EQUAL(actual_mean, expected_mean, 0.1);
    CU_ASSERT_DOUBLE_EQUAL(actual_std, expected_std, 0.1);
    printf("\n  Uniform - Mean: %.4f (expected: %.4f)", actual_mean, expected_mean);
    printf("\n  Uniform - Std:  %.4f (expected: %.4f)", actual_std, expected_std);
    ndarray_free(arr);
}

void test_randunif_bounds(void) {
    size_t dims[] = {1, 10000, 0};
    double low = -5.0, high = 5.0;
    NDArray arr = ndarray_new_randunif(dims, low, high);
    CU_ASSERT_PTR_NOT_NULL(arr);
    // All values should be within bounds
    for (size_t i = 0; i < 10000; i++) {
        CU_ASSERT(arr->data[i] >= low);
        CU_ASSERT(arr->data[i] < high);
    }
    printf("\n  All values within [%.1f, %.1f)", low, high);
    ndarray_free(arr);
}

void test_randpoisson_mean_variance(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 5.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double actual_mean = calc_mean(arr->data, LARGE_SAMPLE);
    double actual_std = calc_std(arr->data, LARGE_SAMPLE, actual_mean);
    // For Poisson distribution: mean = variance = lambda
    double expected_std = sqrt(lambda);
    // Mean should be close to lambda
    CU_ASSERT_DOUBLE_EQUAL(actual_mean, lambda, lambda * 0.05);
    // Std should be close to sqrt(lambda)
    CU_ASSERT_DOUBLE_EQUAL(actual_std, expected_std, expected_std * 0.1);
    printf("\n  Poisson(λ=%.1f) - Mean: %.4f (expected: %.4f)", 
           lambda, actual_mean, lambda);
    printf("\n  Poisson(λ=%.1f) - Std:  %.4f (expected: %.4f)", 
           lambda, actual_std, expected_std);
    ndarray_free(arr);
}

void test_randpoisson_integer_values(void) {
    size_t dims[] = {1, 10000, 0};
    double lambda = 3.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    // All values should be non-negative integers
    for (size_t i = 0; i < 10000; i++) {
        CU_ASSERT(arr->data[i] >= 0.0);
        CU_ASSERT(arr->data[i] == floor(arr->data[i]));
    }
    printf("\n  All values are non-negative integers");
    ndarray_free(arr);
}

void test_randpoisson_small_lambda(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 0.5;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double actual_mean = calc_mean(arr->data, LARGE_SAMPLE);
    double actual_var = calc_std(arr->data, LARGE_SAMPLE, actual_mean);
    actual_var = actual_var * actual_var;
    // For small lambda, mean ≈ variance ≈ lambda
    CU_ASSERT_DOUBLE_EQUAL(actual_mean, lambda, 0.05);
    CU_ASSERT_DOUBLE_EQUAL(actual_var, lambda, 0.1);
    printf("\n  Poisson(λ=%.1f) - Mean: %.4f, Variance: %.4f", 
           lambda, actual_mean, actual_var);
    ndarray_free(arr);
}

void test_randpoisson_large_lambda(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 20.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double actual_mean = calc_mean(arr->data, LARGE_SAMPLE);
    double actual_std = calc_std(arr->data, LARGE_SAMPLE, actual_mean);
    double expected_std = sqrt(lambda);
    CU_ASSERT_DOUBLE_EQUAL(actual_mean, lambda, lambda * 0.05);
    CU_ASSERT_DOUBLE_EQUAL(actual_std, expected_std, expected_std * 0.1);
    printf("\n  Poisson(λ=%.1f) - Mean: %.4f (expected: %.4f)", 
           lambda, actual_mean, lambda);
    printf("\n  Poisson(λ=%.1f) - Std:  %.4f (expected: %.4f)", 
           lambda, actual_std, expected_std);
    ndarray_free(arr);
}

void test_randpoisson_chi_squared(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 5.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double chi_sq = chi_squared_poisson(arr->data, LARGE_SAMPLE, lambda);
    // Chi-squared critical value depends on degrees of freedom
    // For Poisson with λ=5, we expect ~10-15 bins with expected freq >= 5
    // Using conservative critical value at α=0.01 with ~12 df: ~26.22
    CU_ASSERT(chi_sq < 30.0);
    printf("\n  Chi-squared: %.4f (should be < 30.0)", chi_sq);
    ndarray_free(arr);
}

void test_randpoisson_dispersion(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 7.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double var = calc_std(arr->data, LARGE_SAMPLE, mean);
    var = var * var;
    // Index of dispersion (variance-to-mean ratio) should be ~1 for Poisson
    double dispersion = var / mean;
    CU_ASSERT_DOUBLE_EQUAL(dispersion, 1.0, 0.1);
    printf("\n  Index of dispersion: %.4f (expected: ~1.0)", dispersion);
    printf("\n  Mean: %.4f, Variance: %.4f", mean, var);
    ndarray_free(arr);
}

void test_randpoisson_autocorrelation(void) {
    size_t dims[] = {1, LARGE_SAMPLE, 0};
    double lambda = 5.0;
    NDArray arr = ndarray_new_randpoisson(dims, lambda);
    CU_ASSERT_PTR_NOT_NULL(arr);
    double mean = calc_mean(arr->data, LARGE_SAMPLE);
    double acf1 = autocorrelation_lag1(arr->data, LARGE_SAMPLE, mean);
    // Autocorrelation should be near 0 for independent samples
    CU_ASSERT(fabs(acf1) < 0.05);
    printf("\n  Lag-1 autocorrelation: %.6f (expected: ~0.0)", acf1);
    ndarray_free(arr);
}

/* Test registration function */
void register_randquality_tests(CU_pSuite suite) {
    CU_add_test(suite, "Gaussian: Mean & Variance", test_randnorm_mean_variance);
    CU_add_test(suite, "Gaussian: Standard Normal", test_randnorm_standard_normal);
    CU_add_test(suite, "Gaussian: Higher Moments", test_randnorm_higher_moments);
    CU_add_test(suite, "Gaussian: Kolmogorov-Smirnov", test_randnorm_kolmogorov_smirnov);
    CU_add_test(suite, "Gaussian: Chi-squared", test_randnorm_chi_squared);
    CU_add_test(suite, "Gaussian: Runs Test", test_randnorm_runs_test);
    CU_add_test(suite, "Gaussian: Autocorrelation", test_randnorm_autocorrelation);
    CU_add_test(suite, "Uniform: Mean & Variance", test_randunif_mean_variance);
    CU_add_test(suite, "Uniform: Bounds Check", test_randunif_bounds);
    CU_add_test(suite, "Poisson: Mean & Variance", test_randpoisson_mean_variance);
    CU_add_test(suite, "Poisson: Integer Values", test_randpoisson_integer_values);
    CU_add_test(suite, "Poisson: Small Lambda", test_randpoisson_small_lambda);
    CU_add_test(suite, "Poisson: Large Lambda", test_randpoisson_large_lambda);
    CU_add_test(suite, "Poisson: Chi-squared", test_randpoisson_chi_squared);
    CU_add_test(suite, "Poisson: Dispersion Index", test_randpoisson_dispersion);
    CU_add_test(suite, "Poisson: Autocorrelation", test_randpoisson_autocorrelation);
}
