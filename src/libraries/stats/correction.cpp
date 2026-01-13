#include "correction.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace STATSLIB {

Eigen::VectorXd Correction::bonferroni(const Eigen::VectorXd& p_values)
{
    long n_tests = p_values.size();
    Eigen::VectorXd p_corr = p_values * (double)n_tests;
    
    // Clamp to 1.0
    for(int i=0; i<n_tests; ++i) {
        if (p_corr[i] > 1.0) p_corr[i] = 1.0;
    }
    
    return p_corr;
}

Eigen::VectorXd Correction::holm(const Eigen::VectorXd& p_values, double alpha)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // Sort indices by p-values
    std::vector<int> idx = sortIndices(p_values);
    
    Eigen::VectorXd p_sorted(n);
    for(int i = 0; i < n; ++i) {
        p_sorted[i] = p_values[idx[i]];
    }
    
    // Apply Holm correction
    Eigen::VectorXd p_corr_sorted(n);
    for(int i = 0; i < n; ++i) {
        double correction_factor = (double)(n - i);
        p_corr_sorted[i] = std::min(1.0, p_sorted[i] * correction_factor);
    }
    
    // Ensure monotonicity (step-down procedure)
    for(int i = 1; i < n; ++i) {
        p_corr_sorted[i] = std::max(p_corr_sorted[i], p_corr_sorted[i-1]);
    }
    
    return restoreOrder(p_corr_sorted, idx);
}

Eigen::VectorXd Correction::sidak(const Eigen::VectorXd& p_values)
{
    long n_tests = p_values.size();
    Eigen::VectorXd p_corr(n_tests);
    
    for(int i = 0; i < n_tests; ++i) {
        // p_corr = 1 - (1 - p)^n_tests
        p_corr[i] = 1.0 - std::pow(1.0 - p_values[i], (double)n_tests);
        if (p_corr[i] > 1.0) p_corr[i] = 1.0;
    }
    
    return p_corr;
}

Eigen::VectorXd Correction::holmSidak(const Eigen::VectorXd& p_values, double alpha)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // Sort indices by p-values
    std::vector<int> idx = sortIndices(p_values);
    
    Eigen::VectorXd p_sorted(n);
    for(int i = 0; i < n; ++i) {
        p_sorted[i] = p_values[idx[i]];
    }
    
    // Apply Holm-Sidak correction
    Eigen::VectorXd p_corr_sorted(n);
    for(int i = 0; i < n; ++i) {
        double m = (double)(n - i);  // Number of remaining tests
        // p_corr = 1 - (1 - p)^m
        p_corr_sorted[i] = 1.0 - std::pow(1.0 - p_sorted[i], m);
        if (p_corr_sorted[i] > 1.0) p_corr_sorted[i] = 1.0;
    }
    
    // Ensure monotonicity (step-down procedure)
    for(int i = 1; i < n; ++i) {
        p_corr_sorted[i] = std::max(p_corr_sorted[i], p_corr_sorted[i-1]);
    }
    
    return restoreOrder(p_corr_sorted, idx);
}

Eigen::VectorXd Correction::fdr(const Eigen::VectorXd& p_values)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // Sort indices by p-values
    std::vector<int> idx = sortIndices(p_values);
    
    Eigen::VectorXd p_sorted(n);
    for(int i = 0; i < n; ++i) {
        p_sorted[i] = p_values[idx[i]];
    }
    
    // Compute q-values: q[i] = p[i] * n / (i + 1)
    Eigen::VectorXd p_corr_sorted(n);
    for(int i = 0; i < n; ++i) {
        double rank = (double)(i + 1);
        p_corr_sorted[i] = p_sorted[i] * (double)n / rank;
    }
    
    // Ensure monotonicity (from end to start)
    for(int i = n-2; i >= 0; --i) {
        p_corr_sorted[i] = std::min(p_corr_sorted[i], p_corr_sorted[i+1]);
    }
    
    // Clamp to 1.0
    for(int i = 0; i < n; ++i) {
        if (p_corr_sorted[i] > 1.0) p_corr_sorted[i] = 1.0;
    }
    
    return restoreOrder(p_corr_sorted, idx);
}

Eigen::VectorXd Correction::fdrBY(const Eigen::VectorXd& p_values)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // Compute harmonic sum: c(n) = sum(1/i) for i=1 to n
    double harmonic_sum = 0.0;
    for(int i = 1; i <= n; ++i) {
        harmonic_sum += 1.0 / (double)i;
    }
    
    // Sort indices by p-values
    std::vector<int> idx = sortIndices(p_values);
    
    Eigen::VectorXd p_sorted(n);
    for(int i = 0; i < n; ++i) {
        p_sorted[i] = p_values[idx[i]];
    }
    
    // Compute q-values: q[i] = p[i] * n * c(n) / (i + 1)
    Eigen::VectorXd p_corr_sorted(n);
    for(int i = 0; i < n; ++i) {
        double rank = (double)(i + 1);
        p_corr_sorted[i] = p_sorted[i] * (double)n * harmonic_sum / rank;
    }
    
    // Ensure monotonicity (from end to start)
    for(int i = n-2; i >= 0; --i) {
        p_corr_sorted[i] = std::min(p_corr_sorted[i], p_corr_sorted[i+1]);
    }
    
    // Clamp to 1.0
    for(int i = 0; i < n; ++i) {
        if (p_corr_sorted[i] > 1.0) p_corr_sorted[i] = 1.0;
    }
    
    return restoreOrder(p_corr_sorted, idx);
}

Eigen::VectorXd Correction::fdrTwoStage(const Eigen::VectorXd& p_values, double alpha)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // Stage 1: Apply standard BH procedure
    Eigen::VectorXd q_values_stage1 = fdr(p_values);
    
    // Count rejections at level alpha
    int R1 = countRejected(q_values_stage1, alpha);
    
    if (R1 == 0) {
        // No rejections in stage 1, return stage 1 results
        return q_values_stage1;
    }
    
    // Stage 2: Estimate pi0 and apply adaptive procedure
    double pi0_hat = estimatePi0(p_values);
    
    // Sort indices by p-values
    std::vector<int> idx = sortIndices(p_values);
    
    Eigen::VectorXd p_sorted(n);
    for(int i = 0; i < n; ++i) {
        p_sorted[i] = p_values[idx[i]];
    }
    
    // Compute adaptive q-values: q[i] = p[i] * pi0_hat * n / (i + 1)
    Eigen::VectorXd p_corr_sorted(n);
    for(int i = 0; i < n; ++i) {
        double rank = (double)(i + 1);
        p_corr_sorted[i] = p_sorted[i] * pi0_hat * (double)n / rank;
    }
    
    // Ensure monotonicity (from end to start)
    for(int i = n-2; i >= 0; --i) {
        p_corr_sorted[i] = std::min(p_corr_sorted[i], p_corr_sorted[i+1]);
    }
    
    // Clamp to 1.0
    for(int i = 0; i < n; ++i) {
        if (p_corr_sorted[i] > 1.0) p_corr_sorted[i] = 1.0;
    }
    
    return restoreOrder(p_corr_sorted, idx);
}

Eigen::VectorXd Correction::multipleComparison(const Eigen::VectorXd& p_values, 
                                              Method method, 
                                              double alpha)
{
    switch (method) {
        case Method::Bonferroni:
            return bonferroni(p_values);
        case Method::Holm:
            return holm(p_values, alpha);
        case Method::Sidak:
            return sidak(p_values);
        case Method::HolmSidak:
            return holmSidak(p_values, alpha);
        case Method::FDR_BH:
            return fdr(p_values);
        case Method::FDR_BY:
            return fdrBY(p_values);
        case Method::FDR_TST:
            return fdrTwoStage(p_values, alpha);
        case Method::None:
        default:
            return p_values;
    }
}

std::vector<bool> Correction::getRejectionMask(const Eigen::VectorXd& p_corrected, 
                                              double alpha)
{
    std::vector<bool> rejected(p_corrected.size());
    for(int i = 0; i < p_corrected.size(); ++i) {
        rejected[i] = p_corrected[i] <= alpha;
    }
    return rejected;
}

int Correction::countRejected(const Eigen::VectorXd& p_corrected, double alpha)
{
    int count = 0;
    for(int i = 0; i < p_corrected.size(); ++i) {
        if (p_corrected[i] <= alpha) {
            count++;
        }
    }
    return count;
}

double Correction::estimatePi0(const Eigen::VectorXd& p_values, double lambda)
{
    if (p_values.size() == 0) return 1.0;
    
    // Count p-values greater than lambda
    int count = 0;
    for(int i = 0; i < p_values.size(); ++i) {
        if (p_values[i] > lambda) {
            count++;
        }
    }
    
    // Estimate pi0 = (# p-values > lambda) / ((1 - lambda) * n)
    double pi0_hat = (double)count / ((1.0 - lambda) * (double)p_values.size());
    
    // Clamp to [0, 1]
    return std::min(1.0, std::max(0.0, pi0_hat));
}

std::vector<int> Correction::sortIndices(const Eigen::VectorXd& p_values)
{
    long n = p_values.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    
    // Sort indices based on p_values
    std::sort(idx.begin(), idx.end(), [&p_values](int i1, int i2) {
        return p_values[i1] < p_values[i2];
    });
    
    return idx;
}

Eigen::VectorXd Correction::restoreOrder(const Eigen::VectorXd& sorted_values, 
                                        const std::vector<int>& indices)
{
    long n = sorted_values.size();
    Eigen::VectorXd result(n);
    for(int i = 0; i < n; ++i) {
        result[indices[i]] = sorted_values[i];
    }
    return result;
}

} // NAMESPACE
