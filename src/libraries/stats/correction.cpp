#include "correction.h"
#include <algorithm>
#include <numeric>

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

Eigen::VectorXd Correction::fdr(const Eigen::VectorXd& p_values)
{
    long n = p_values.size();
    if (n == 0) return p_values;
    
    // 1. Sort P-values
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    
    // Sort indices based on p_values
    std::sort(idx.begin(), idx.end(), [&p_values](int i1, int i2) {
        return p_values[i1] < p_values[i2];
    });
    
    // 2. Compute q-values
    // p_corr[i] = p[i] * n / (i + 1)
    // Then ensure monotonicity: p_corr[i] = min(p_corr[i], p_corr[i+1])
    
    Eigen::VectorXd p_sorted(n);
    for(int i=0; i<n; ++i) p_sorted[i] = p_values[idx[i]];
    
    Eigen::VectorXd p_corr_sorted = Eigen::VectorXd::Zero(n);
    
    for(int i=0; i<n; ++i) {
        double rank = (double)(i + 1);
        double q = p_sorted[i] * (double)n / rank;
        p_corr_sorted[i] = q;
    }
    
    // Monotonicity (from end to start)
    for(int i=n-2; i>=0; --i) {
        if (p_corr_sorted[i] > p_corr_sorted[i+1]) {
            p_corr_sorted[i] = p_corr_sorted[i+1];
        }
    }
    
    // Clamp to 1.0
    for(int i=0; i<n; ++i) {
        if (p_corr_sorted[i] > 1.0) p_corr_sorted[i] = 1.0;
    }
    
    // 3. Restore order
    Eigen::VectorXd result(n);
    for(int i=0; i<n; ++i) {
        result[idx[i]] = p_corr_sorted[i];
    }
    
    return result;
}

} // NAMESPACE
