#include "ttest.h"
#include <cmath>
#include <iostream>

namespace STATSLIB {

// Helper to compute T-distribution CDF/Survival function would be needed for P-values.
// For now, we return 0.0 for P-values or a very rough approximation.
// Or we can use `std::erfc` for normal approximation if dof is large.
double p_value_from_t(double t, double df) {
    // Very rough Normal approximation for large df
    // p = 2 * (1 - CDF(|t|)) = 2 * (1 - (0.5 * erfc(-|t|/sqrt(2)))) ?
    // CDF_normal(x) = 0.5 * (1 + erf(x/sqrt(2)))
    // SF_normal(x) = 1 - CDF(x) = 0.5 * (1 - erf(x/sqrt(2))) = 0.5 * erfc(x/sqrt(2))
    // 2-sided p = 2 * SF(|t|) = erfc(|t|/sqrt(2))
    // This assumes T converges to Normal, which is true for large df (>30).
    return std::erfc(std::abs(t) / std::sqrt(2.0));
}

std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> TTest::ttest_1samp(const Eigen::MatrixXd& data)
{
    // data: (n_samples, n_features)
    int n = data.rows();
    if (n < 2) return {Eigen::RowVectorXd::Zero(data.cols()), Eigen::RowVectorXd::Zero(data.cols())};
    
    // Mean
    Eigen::RowVectorXd mean = data.colwise().mean();
    
    // Variance (unbiased)
    Eigen::MatrixXd centered = data.rowwise() - mean;
    Eigen::RowVectorXd var = (centered.array().square().colwise().sum()) / (n - 1);
    
    // Std Error of Mean = sqrt(var / n)
    Eigen::RowVectorXd sem = (var / n).array().sqrt();
    
    // T = mean / sem
    Eigen::RowVectorXd t_vals = mean.array() / sem.array();
    
    // P-values
    Eigen::RowVectorXd p_vals(data.cols());
    for(int i=0; i<data.cols(); ++i) {
        p_vals[i] = p_value_from_t(t_vals[i], n - 1);
    }
    
    return {t_vals, p_vals};
}

std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> TTest::ttest_ind(const Eigen::MatrixXd& data1, const Eigen::MatrixXd& data2)
{
    // data1: (n1, n_features)
    // data2: (n2, n_features)
    int n1 = data1.rows();
    int n2 = data2.rows();
    
    if (n1 < 2 || n2 < 2 || data1.cols() != data2.cols()) {
         return {Eigen::RowVectorXd::Zero(data1.cols()), Eigen::RowVectorXd::Zero(data1.cols())};
    }
    
    // Mean
    Eigen::RowVectorXd mean1 = data1.colwise().mean();
    Eigen::RowVectorXd mean2 = data2.colwise().mean();
    
    // Variance
    Eigen::MatrixXd c1 = data1.rowwise() - mean1;
    Eigen::RowVectorXd var1 = (c1.array().square().colwise().sum()) / (n1 - 1);
    
    Eigen::MatrixXd c2 = data2.rowwise() - mean2;
    Eigen::RowVectorXd var2 = (c2.array().square().colwise().sum()) / (n2 - 1);
    
    // Pooled Variance (assuming equal variance)
    // s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)
    Eigen::RowVectorXd pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    
    // SE = sqrt(s_p^2 * (1/n1 + 1/n2))
    Eigen::RowVectorXd se = (pooled_var * (1.0/n1 + 1.0/n2)).array().sqrt();
    
    // T = (mean1 - mean2) / SE
    Eigen::RowVectorXd t_vals = (mean1 - mean2).array() / se.array();
    
    // P-values
    Eigen::RowVectorXd p_vals(data1.cols());
    for(int i=0; i<data1.cols(); ++i) {
        p_vals[i] = p_value_from_t(t_vals[i], n1 + n2 - 2);
    }
    
    return {t_vals, p_vals};
}

} // NAMESPACE
