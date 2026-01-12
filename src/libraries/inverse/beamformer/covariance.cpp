#include "covariance.h"
#include <iostream>

namespace INVERSELIB {

Covariance::Covariance() 
    : nfree(0), loglik(0.0), is_empty(true)
{
}

Covariance::Covariance(const Eigen::MatrixXd& data, const std::vector<std::string>& names)
    : data(data), names(names), nfree(0), loglik(0.0), method("empirical"), is_empty(false)
{
}

Covariance Covariance::compute_empirical(const Eigen::MatrixXd& data, const std::vector<std::string>& names)
{
    Covariance cov;
    cov.names = names;
    
    // data: (n_channels, n_samples)
    int n_samples = data.cols();
    if (n_samples < 2) {
        cov.data = Eigen::MatrixXd::Zero(data.rows(), data.rows());
        cov.nfree = 0;
        cov.is_empty = true;
        return cov;
    }
    
    // Center data
    Eigen::VectorXd mean = data.rowwise().mean();
    Eigen::MatrixXd centered = data.colwise() - mean;
    
    // Cov = (X * X^T) / (n - 1)
    cov.data = (centered * centered.transpose()) / (double)(n_samples - 1);
    cov.nfree = n_samples - 1;
    cov.is_empty = false;
    cov.method = "empirical";
    
    return cov;
}

Covariance Covariance::compute_empirical(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<std::string>& names)
{
    Covariance cov;
    cov.names = names;
    
    if (epochs.empty()) {
        cov.is_empty = true;
        return cov;
    }
    
    int n_epochs = epochs.size();
    int n_channels = epochs[0].rows();
    int n_times = epochs[0].cols();
    
    // Validate dimensions
    for (const auto& epoch : epochs) {
        if (epoch.rows() != n_channels || epoch.cols() != n_times) {
            std::cerr << "Covariance::compute_empirical: Epoch dimensions mismatch." << std::endl;
            cov.is_empty = true;
            return cov;
        }
    }
    
    // Accumulate scatter matrix: sum((X - mean) * (X - mean)^T)
    // For empirical covariance across epochs, we usually assume a common mean (often 0 for noise cov) or per-epoch mean.
    // MNE-Python default: subtract mean per epoch.
    
    Eigen::MatrixXd scatter = Eigen::MatrixXd::Zero(n_channels, n_channels);
    long long total_samples = 0;
    
    for (const auto& epoch : epochs) {
        Eigen::VectorXd mean = epoch.rowwise().mean();
        Eigen::MatrixXd centered = epoch.colwise() - mean;
        scatter += centered * centered.transpose();
        total_samples += n_times;
    }
    
    // DoF = N_epochs * (N_times - 1)
    // Or just (Total_Samples - 1) ? 
    // MNE-Python: n_samples - 1 (if keeping mean), or n_samples - n_epochs (if removing mean per epoch).
    // Let's use n_samples - n_epochs (unbiased estimator removing per-epoch mean)
    
    long long dof = total_samples - n_epochs;
    if (dof < 1) dof = 1;
    
    cov.data = scatter / (double)dof;
    cov.nfree = (int)dof;
    cov.is_empty = false;
    cov.method = "empirical";
    
    return cov;
}

Covariance Covariance::regularize(double lambda) const
{
    Covariance reg_cov = *this;
    if (is_empty) return reg_cov;
    
    int n = data.rows();
    for(int i=0; i<n; ++i) {
        reg_cov.data(i, i) += lambda;
    }
    
    return reg_cov;
}

Covariance Covariance::regularize_scaled(double reg) const
{
    if (is_empty) return *this;
    
    double trace = data.trace();
    double avg_eig = trace / (double)data.rows();
    double lambda = reg * avg_eig;
    
    return regularize(lambda);
}

} // NAMESPACE
