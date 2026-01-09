#include "covariance.h"

namespace INVERSELIB {

Eigen::MatrixXd Covariance::compute_empirical(const Eigen::MatrixXd& data)
{
    // data: (n_channels, n_samples)
    int n_samples = data.cols();
    if (n_samples < 2) return Eigen::MatrixXd::Zero(data.rows(), data.rows());
    
    // Center data
    Eigen::VectorXd mean = data.rowwise().mean();
    Eigen::MatrixXd centered = data.colwise() - mean;
    
    // Cov = (X * X^T) / (n - 1)
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(n_samples - 1);
    
    return cov;
}

Eigen::MatrixXd Covariance::regularize(const Eigen::MatrixXd& cov, double lambda)
{
    int n = cov.rows();
    Eigen::MatrixXd reg = cov;
    
    // Add lambda to diagonal
    for(int i=0; i<n; ++i) {
        reg(i, i) += lambda;
    }
    
    return reg;
}

} // NAMESPACE
