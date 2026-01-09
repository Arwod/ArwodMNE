#include "lcmv.h"
#include <iostream>
#include <Eigen/Dense>

namespace INVERSELIB {

Eigen::MatrixXd LCMV::compute_weights(const Eigen::MatrixXd& leadfield, 
                                      const Eigen::MatrixXd& data_cov, 
                                      double reg)
{
    // L: (n_channels, n_sources)
    // C: (n_channels, n_channels)
    
    // 1. Regularize Covariance
    // lambda = reg * trace(C) / n_channels  (typical scaling)
    // But user might pass absolute lambda. Let's assume 'reg' is fraction of average eigenvalue.
    double trace = data_cov.trace();
    double avg_eig = trace / (double)data_cov.rows();
    double lambda = reg * avg_eig;
    
    Eigen::MatrixXd C_inv = (data_cov + lambda * Eigen::MatrixXd::Identity(data_cov.rows(), data_cov.cols())).inverse();
    
    // 2. Compute Weights
    // W = (L^T C^-1 L)^-1 L^T C^-1
    
    // Numerator term: L^T C^-1
    // (n_sources, n_channels)
    Eigen::MatrixXd num = leadfield.transpose() * C_inv;
    
    // Denominator term: L^T C^-1 L
    // (n_sources, n_sources)
    Eigen::MatrixXd den = num * leadfield;
    
    // Invert denominator
    Eigen::MatrixXd den_inv = den.inverse();
    
    // W = den_inv * num
    Eigen::MatrixXd W = den_inv * num;
    
    return W;
}

Eigen::MatrixXd LCMV::apply(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& data)
{
    // W: (n_sources, n_channels)
    // data: (n_channels, n_times)
    // S = W * data
    return weights * data;
}

} // NAMESPACE
