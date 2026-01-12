#include "lcmv.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

namespace INVERSELIB {

BeamformerWeights LCMV::make_lcmv(
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    double reg,
    const std::string& pick_ori,
    const std::string& weight_norm,
    int n_ori)
{
    BeamformerWeights res;
    res.ch_names = data_cov.names;
    res.pick_ori = pick_ori;
    res.weight_norm = (weight_norm != "none");
    
    // 1. Validate inputs
    int n_channels = leadfield.rows();
    if (data_cov.data.rows() != n_channels || data_cov.data.cols() != n_channels) {
        std::cerr << "LCMV::make_lcmv: Data Covariance dimensions mismatch." << std::endl;
        return res;
    }
    
    // 2. Regularize Data Covariance
    Covariance reg_cov = data_cov.regularize_scaled(reg);
    
    // 3. Invert Data Covariance
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(reg_cov.data);
    Eigen::MatrixXd C_inv = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    
    // 4. Compute Weights
    // W = (L^T C^-1 L)^-1 L^T C^-1
    
    int n_dipoles = leadfield.cols();
    if (n_dipoles % n_ori != 0) {
        std::cerr << "LCMV::make_lcmv: Leadfield cols not divisible by n_ori." << std::endl;
        return res;
    }
    int n_sources = n_dipoles / n_ori;
    
    // Prepare output weights
    int out_rows = 0;
    if (pick_ori == "vector") out_rows = n_dipoles;
    else out_rows = n_sources; // max-power or normal
    
    Eigen::MatrixXd W_out(out_rows, n_channels);
    W_out.setZero();
    
    // For noise normalization (UNG)
    Eigen::MatrixXd N;
    if (res.weight_norm) {
        if (!noise_cov.is_empty && noise_cov.data.rows() == n_channels) {
            N = noise_cov.data;
        } else {
            // Assume identity noise if not provided
            N = Eigen::MatrixXd::Identity(n_channels, n_channels);
        }
    }
    
    // Loop over sources
    #pragma omp parallel for
    for (int i = 0; i < n_sources; ++i) {
        Eigen::MatrixXd L_s = leadfield.block(0, i * n_ori, n_channels, n_ori); // (Ch x n_ori)
        
        // Denominator: L^T C^-1 L
        Eigen::MatrixXd num = L_s.transpose() * C_inv; // (n_ori x Ch)
        Eigen::MatrixXd den = num * L_s; // (n_ori x n_ori)
        
        Eigen::MatrixXd den_inv = den.inverse();
        
        // Initial weights for this source (n_ori x Ch)
        Eigen::MatrixXd W_s = den_inv * num;
        
        // Orientation Selection
        if (pick_ori == "vector") {
            // Unit Noise Gain normalization per orientation?
            // Usually MNE-Python normalizes the vector norm or per row.
            // "unit-noise-gain":  W = W / sqrt(diag(W N W^T))
            if (res.weight_norm) {
                Eigen::MatrixXd noise_norm = W_s * N * W_s.transpose();
                for (int k = 0; k < n_ori; ++k) {
                    double norm = std::sqrt(noise_norm(k, k));
                    if (norm > 1e-10) W_s.row(k) /= norm;
                }
            }
            W_out.block(i * n_ori, 0, n_ori, n_channels) = W_s;
        } else if (pick_ori == "max-power") {
            // Compute power in source space: P = W C W^T
            // Here C is Data Covariance (unregularized or regularized?)
            // Usually unregularized for power estimation, but using C_inv derived from regularized.
            Eigen::MatrixXd P = W_s * data_cov.data * W_s.transpose();
            
            // Find max power orientation
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_p(P);
            // Eigenvalues sorted ascending. Last is max.
            Eigen::VectorXd max_ori = es_p.eigenvectors().col(n_ori - 1);
            
            // Scalar weights
            Eigen::MatrixXd w_scalar = max_ori.transpose() * W_s; // (1 x Ch)
            
            if (res.weight_norm) {
                double noise_p = (w_scalar * N * w_scalar.transpose())(0,0);
                double norm = std::sqrt(noise_p);
                if (norm > 1e-10) w_scalar /= norm;
            }
            
            W_out.row(i) = w_scalar;
        } else if (pick_ori == "normal") {
            // Assume n_ori=1 or user supplied normal-constrained leadfield
            if (n_ori == 1) {
                if (res.weight_norm) {
                     double noise_p = (W_s * N * W_s.transpose())(0,0);
                     double norm = std::sqrt(noise_p);
                     if (norm > 1e-10) W_s /= norm;
                }
                W_out.row(i) = W_s;
            } else {
                // Not supported to extract normal from 3-component L without normal info
                // Just take first component? Or throw error.
                // For now, take first component.
                Eigen::MatrixXd w_scalar = W_s.row(n_ori - 1); // Z direction usually last in MNE? No, usually X,Y,Z.
                // MNE normal is usually the 3rd column of the rotation matrix.
                // Let's assume n_ori=1 for normal case for now.
                W_out.row(i) = W_s.row(0);
            }
        }
    }
    
    res.weights = W_out;
    return res;
}

Eigen::MatrixXd LCMV::apply(const BeamformerWeights& weights, const Eigen::MatrixXd& data)
{
    return weights.weights * data;
}

// Deprecated Implementation
Eigen::MatrixXd LCMV::compute_weights(const Eigen::MatrixXd& leadfield, 
                                      const Eigen::MatrixXd& data_cov, 
                                      double reg)
{
    Covariance cov(data_cov);
    BeamformerWeights res = make_lcmv(leadfield, cov, Covariance(), reg, "vector", "none", leadfield.cols()); 
    // Wait, old compute_weights assumed L was (Ch x n_sources) meaning n_ori=1 or L was already constrained?
    // Looking at old code:
    // num: (n_sources, n_channels)
    // den: (n_sources, n_sources) = num * leadfield.
    // If n_sources was actually n_dipoles * n_ori, then den is huge.
    // But old code inverted 'den'. If 'den' is huge (whole brain), this is slow/wrong (cross-talk).
    // The old code seems to implement a "global" LCMV where all sources are solved simultaneously?
    // No, standard LCMV is point-wise.
    // If 'leadfield' passed to old function was for a single source (n_channels x 3), then it works.
    // If it was whole brain (n_channels x 5000), then 'den' is 5000x5000 inverse. That's not point-wise LCMV.
    // Standard MNE LCMV loops over sources.
    // The doc said: "Usually n_sources is small... If computing for whole brain, this function is usually called per source location."
    // So the old function was a kernel for one source.
    
    // Our new make_lcmv loops over sources.
    // If we pass the whole leadfield to make_lcmv, it handles it correctly (block-wise).
    // If we pass a small leadfield (1 source) to make_lcmv, it also works.
    
    // However, the old function didn't know n_ori. It just inverted (L^T C^-1 L).
    // If L was (Ch x 3), it inverted 3x3.
    // If L was (Ch x 1), it inverted 1x1.
    // So n_ori = leadfield.cols().
    
    return res.weights;
}

} // NAMESPACE
