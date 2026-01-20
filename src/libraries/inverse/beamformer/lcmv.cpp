#include "lcmv.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

namespace INVERSELIB {

// Forward declaration of helper function
BeamformerWeights apply_orientation_selection_and_normalization(
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    const std::string& weight_norm,
    int n_ori,
    const std::string& pick_ori);

BeamformerWeights LCMV::apply_orientation_selection_and_normalization(
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    const std::string& weight_norm,
    int n_ori,
    const std::string& pick_ori)
{
    BeamformerWeights res;
    
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    
    // Prepare noise covariance for normalization
    Eigen::MatrixXd N;
    if (weight_norm != "none") {
        if (!noise_cov.is_empty && noise_cov.data.rows() == n_channels) {
            N = noise_cov.data;
        } else {
            // Assume identity noise if not provided
            N = Eigen::MatrixXd::Identity(n_channels, n_channels);
        }
    }
    
    // Prepare output weights
    int out_rows = 0;
    if (pick_ori == "vector") out_rows = n_dipoles;
    else out_rows = n_sources; // max-power or normal
    
    Eigen::MatrixXd W_out(out_rows, n_channels);
    W_out.setZero();
    
    // Initialize max_power_ori matrix if needed
    if (pick_ori == "max-power") {
        res.max_power_ori = Eigen::MatrixXd::Zero(n_sources, 3);
    }
    
    // Loop over sources
    for (int i = 0; i < n_sources; ++i) {
        Eigen::MatrixXd L_s = leadfield.block(0, i * n_ori, n_channels, n_ori); // (Ch x n_ori)
        Eigen::MatrixXd W_s = weights.block(i * n_ori, 0, n_ori, n_channels); // (n_ori x Ch)
        
        // Orientation Selection
        if (pick_ori == "vector") {
            // Apply normalization based on weight_norm parameter
            if (weight_norm == "unit-noise-gain") {
                // Standard UNG: W = W / sqrt(diag(W N W^T))
                Eigen::MatrixXd noise_norm = W_s * N * W_s.transpose();
                for (int k = 0; k < n_ori; ++k) {
                    double norm = std::sqrt(noise_norm(k, k));
                    if (norm > 1e-10) W_s.row(k) /= norm;
                }
            } else if (weight_norm == "unit-noise-gain-invariant") {
                // UNGI normalization - more robust to leadfield scaling
                Eigen::MatrixXd G_s = L_s.transpose() * N.inverse() * L_s;
                Eigen::MatrixXd G_inv = G_s.inverse();
                double trace_G_inv = G_inv.trace();
                W_s *= std::sqrt(trace_G_inv / n_ori);
            } else if (weight_norm == "nai" || weight_norm == "neural-activity-index") {
                // Neural Activity Index normalization
                Eigen::MatrixXd noise_power = W_s * N * W_s.transpose();
                Eigen::MatrixXd signal_power = W_s * L_s;
                for (int k = 0; k < n_ori; ++k) {
                    double noise_p = noise_power(k, k);
                    double signal_p = signal_power(k, k);
                    if (noise_p > 1e-10 && signal_p > 1e-10) {
                        W_s.row(k) *= std::sqrt(signal_p / noise_p);
                    }
                }
            }
            W_out.block(i * n_ori, 0, n_ori, n_channels) = W_s;
        } else if (pick_ori == "max-power") {
            // Compute power in source space: P = W C W^T
            Eigen::MatrixXd P = W_s * data_cov.data * W_s.transpose();
            
            // Find max power orientation
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_p(P);
            // Eigenvalues sorted ascending. Last is max.
            Eigen::VectorXd max_ori = es_p.eigenvectors().col(n_ori - 1);
            
            // Store max power orientation
            if (n_ori == 3) {
                res.max_power_ori.row(i) = max_ori.transpose();
            }
            
            // Scalar weights
            Eigen::MatrixXd w_scalar = max_ori.transpose() * W_s; // (1 x Ch)
            
            if (weight_norm == "unit-noise-gain") {
                double noise_p = (w_scalar * N * w_scalar.transpose())(0,0);
                double norm = std::sqrt(noise_p);
                if (norm > 1e-10) w_scalar /= norm;
            } else if (weight_norm == "unit-noise-gain-invariant") {
                Eigen::MatrixXd G_s = L_s.transpose() * N.inverse() * L_s;
                double trace_G = G_s.trace();
                w_scalar *= std::sqrt(trace_G);
            } else if (weight_norm == "nai" || weight_norm == "neural-activity-index") {
                double noise_p = (w_scalar * N * w_scalar.transpose())(0,0);
                double signal_p = (w_scalar * L_s * max_ori)(0,0);
                if (noise_p > 1e-10 && signal_p > 1e-10) {
                    w_scalar *= std::sqrt(signal_p / noise_p);
                }
            }
            
            W_out.row(i) = w_scalar;
        } else if (pick_ori == "normal") {
            // Assume n_ori=1 or user supplied normal-constrained leadfield
            if (n_ori == 1) {
                if (weight_norm == "unit-noise-gain") {
                     double noise_p = (W_s * N * W_s.transpose())(0,0);
                     double norm = std::sqrt(noise_p);
                     if (norm > 1e-10) W_s /= norm;
                } else if (weight_norm == "unit-noise-gain-invariant") {
                    Eigen::MatrixXd G_s = L_s.transpose() * N.inverse() * L_s;
                    double trace_G = G_s.trace();
                    W_s *= std::sqrt(trace_G);
                } else if (weight_norm == "nai" || weight_norm == "neural-activity-index") {
                    double noise_p = (W_s * N * W_s.transpose())(0,0);
                    double signal_p = (W_s * L_s)(0,0);
                    if (noise_p > 1e-10 && signal_p > 1e-10) {
                        W_s *= std::sqrt(signal_p / noise_p);
                    }
                }
                W_out.row(i) = W_s;
            } else {
                // Take the normal component (assume it's the last orientation)
                Eigen::MatrixXd w_scalar = W_s.row(n_ori - 1); // Assume normal is last
                W_out.row(i) = w_scalar;
            }
        }
    }
    
    res.weights = W_out;
    return res;
}

BeamformerWeights LCMV::make_lcmv(
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    double reg,
    const std::string& pick_ori,
    const std::string& weight_norm,
    int n_ori,
    double depth,
    bool reduce_rank,
    const std::string& inversion)
{
    BeamformerWeights res;
    res.ch_names = data_cov.names;
    res.pick_ori = pick_ori;
    res.weight_norm = weight_norm;
    res.depth_param = depth;
    res.inversion_method = inversion;
    res.is_free_ori = (pick_ori == "vector");
    
    // 1. Validate inputs
    int n_channels = leadfield.rows();
    if (data_cov.data.rows() != n_channels || data_cov.data.cols() != n_channels) {
        std::cerr << "LCMV::make_lcmv: Data Covariance dimensions mismatch." << std::endl;
        return res;
    }
    
    int n_dipoles = leadfield.cols();
    if (n_dipoles % n_ori != 0) {
        std::cerr << "LCMV::make_lcmv: Leadfield cols not divisible by n_ori." << std::endl;
        return res;
    }
    int n_sources = n_dipoles / n_ori;
    res.n_sources = n_sources;
    
    // 2. Apply depth weighting if requested
    Eigen::MatrixXd weighted_leadfield = leadfield;
    if (depth > 0.0) {
        // For now, apply simple depth weighting without source positions
        // In a full implementation, this would require source positions
        for (int i = 0; i < n_sources; ++i) {
            // Simple depth weighting: scale by (1 + depth * norm)
            Eigen::MatrixXd L_s = leadfield.block(0, i * n_ori, n_channels, n_ori);
            double norm = L_s.norm();
            double weight = std::pow(norm, depth);
            weighted_leadfield.block(0, i * n_ori, n_channels, n_ori) *= weight;
        }
    }
    
    // 3. Choose computation method
    if (inversion == "matrix") {
        res = compute_matrix_inversion_weights(weighted_leadfield, data_cov, noise_cov, 
                                             reg, weight_norm, n_ori, pick_ori);
    } else if (inversion == "single") {
        res = compute_single_dipole_weights(weighted_leadfield, data_cov, noise_cov, 
                                          reg, weight_norm, n_ori, pick_ori);
    } else {
        std::cerr << "LCMV::make_lcmv: Unknown inversion method: " << inversion << std::endl;
        return res;
    }
    
    // Copy metadata
    res.ch_names = data_cov.names;
    res.pick_ori = pick_ori;
    res.weight_norm = weight_norm;
    res.depth_param = depth;
    res.inversion_method = inversion;
    res.is_free_ori = (pick_ori == "vector");
    res.n_sources = n_sources;
    res.rank = std::min(n_channels, data_cov.nfree);
    
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

Eigen::MatrixXd LCMV::compute_resolution_matrix(
    const Eigen::MatrixXd& leadfield,
    const BeamformerWeights& weights,
    int n_ori)
{
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    
    // Resolution matrix R = W * L
    // where W is (n_sources x n_channels) and L is (n_channels x n_sources*n_ori)
    // For vector case, we need to handle the orientation dimension
    
    Eigen::MatrixXd resolution(n_sources, n_sources);
    resolution.setZero();
    
    if (weights.pick_ori == "vector") {
        // For vector case, compute power-based resolution
        for (int i = 0; i < n_sources; ++i) {
            for (int j = 0; j < n_sources; ++j) {
                Eigen::MatrixXd W_i = weights.weights.block(i * n_ori, 0, n_ori, n_channels);
                Eigen::MatrixXd L_j = leadfield.block(0, j * n_ori, n_channels, n_ori);
                
                // Compute W_i * L_j (n_ori x n_ori matrix)
                Eigen::MatrixXd interaction = W_i * L_j;
                
                // Take Frobenius norm squared as resolution measure
                resolution(i, j) = interaction.squaredNorm();
            }
        }
    } else {
        // For scalar case (max-power or normal)
        for (int i = 0; i < n_sources; ++i) {
            for (int j = 0; j < n_sources; ++j) {
                Eigen::VectorXd W_i = weights.weights.row(i);
                Eigen::MatrixXd L_j = leadfield.block(0, j * n_ori, n_channels, n_ori);
                
                // Compute W_i * L_j (1 x n_ori vector)
                Eigen::VectorXd interaction = W_i.transpose().transpose() * L_j;
                
                // Take squared norm as resolution measure
                resolution(i, j) = interaction.squaredNorm();
            }
        }
    }
    
    return resolution;
}

BeamformerWeights LCMV::apply_nai_normalization(
    const BeamformerWeights& weights,
    const Eigen::MatrixXd& leadfield,
    const Covariance& noise_cov,
    int n_ori)
{
    BeamformerWeights normalized_weights = weights;
    
    int n_channels = leadfield.rows();
    int n_sources = weights.weights.rows() / (weights.pick_ori == "vector" ? n_ori : 1);
    
    Eigen::MatrixXd N = noise_cov.is_empty ? 
        Eigen::MatrixXd::Identity(n_channels, n_channels) : noise_cov.data;
    
    if (weights.pick_ori == "vector") {
        for (int i = 0; i < n_sources; ++i) {
            Eigen::MatrixXd W_i = weights.weights.block(i * n_ori, 0, n_ori, n_channels);
            Eigen::MatrixXd L_i = leadfield.block(0, i * n_ori, n_channels, n_ori);
            
            // Compute noise and signal power for each orientation
            Eigen::MatrixXd noise_power = W_i * N * W_i.transpose();
            Eigen::MatrixXd signal_power = W_i * L_i;
            
            for (int k = 0; k < n_ori; ++k) {
                double noise_p = noise_power(k, k);
                double signal_p = signal_power(k, k);
                if (noise_p > 1e-10 && signal_p > 1e-10) {
                    normalized_weights.weights.block(i * n_ori + k, 0, 1, n_channels) *= 
                        std::sqrt(signal_p / noise_p);
                }
            }
        }
    } else {
        for (int i = 0; i < n_sources; ++i) {
            Eigen::VectorXd W_i = weights.weights.row(i);
            Eigen::MatrixXd L_i = leadfield.block(0, i * n_ori, n_channels, n_ori);
            
            double noise_p = W_i.transpose().dot(N * W_i);
            double signal_p = (W_i.transpose().transpose() * L_i).squaredNorm();
            
            if (noise_p > 1e-10 && signal_p > 1e-10) {
                normalized_weights.weights.row(i) *= std::sqrt(signal_p / noise_p);
            }
        }
    }
    
    return normalized_weights;
}

BeamformerWeights LCMV::apply_neural_activity_index_normalization(
    const BeamformerWeights& weights,
    const Eigen::MatrixXd& leadfield,
    const Covariance& noise_cov,
    int n_ori)
{
    return apply_nai_normalization(weights, leadfield, noise_cov, n_ori);
}

Eigen::MatrixXd LCMV::apply_depth_weighting(
    const Eigen::MatrixXd& leadfield,
    double depth,
    const Eigen::MatrixXd& source_positions,
    int n_ori)
{
    if (depth <= 0.0) {
        return leadfield;
    }
    
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    
    Eigen::MatrixXd weighted_leadfield = leadfield;
    
    if (source_positions.rows() == n_sources && source_positions.cols() == 3) {
        // Use actual source positions for depth weighting
        for (int i = 0; i < n_sources; ++i) {
            // Compute distance from origin (head center)
            double distance = source_positions.row(i).norm();
            
            // Apply depth weighting: weight = distance^depth
            double weight = std::pow(distance / 100.0, depth); // Normalize by 100mm
            
            weighted_leadfield.block(0, i * n_ori, n_channels, n_ori) *= weight;
        }
    } else {
        // Fallback: use leadfield norm as proxy for depth
        for (int i = 0; i < n_sources; ++i) {
            Eigen::MatrixXd L_s = leadfield.block(0, i * n_ori, n_channels, n_ori);
            double norm = L_s.norm();
            double weight = std::pow(norm, depth);
            weighted_leadfield.block(0, i * n_ori, n_channels, n_ori) *= weight;
        }
    }
    
    return weighted_leadfield;
}

BeamformerWeights LCMV::compute_matrix_inversion_weights(
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    double reg,
    const std::string& weight_norm,
    int n_ori,
    const std::string& pick_ori)
{
    BeamformerWeights res;
    
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    
    // Regularize Data Covariance
    Covariance reg_cov = data_cov.regularize_scaled(reg);
    
    // Invert Data Covariance
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(reg_cov.data);
    Eigen::MatrixXd C_inv = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    
    // Compute weights for all sources simultaneously using matrix inversion
    // This is more efficient but uses more memory
    
    // For vector case: W = (L^T C^-1 L)^-1 L^T C^-1
    Eigen::MatrixXd LT_Cinv = leadfield.transpose() * C_inv; // (n_dipoles x n_channels)
    Eigen::MatrixXd LT_Cinv_L = LT_Cinv * leadfield; // (n_dipoles x n_dipoles)
    
    // Add regularization to avoid numerical issues
    LT_Cinv_L += 1e-12 * Eigen::MatrixXd::Identity(n_dipoles, n_dipoles);
    
    Eigen::MatrixXd LT_Cinv_L_inv = LT_Cinv_L.inverse();
    Eigen::MatrixXd W_all = LT_Cinv_L_inv * LT_Cinv; // (n_dipoles x n_channels)
    
    // Apply orientation selection and normalization
    res = apply_orientation_selection_and_normalization(
        W_all, leadfield, data_cov, noise_cov, weight_norm, n_ori, pick_ori);
    
    return res;
}

BeamformerWeights LCMV::compute_single_dipole_weights(
    const Eigen::MatrixXd& leadfield,
    const Covariance& data_cov,
    const Covariance& noise_cov,
    double reg,
    const std::string& weight_norm,
    int n_ori,
    const std::string& pick_ori)
{
    BeamformerWeights res;
    
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    
    // Regularize Data Covariance
    Covariance reg_cov = data_cov.regularize_scaled(reg);
    
    // Invert Data Covariance
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(reg_cov.data);
    Eigen::MatrixXd C_inv = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
    
    // Compute weights source by source (more memory efficient)
    Eigen::MatrixXd W_out(n_dipoles, n_channels);
    W_out.setZero();
    
    // Loop over sources
    for (int i = 0; i < n_sources; ++i) {
        Eigen::MatrixXd L_s = leadfield.block(0, i * n_ori, n_channels, n_ori); // (Ch x n_ori)
        
        // Compute weights for this source: W_s = (L_s^T C^-1 L_s)^-1 L_s^T C^-1
        Eigen::MatrixXd num = L_s.transpose() * C_inv; // (n_ori x Ch)
        Eigen::MatrixXd den = num * L_s; // (n_ori x n_ori)
        
        Eigen::MatrixXd den_inv = den.inverse();
        Eigen::MatrixXd W_s = den_inv * num; // (n_ori x Ch)
        
        W_out.block(i * n_ori, 0, n_ori, n_channels) = W_s;
    }
    
    // Apply orientation selection and normalization
    res = apply_orientation_selection_and_normalization(
        W_out, leadfield, data_cov, noise_cov, weight_norm, n_ori, pick_ori);
    
    return res;
}

BeamformerWeights LCMV::apply_ungi_normalization(
    const BeamformerWeights& weights,
    const Eigen::MatrixXd& leadfield,
    const Covariance& noise_cov,
    int n_ori)
{
    BeamformerWeights normalized_weights = weights;
    
    int n_channels = leadfield.rows();
    int n_sources = weights.weights.rows() / (weights.pick_ori == "vector" ? n_ori : 1);
    
    Eigen::MatrixXd N = noise_cov.is_empty ? 
        Eigen::MatrixXd::Identity(n_channels, n_channels) : noise_cov.data;
    
    Eigen::MatrixXd N_inv = N.inverse();
    
    if (weights.pick_ori == "vector") {
        for (int i = 0; i < n_sources; ++i) {
            Eigen::MatrixXd L_i = leadfield.block(0, i * n_ori, n_channels, n_ori);
            
            // Compute G = L^T * N^-1 * L
            Eigen::MatrixXd G_i = L_i.transpose() * N_inv * L_i;
            Eigen::MatrixXd G_inv = G_i.inverse();
            double trace_G_inv = G_inv.trace();
            
            // Apply UNGI normalization
            double norm_factor = std::sqrt(trace_G_inv / n_ori);
            normalized_weights.weights.block(i * n_ori, 0, n_ori, n_channels) *= norm_factor;
        }
    } else {
        for (int i = 0; i < n_sources; ++i) {
            Eigen::MatrixXd L_i = leadfield.block(0, i * n_ori, n_channels, n_ori);
            
            // Compute G = L^T * N^-1 * L
            Eigen::MatrixXd G_i = L_i.transpose() * N_inv * L_i;
            double trace_G = G_i.trace();
            
            // Apply UNGI normalization
            double norm_factor = std::sqrt(trace_G);
            normalized_weights.weights.row(i) *= norm_factor;
        }
    }
    
    return normalized_weights;
}

} // NAMESPACE
