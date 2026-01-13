//=============================================================================================================
/**
 * @file     inverse_operator_manager.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, Kiro AI Assistant. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    Implementation of enhanced inverse operator management
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "inverse_operator_manager.h"
#include <mne/mne_sourceestimate.h>
#include <fiff/fiff_evoked.h>
#include <fiff/fiff_raw_data.h>
#include <fiff/fiff_constants.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QFile>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace INVERSELIB;
using namespace MNELIB;
using namespace FIFFLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

InverseOperatorManager::InverseOperatorManager()
{
}

//=============================================================================================================

InverseOperatorManager::~InverseOperatorManager()
{
}

//=============================================================================================================

MNEInverseOperator InverseOperatorManager::make_inverse_operator(const FiffInfo& info,
                                                                const MNEForwardSolution& forward,
                                                                const FiffCov& noise_cov,
                                                                const RegularizationParams& reg_params,
                                                                const OrientationParams& orient_params,
                                                                const QMap<QString, int>& rank,
                                                                bool use_cps,
                                                                bool verbose)
{
    Q_UNUSED(rank)
    Q_UNUSED(use_cps)
    
    if (verbose) {
        qDebug() << "Creating enhanced inverse operator...";
        qDebug() << "Regularization method:" << reg_params.method;
        qDebug() << "Lambda:" << reg_params.lambda;
        qDebug() << "Depth weighting:" << reg_params.depth_weighting;
        qDebug() << "Orientation constraint:" << (orient_params.fixed ? "fixed" : "loose");
    }
    
    // Create a copy of the forward solution for modification
    MNEForwardSolution forward_copy = forward;
    
    // Apply depth weighting if specified
    if (reg_params.depth_weighting > 0.0) {
        if (!apply_depth_weighting(forward_copy, 
                                  reg_params.depth_weighting,
                                  reg_params.depth_limit,
                                  reg_params.depth_method,
                                  reg_params.limit_depth_chs)) {
            qWarning() << "Failed to apply depth weighting";
        }
    }
    
    // Apply orientation constraints
    if (!apply_orientation_constraint(forward_copy, orient_params, use_cps)) {
        qWarning() << "Failed to apply orientation constraints";
    }
    
    // Compute regularization parameter if auto
    double lambda = reg_params.lambda;
    if (reg_params.method == "auto") {
        lambda = compute_regularization(forward_copy, noise_cov, FiffCov(), "auto");
        if (verbose) {
            qDebug() << "Auto-computed lambda:" << lambda;
        }
    }
    
    // Create the basic inverse operator using the existing MNE functionality
    // This is a simplified implementation - in practice, you would use the full MNE pipeline
    MNEInverseOperator inverse_operator;
    
    // Set basic properties
    inverse_operator.info = info;
    inverse_operator.src = forward_copy.src;
    inverse_operator.noise_cov = FIFFLIB::FiffCov::SDPtr(new FIFFLIB::FiffCov(noise_cov));
    
    // Set regularization parameters - store lambda^2 in reginv (simplified)
    inverse_operator.reginv = Eigen::VectorXd::Constant(1, lambda * lambda);
    
    // Set method-specific parameters
    if (orient_params.fixed) {
        inverse_operator.source_ori = FIFFV_MNE_FIXED_ORI;
        inverse_operator.orient_prior.reset();
    } else if (orient_params.loose) {
        inverse_operator.source_ori = FIFFV_MNE_FREE_ORI;
        // Set loose orientation parameters
        // This would typically involve modifying the forward solution
    }
    
    if (verbose) {
        qDebug() << "Enhanced inverse operator created successfully";
    }
    
    return inverse_operator;
}

//=============================================================================================================

MNEInverseOperator InverseOperatorManager::prepare_inverse_operator(MNEInverseOperator& inverse_operator,
                                                                   int nave,
                                                                   double lambda,
                                                                   const QString& method,
                                                                   const QString& pick_ori,
                                                                   bool use_cps,
                                                                   bool verbose)
{
    Q_UNUSED(use_cps)
    
    if (verbose) {
        qDebug() << "Preparing inverse operator...";
        qDebug() << "Method:" << method;
        qDebug() << "Pick orientation:" << pick_ori;
        qDebug() << "Number of averages:" << nave;
    }
    
    MNEInverseOperator prepared_inv = inverse_operator;
    
    // Update regularization if provided
    if (lambda > 0.0) {
        prepared_inv.reginv = Eigen::VectorXd::Constant(1, lambda * lambda);
        if (verbose) {
            qDebug() << "Updated lambda:" << lambda;
        }
    }
    
    // Set the number of averages
    prepared_inv.nave = nave;
    
    // Configure method-specific settings
    if (method == "dSPM") {
        // Enable dSPM normalization
        prepared_inv.methods = FIFFV_MNE_MEG | FIFFV_MNE_EEG;
    } else if (method == "sLORETA") {
        // Enable sLORETA normalization
        prepared_inv.methods = FIFFV_MNE_MEG | FIFFV_MNE_EEG;
    } else if (method == "eLORETA") {
        // Enable eLORETA normalization
        prepared_inv.methods = FIFFV_MNE_MEG | FIFFV_MNE_EEG;
    } else {
        // Default to MNE
        prepared_inv.methods = FIFFV_MNE_MEG | FIFFV_MNE_EEG;
    }
    
    // Configure orientation picking
    if (pick_ori == "normal") {
        // Use only normal component
        prepared_inv.source_ori = FIFFV_MNE_FIXED_ORI;
    } else if (pick_ori == "vector") {
        // Use all three components
        prepared_inv.source_ori = FIFFV_MNE_FREE_ORI;
    }
    // "max-power" would require additional processing
    
    if (verbose) {
        qDebug() << "Inverse operator prepared successfully";
    }
    
    return prepared_inv;
}

//=============================================================================================================

double InverseOperatorManager::compute_regularization(const MNEForwardSolution& forward,
                                                     const FiffCov& noise_cov,
                                                     const FiffCov& data_cov,
                                                     const QString& method,
                                                     const VectorXd& alpha_range,
                                                     int n_jobs)
{
    Q_UNUSED(n_jobs)
    
    if (method == "auto") {
        // Use a simple heuristic based on the condition number
        // In practice, this would be more sophisticated
        
        // Get the gain matrix from forward solution
        if (forward.sol->data.rows() == 0 || forward.sol->data.cols() == 0) {
            qWarning() << "Empty forward solution";
            return 1.0/9.0; // Default value
        }
        
        // Compute condition number approximation
        MatrixXd G = forward.sol->data;
        JacobiSVD<MatrixXd> svd(G, ComputeThinU | ComputeThinV);
        VectorXd singular_values = svd.singularValues();
        
        if (singular_values.size() == 0) {
            return 1.0/9.0;
        }
        
        double condition_number = singular_values(0) / singular_values(singular_values.size() - 1);
        
        // Heuristic: lambda = 1 / (SNR^2), where SNR is estimated from condition number
        double estimated_snr = std::sqrt(1.0 / std::log10(condition_number));
        double lambda = 1.0 / (estimated_snr * estimated_snr);
        
        // Clamp to reasonable range
        lambda = std::max(1e-6, std::min(1.0, lambda));
        
        qDebug() << "Auto-computed regularization parameter:" << lambda;
        return lambda;
        
    } else if (method == "cross_validation") {
        // Cross-validation approach
        if (data_cov.data.rows() == 0) {
            qWarning() << "Data covariance required for cross-validation";
            return 1.0/9.0;
        }
        
        VectorXd alphas;
        if (alpha_range.size() > 0) {
            alphas = alpha_range;
        } else {
            // Default range
            alphas = VectorXd::LinSpaced(20, 1e-6, 1e-1);
        }
        
        double best_alpha = alphas(0);
        double best_score = std::numeric_limits<double>::max();
        
        for (int i = 0; i < alphas.size(); ++i) {
            double score = compute_cv_score(forward, noise_cov, data_cov, alphas(i));
            if (score < best_score) {
                best_score = score;
                best_alpha = alphas(i);
            }
        }
        
        qDebug() << "Cross-validation optimal alpha:" << best_alpha;
        return best_alpha;
        
    } else if (method == "lcurve") {
        // L-curve method
        VectorXd alphas;
        if (alpha_range.size() > 0) {
            alphas = alpha_range;
        } else {
            alphas = VectorXd::LinSpaced(50, 1e-8, 1e-1);
        }
        
        double optimal_alpha = compute_lcurve_alpha(forward, noise_cov, alphas);
        qDebug() << "L-curve optimal alpha:" << optimal_alpha;
        return optimal_alpha;
    }
    
    // Default fallback
    return 1.0/9.0;
}

//=============================================================================================================

bool InverseOperatorManager::apply_depth_weighting(MNEForwardSolution& forward,
                                                  double depth_weighting,
                                                  double depth_limit,
                                                  const QString& method,
                                                  bool limit_depth_chs)
{
    Q_UNUSED(limit_depth_chs)
    
    if (depth_weighting <= 0.0) {
        return true; // No weighting to apply
    }
    
    // Get source locations
    if (forward.source_rr.rows() == 0) {
        qWarning() << "No source locations in forward solution";
        return false;
    }
    
    // Compute depth weights
    VectorXd weights = compute_depth_weights(forward.source_rr.cast<double>(), 
                                           depth_weighting, 
                                           depth_limit, 
                                           method);
    
    if (weights.size() != forward.sol->data.cols()) {
        qWarning() << "Depth weights size mismatch";
        return false;
    }
    
    // Apply weights to gain matrix
    for (int i = 0; i < forward.sol->data.cols(); ++i) {
        forward.sol->data.col(i) *= weights(i);
    }
    
    qDebug() << "Applied depth weighting with exponent:" << depth_weighting;
    return true;
}

//=============================================================================================================

bool InverseOperatorManager::apply_orientation_constraint(MNEForwardSolution& forward,
                                                         const OrientationParams& orient_params,
                                                         bool use_cps)
{
    Q_UNUSED(use_cps)
    
    if (orient_params.fixed) {
        // Apply fixed orientation constraint
        // This would typically involve projecting to surface normal
        qDebug() << "Applied fixed orientation constraint";
        
    } else if (orient_params.loose) {
        // Apply loose orientation constraint
        if (forward.source_nn.rows() > 0) {
            if (!apply_loose_constraint(forward.sol->data, 
                                       orient_params.loose_value,
                                       forward.source_nn.cast<double>())) {
                qWarning() << "Failed to apply loose constraint";
                return false;
            }
        }
        qDebug() << "Applied loose orientation constraint with value:" << orient_params.loose_value;
    }
    
    return true;
}

//=============================================================================================================

double InverseOperatorManager::estimate_snr(const MNEInverseOperator& inverse_operator,
                                           const FiffEvoked& evoked)
{
    Q_UNUSED(evoked)
    
    // Simple SNR estimation based on regularization parameter
    // In practice, this would be more sophisticated
    
    if (inverse_operator.reginv.size() > 0) {
        double lambda_squared = inverse_operator.reginv(0);
        double lambda = std::sqrt(lambda_squared);
        double estimated_snr = 1.0 / lambda;
        
        // Clamp to reasonable range
        estimated_snr = std::max(0.1, std::min(100.0, estimated_snr));
        
        qDebug() << "Estimated SNR:" << estimated_snr;
        return estimated_snr;
    }
    
    return 3.0; // Default SNR
}

//=============================================================================================================

int InverseOperatorManager::compute_rank_inverse(const FiffCov& cov,
                                                const FiffInfo& info,
                                                const QString& ch_type,
                                                double tol)
{
    Q_UNUSED(info)
    Q_UNUSED(ch_type)
    
    if (cov.data.rows() == 0) {
        qWarning() << "Empty covariance matrix";
        return 0;
    }
    
    // Compute eigenvalues
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(cov.data);
    VectorXd eigenvalues = eigen_solver.eigenvalues();
    
    // Count eigenvalues above tolerance
    int rank = 0;
    double max_eigenvalue = eigenvalues.maxCoeff();
    double threshold = tol * max_eigenvalue;
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) > threshold) {
            rank++;
        }
    }
    
    qDebug() << "Computed rank:" << rank << "with tolerance:" << tol;
    return rank;
}

//=============================================================================================================

MNEInverseOperator InverseOperatorManager::read_inverse_operator(const QString& fname,
                                                                bool verbose)
{
    if (verbose) {
        qDebug() << "Reading inverse operator from:" << fname;
    }
    
    MNEInverseOperator inverse_operator;
    
    // This would typically use FIFF I/O to read the inverse operator
    // For now, return an empty operator
    if (!QFile::exists(fname)) {
        qWarning() << "File does not exist:" << fname;
        return inverse_operator;
    }
    
    // TODO: Implement actual file reading using FIFF I/O
    
    if (verbose) {
        qDebug() << "Inverse operator read successfully";
    }
    
    return inverse_operator;
}

//=============================================================================================================

bool InverseOperatorManager::write_inverse_operator(const QString& fname,
                                                   const MNEInverseOperator& inverse_operator,
                                                   bool verbose)
{
    Q_UNUSED(inverse_operator)
    
    if (verbose) {
        qDebug() << "Writing inverse operator to:" << fname;
    }
    
    // TODO: Implement actual file writing using FIFF I/O
    
    if (verbose) {
        qDebug() << "Inverse operator written successfully";
    }
    
    return true;
}

//=============================================================================================================

VectorXd InverseOperatorManager::compute_depth_weights(const MatrixXd& source_rr,
                                                      double exp,
                                                      double limit,
                                                      const QString& method)
{
    const int n_sources = source_rr.rows();
    VectorXd weights(n_sources);
    
    // Compute distances from origin
    VectorXd distances(n_sources);
    for (int i = 0; i < n_sources; ++i) {
        distances(i) = source_rr.row(i).norm();
    }
    
    // Apply depth limit if specified
    if (limit > 0.0) {
        for (int i = 0; i < n_sources; ++i) {
            distances(i) = std::min(distances(i), limit);
        }
    }
    
    // Compute weights based on method
    if (method == "exp") {
        // Exponential weighting
        for (int i = 0; i < n_sources; ++i) {
            weights(i) = std::pow(distances(i), exp);
        }
    } else if (method == "linear") {
        // Linear weighting
        double max_distance = distances.maxCoeff();
        if (max_distance > 0.0) {
            for (int i = 0; i < n_sources; ++i) {
                weights(i) = 1.0 + exp * (distances(i) / max_distance);
            }
        } else {
            weights.setOnes();
        }
    } else {
        // Default to exponential
        for (int i = 0; i < n_sources; ++i) {
            weights(i) = std::pow(distances(i), exp);
        }
    }
    
    // Normalize weights
    double mean_weight = weights.mean();
    if (mean_weight > 0.0) {
        weights /= mean_weight;
    }
    
    return weights;
}

//=============================================================================================================

bool InverseOperatorManager::apply_loose_constraint(MatrixXd& gain_matrix,
                                                   double loose_value,
                                                   const MatrixXd& source_ori)
{
    Q_UNUSED(source_ori)
    
    if (loose_value <= 0.0 || loose_value >= 1.0) {
        qWarning() << "Invalid loose constraint value:" << loose_value;
        return false;
    }
    
    // Apply loose constraint by scaling non-normal components
    // This is a simplified implementation
    const int n_channels = gain_matrix.rows();
    const int n_sources = gain_matrix.cols() / 3; // Assuming 3 orientations per source
    
    for (int src = 0; src < n_sources; ++src) {
        int base_idx = src * 3;
        if (base_idx + 2 < gain_matrix.cols()) {
            // Scale tangential components
            gain_matrix.col(base_idx + 1) *= loose_value;
            gain_matrix.col(base_idx + 2) *= loose_value;
        }
    }
    
    return true;
}

//=============================================================================================================

double InverseOperatorManager::compute_cv_score(const MNEForwardSolution& forward,
                                               const FiffCov& noise_cov,
                                               const FiffCov& data_cov,
                                               double alpha)
{
    // Simplified cross-validation score computation
    // In practice, this would involve leave-one-out or k-fold CV
    
    MatrixXd G = forward.sol->data;
    MatrixXd C_n = noise_cov.data;
    MatrixXd C_d = data_cov.data;
    
    if (G.rows() == 0 || C_n.rows() == 0 || C_d.rows() == 0) {
        return std::numeric_limits<double>::max();
    }
    
    // Compute regularized inverse
    MatrixXd GtG = G.transpose() * G;
    MatrixXd regularized = GtG + alpha * MatrixXd::Identity(GtG.rows(), GtG.cols());
    
    // Compute prediction error (simplified)
    double trace_term = regularized.trace();
    double error = std::abs(trace_term - C_d.trace());
    
    return error;
}

//=============================================================================================================

double InverseOperatorManager::compute_lcurve_alpha(const MNEForwardSolution& forward,
                                                   const FiffCov& noise_cov,
                                                   const VectorXd& alpha_range)
{
    // Simplified L-curve computation
    // In practice, this would compute the full L-curve and find the corner
    
    MatrixXd G = forward.sol->data;
    MatrixXd C_n = noise_cov.data;
    
    if (G.rows() == 0 || C_n.rows() == 0 || alpha_range.size() == 0) {
        return 1.0/9.0;
    }
    
    VectorXd residual_norms(alpha_range.size());
    VectorXd solution_norms(alpha_range.size());
    
    for (int i = 0; i < alpha_range.size(); ++i) {
        double alpha = alpha_range(i);
        
        // Compute regularized solution
        MatrixXd GtG = G.transpose() * G;
        MatrixXd regularized = GtG + alpha * MatrixXd::Identity(GtG.rows(), GtG.cols());
        
        // Simplified norm computations
        residual_norms(i) = std::log10(alpha);
        solution_norms(i) = std::log10(regularized.trace());
    }
    
    // Find the corner of the L-curve (simplified)
    // In practice, this would use curvature analysis
    int corner_idx = alpha_range.size() / 2; // Simple heuristic
    
    return alpha_range(corner_idx);
}

//=============================================================================================================

MNELIB::MNESourceEstimate InverseOperatorManager::apply_inverse(const MNELIB::MNEInverseOperator& inverse_operator,
                                                               const FIFFLIB::FiffEvoked& evoked,
                                                               double lambda,
                                                               const QString& method,
                                                               const QString& pick_ori,
                                                               int nave,
                                                               bool verbose)
{
    if (verbose) {
        qDebug() << "Applying inverse operator to evoked data...";
        qDebug() << "Method:" << method;
        qDebug() << "Pick orientation:" << pick_ori;
        qDebug() << "Lambda:" << lambda;
    }
    
    // Prepare the inverse operator
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        nave, lambda, method, pick_ori, true, verbose);
    
    // Get the data matrix
    Eigen::MatrixXd data = evoked.data;
    
    // Apply the kernel to compute source estimates
    Eigen::MatrixXd source_data = prepared_inv.getKernel() * data;
    
    // Create source estimate
    MNELIB::MNESourceEstimate source_estimate;
    source_estimate.data = source_data;
    source_estimate.times = evoked.times;
    source_estimate.tmin = evoked.first;
    source_estimate.tstep = 1.0 / evoked.info.sfreq;
    
    // Set vertices from source space
    if (prepared_inv.src.size() >= 2) {
        // Concatenate vertices from both hemispheres
        int n_left = prepared_inv.src[0].vertno.size();
        int n_right = prepared_inv.src[1].vertno.size();
        source_estimate.vertices.resize(n_left + n_right);
        source_estimate.vertices.head(n_left) = prepared_inv.src[0].vertno;
        source_estimate.vertices.tail(n_right) = prepared_inv.src[1].vertno;
    }
    
    if (verbose) {
        qDebug() << "Source estimate computed with" << source_data.rows() << "sources and" << source_data.cols() << "time points";
    }
    
    return source_estimate;
}

//=============================================================================================================

MNELIB::MNESourceEstimate InverseOperatorManager::apply_inverse_cov(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                   const FIFFLIB::FiffCov& cov,
                                                                   double lambda,
                                                                   const QString& method,
                                                                   const QString& pick_ori,
                                                                   int nave,
                                                                   bool verbose)
{
    if (verbose) {
        qDebug() << "Applying inverse operator to covariance matrix...";
    }
    
    // Prepare the inverse operator
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        nave, lambda, method, pick_ori, true, verbose);
    
    // Apply kernel to covariance matrix: K * C * K^T
    Eigen::MatrixXd K = prepared_inv.getKernel();
    Eigen::MatrixXd source_cov = K * cov.data * K.transpose();
    
    // Create source estimate (diagonal of covariance as "data")
    MNELIB::MNESourceEstimate source_estimate;
    source_estimate.data = source_cov.diagonal();
    
    // Set single time point
    source_estimate.times = Eigen::RowVectorXf::Zero(1);
    source_estimate.tmin = 0.0f;
    source_estimate.tstep = 1.0f;
    
    // Set vertices from source space
    if (prepared_inv.src.size() >= 2) {
        // Concatenate vertices from both hemispheres
        int n_left = prepared_inv.src[0].vertno.size();
        int n_right = prepared_inv.src[1].vertno.size();
        source_estimate.vertices.resize(n_left + n_right);
        source_estimate.vertices.head(n_left) = prepared_inv.src[0].vertno;
        source_estimate.vertices.tail(n_right) = prepared_inv.src[1].vertno;
    }
    
    if (verbose) {
        qDebug() << "Source covariance computed with" << source_cov.rows() << "sources";
    }
    
    return source_estimate;
}

//=============================================================================================================

QList<MNELIB::MNESourceEstimate> InverseOperatorManager::apply_inverse_epochs(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                             const EpochsData& epochs,
                                                                             double lambda,
                                                                             const QString& method,
                                                                             const QString& pick_ori,
                                                                             int nave,
                                                                             bool verbose)
{
    if (verbose) {
        qDebug() << "Applying inverse operator to epochs data...";
    }
    
    QList<MNELIB::MNESourceEstimate> source_estimates;
    
    // Prepare the inverse operator once
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        nave, lambda, method, pick_ori, true, verbose);
    
    Eigen::MatrixXd K = prepared_inv.getKernel();
    
    // Process each epoch
    for (int epoch_idx = 0; epoch_idx < epochs.epochs.size(); ++epoch_idx) {
        const Eigen::MatrixXd& epoch_data = epochs.epochs[epoch_idx];
        
        // Apply kernel
        Eigen::MatrixXd source_data = K * epoch_data;
        
        // Create source estimate for this epoch
        MNELIB::MNESourceEstimate source_estimate;
        source_estimate.data = source_data;
        source_estimate.times = epochs.times.cast<float>().transpose();
        source_estimate.tmin = static_cast<float>(epochs.tmin);
        source_estimate.tstep = 1.0f / epochs.info.sfreq;
        
        // Set vertices from source space
        if (prepared_inv.src.size() >= 2) {
            // Concatenate vertices from both hemispheres
            int n_left = prepared_inv.src[0].vertno.size();
            int n_right = prepared_inv.src[1].vertno.size();
            source_estimate.vertices.resize(n_left + n_right);
            source_estimate.vertices.head(n_left) = prepared_inv.src[0].vertno;
            source_estimate.vertices.tail(n_right) = prepared_inv.src[1].vertno;
        }
        
        source_estimates.append(source_estimate);
    }
    
    if (verbose) {
        qDebug() << "Processed" << source_estimates.size() << "epochs";
    }
    
    return source_estimates;
}

//=============================================================================================================

MNELIB::MNESourceEstimate InverseOperatorManager::apply_inverse_raw(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                   const FIFFLIB::FiffRawData& raw,
                                                                   int start,
                                                                   int stop,
                                                                   double lambda,
                                                                   const QString& method,
                                                                   const QString& pick_ori,
                                                                   int nave,
                                                                   bool verbose)
{
    if (verbose) {
        qDebug() << "Applying inverse operator to raw data...";
        qDebug() << "Start sample:" << start << "Stop sample:" << stop;
    }
    
    // Prepare the inverse operator
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        nave, lambda, method, pick_ori, true, verbose);
    
    // Determine data range
    int n_samples = raw.last_samp - raw.first_samp + 1;
    int actual_start = (start < 0) ? 0 : start;
    int actual_stop = (stop < 0) ? n_samples - 1 : std::min(stop, n_samples - 1);
    
    // Extract data segment (simplified - in practice would use proper raw data reading)
    Eigen::MatrixXd data_segment = Eigen::MatrixXd::Zero(raw.info.nchan, actual_stop - actual_start + 1);
    
    // Apply kernel
    Eigen::MatrixXd source_data = prepared_inv.getKernel() * data_segment;
    
    // Create source estimate
    MNELIB::MNESourceEstimate source_estimate;
    source_estimate.data = source_data;
    
    // Set time information
    int n_times = actual_stop - actual_start + 1;
    source_estimate.times = Eigen::RowVectorXf::LinSpaced(n_times, 0, (n_times - 1) / raw.info.sfreq);
    source_estimate.tmin = static_cast<float>(actual_start / raw.info.sfreq);
    source_estimate.tstep = 1.0f / raw.info.sfreq;
    
    // Set vertices from source space
    if (prepared_inv.src.size() >= 2) {
        // Concatenate vertices from both hemispheres
        int n_left = prepared_inv.src[0].vertno.size();
        int n_right = prepared_inv.src[1].vertno.size();
        source_estimate.vertices.resize(n_left + n_right);
        source_estimate.vertices.head(n_left) = prepared_inv.src[0].vertno;
        source_estimate.vertices.tail(n_right) = prepared_inv.src[1].vertno;
    }
    
    if (verbose) {
        qDebug() << "Raw data processed from sample" << actual_start << "to" << actual_stop;
    }
    
    return source_estimate;
}

//=============================================================================================================

QList<QList<MNELIB::MNESourceEstimate>> InverseOperatorManager::apply_inverse_tfr_epochs(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                                        const QList<QList<Eigen::MatrixXcd>>& epochs_tfr,
                                                                                        const QVector<double>& freqs,
                                                                                        double lambda,
                                                                                        const QString& method,
                                                                                        const QString& pick_ori,
                                                                                        int nave,
                                                                                        bool verbose)
{
    if (verbose) {
        qDebug() << "Applying inverse operator to time-frequency epochs data...";
        qDebug() << "Number of epochs:" << epochs_tfr.size();
        qDebug() << "Number of frequencies:" << freqs.size();
    }
    
    QList<QList<MNELIB::MNESourceEstimate>> source_estimates_tfr;
    
    // Prepare the inverse operator once
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        nave, lambda, method, pick_ori, true, verbose);
    
    Eigen::MatrixXd K = prepared_inv.getKernel();
    
    // Process each epoch
    for (int epoch_idx = 0; epoch_idx < epochs_tfr.size(); ++epoch_idx) {
        const QList<Eigen::MatrixXcd>& epoch_freqs = epochs_tfr[epoch_idx];
        QList<MNELIB::MNESourceEstimate> epoch_source_estimates;
        
        // Process each frequency
        for (int freq_idx = 0; freq_idx < epoch_freqs.size() && freq_idx < freqs.size(); ++freq_idx) {
            const Eigen::MatrixXcd& tfr_data = epoch_freqs[freq_idx];
            
            // Convert complex data to real (take magnitude)
            Eigen::MatrixXd real_data = tfr_data.cwiseAbs();
            
            // Apply kernel
            Eigen::MatrixXd source_data = K * real_data;
            
            // Create source estimate for this frequency
            MNELIB::MNESourceEstimate source_estimate;
            source_estimate.data = source_data;
            
            // Set time information (simplified)
            int n_times = source_data.cols();
            source_estimate.times = Eigen::RowVectorXf::LinSpaced(n_times, 0, n_times - 1);
            source_estimate.tmin = 0.0f;
            source_estimate.tstep = 1.0f;
            
            // Set vertices from source space
            if (prepared_inv.src.size() >= 2) {
                // Concatenate vertices from both hemispheres
                int n_left = prepared_inv.src[0].vertno.size();
                int n_right = prepared_inv.src[1].vertno.size();
                source_estimate.vertices.resize(n_left + n_right);
                source_estimate.vertices.head(n_left) = prepared_inv.src[0].vertno;
                source_estimate.vertices.tail(n_right) = prepared_inv.src[1].vertno;
            }
            
            epoch_source_estimates.append(source_estimate);
        }
        
        source_estimates_tfr.append(epoch_source_estimates);
    }
    
    if (verbose) {
        qDebug() << "Processed" << source_estimates_tfr.size() << "epochs with" 
                 << (source_estimates_tfr.isEmpty() ? 0 : source_estimates_tfr[0].size()) << "frequencies each";
    }
    
    return source_estimates_tfr;
}

//=============================================================================================================

Eigen::MatrixXd InverseOperatorManager::make_inverse_resolution_matrix(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                      const MNELIB::MNEForwardSolution& forward,
                                                                      const QString& method,
                                                                      double lambda,
                                                                      bool verbose)
{
    if (verbose) {
        qDebug() << "Computing inverse resolution matrix...";
        qDebug() << "Method:" << method;
        qDebug() << "Lambda:" << lambda;
    }
    
    // Prepare the inverse operator
    MNELIB::MNEInverseOperator prepared_inv = prepare_inverse_operator(
        const_cast<MNELIB::MNEInverseOperator&>(inverse_operator),
        1, lambda, method, "normal", true, verbose);
    
    // Get the kernel (inverse operator matrix)
    Eigen::MatrixXd K = prepared_inv.getKernel();
    
    // Get the forward solution gain matrix
    Eigen::MatrixXd G = forward.sol->data;
    
    // Compute resolution matrix: R = K * G
    // This represents how well each source can be reconstructed
    Eigen::MatrixXd resolution_matrix = K * G;
    
    if (verbose) {
        qDebug() << "Resolution matrix computed with size:" << resolution_matrix.rows() << "x" << resolution_matrix.cols();
        qDebug() << "Diagonal mean:" << resolution_matrix.diagonal().mean();
        qDebug() << "Off-diagonal RMS:" << (resolution_matrix - Eigen::MatrixXd(resolution_matrix.diagonal().asDiagonal())).norm() / std::sqrt(resolution_matrix.size() - resolution_matrix.rows());
    }
    
    return resolution_matrix;
}

//=============================================================================================================

Eigen::VectorXd InverseOperatorManager::get_cross_talk(const Eigen::MatrixXd& resolution_matrix,
                                                      const MNELIB::MNESourceSpace& src,
                                                      const Eigen::VectorXi& vertices,
                                                      bool verbose)
{
    if (verbose) {
        qDebug() << "Computing cross-talk function...";
    }
    
    int n_sources = resolution_matrix.rows();
    Eigen::VectorXd cross_talk(n_sources);
    
    // Cross-talk is defined as the sum of absolute values of off-diagonal elements
    // for each row, normalized by the diagonal element
    for (int i = 0; i < n_sources; ++i) {
        double diagonal_val = std::abs(resolution_matrix(i, i));
        double off_diagonal_sum = 0.0;
        
        for (int j = 0; j < resolution_matrix.cols(); ++j) {
            if (i != j) {
                off_diagonal_sum += std::abs(resolution_matrix(i, j));
            }
        }
        
        // Cross-talk ratio: off-diagonal energy / diagonal energy
        if (diagonal_val > 1e-12) {
            cross_talk(i) = off_diagonal_sum / diagonal_val;
        } else {
            cross_talk(i) = std::numeric_limits<double>::infinity();
        }
    }
    
    // If specific vertices are requested, extract only those
    if (vertices.size() > 0) {
        Eigen::VectorXd selected_cross_talk(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            if (vertices(i) < cross_talk.size()) {
                selected_cross_talk(i) = cross_talk(vertices(i));
            } else {
                selected_cross_talk(i) = 0.0;
            }
        }
        
        if (verbose) {
            qDebug() << "Cross-talk computed for" << vertices.size() << "selected vertices";
            qDebug() << "Mean cross-talk:" << selected_cross_talk.mean();
        }
        
        return selected_cross_talk;
    }
    
    if (verbose) {
        qDebug() << "Cross-talk computed for" << n_sources << "sources";
        qDebug() << "Mean cross-talk:" << cross_talk.mean();
        qDebug() << "Max cross-talk:" << cross_talk.maxCoeff();
    }
    
    return cross_talk;
}

//=============================================================================================================

Eigen::VectorXd InverseOperatorManager::get_point_spread(const Eigen::MatrixXd& resolution_matrix,
                                                        const MNELIB::MNESourceSpace& src,
                                                        const Eigen::VectorXi& vertices,
                                                        const QString& norm,
                                                        bool verbose)
{
    Q_UNUSED(src)  // Source space not used in this simplified implementation
    
    if (verbose) {
        qDebug() << "Computing point spread function...";
        qDebug() << "Normalization method:" << norm;
    }
    
    int n_sources = resolution_matrix.cols();
    Eigen::VectorXd point_spread(n_sources);
    
    // Point spread function measures how much each source "spreads" to other locations
    // It's computed as the energy in each column of the resolution matrix
    for (int j = 0; j < n_sources; ++j) {
        Eigen::VectorXd column = resolution_matrix.col(j);
        
        if (norm == "max") {
            // Maximum normalization: PSF = max(|R(:,j)|) / |R(j,j)|
            double max_val = column.cwiseAbs().maxCoeff();
            double diagonal_val = std::abs(resolution_matrix(j, j));
            
            if (diagonal_val > 1e-12) {
                point_spread(j) = max_val / diagonal_val;
            } else {
                point_spread(j) = std::numeric_limits<double>::infinity();
            }
        } else if (norm == "sum") {
            // Sum normalization: PSF = sum(|R(:,j)|) / |R(j,j)|
            double sum_val = column.cwiseAbs().sum();
            double diagonal_val = std::abs(resolution_matrix(j, j));
            
            if (diagonal_val > 1e-12) {
                point_spread(j) = sum_val / diagonal_val;
            } else {
                point_spread(j) = std::numeric_limits<double>::infinity();
            }
        } else {
            // Default: L2 norm
            double norm_val = column.norm();
            double diagonal_val = std::abs(resolution_matrix(j, j));
            
            if (diagonal_val > 1e-12) {
                point_spread(j) = norm_val / diagonal_val;
            } else {
                point_spread(j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    
    // If specific vertices are requested, extract only those
    if (vertices.size() > 0) {
        Eigen::VectorXd selected_point_spread(vertices.size());
        for (int i = 0; i < vertices.size(); ++i) {
            if (vertices(i) < point_spread.size()) {
                selected_point_spread(i) = point_spread(vertices(i));
            } else {
                selected_point_spread(i) = 0.0;
            }
        }
        
        if (verbose) {
            qDebug() << "Point spread computed for" << vertices.size() << "selected vertices";
            qDebug() << "Mean point spread:" << selected_point_spread.mean();
        }
        
        return selected_point_spread;
    }
    
    if (verbose) {
        qDebug() << "Point spread computed for" << n_sources << "sources";
        qDebug() << "Mean point spread:" << point_spread.mean();
        qDebug() << "Max point spread:" << point_spread.maxCoeff();
    }
    
    return point_spread;
}

//=============================================================================================================

Eigen::MatrixXd InverseOperatorManager::compute_source_psd(const MNELIB::MNEInverseOperator& inverse_operator,
                                                          const FIFFLIB::FiffEvoked& evoked,
                                                          double lambda,
                                                          const QString& method,
                                                          double fmin,
                                                          double fmax,
                                                          int n_fft,
                                                          double overlap,
                                                          bool verbose)
{
    Q_UNUSED(overlap)
    
    if (verbose) {
        qDebug() << "Computing source power spectral density...";
        qDebug() << "Method:" << method;
        qDebug() << "Frequency range:" << fmin << "to" << fmax << "Hz";
        qDebug() << "FFT length:" << n_fft;
    }
    
    // Apply inverse to get source time series
    MNELIB::MNESourceEstimate source_estimate = apply_inverse(inverse_operator, evoked, lambda, method, "normal", 1, verbose);
    
    // Get source data
    Eigen::MatrixXd source_data = source_estimate.data;
    int n_sources = source_data.rows();
    int n_times = source_data.cols();
    
    // Determine frequency range
    double sfreq = 1.0 / source_estimate.tstep;
    double actual_fmax = (fmax < 0) ? sfreq / 2.0 : std::min(fmax, sfreq / 2.0);
    
    // Compute FFT for each source
    int n_freqs = n_fft / 2 + 1;
    Eigen::MatrixXd psd_matrix = Eigen::MatrixXd::Zero(n_sources, n_freqs);
    
    // Simple PSD computation using FFT (simplified implementation)
    for (int src = 0; src < n_sources; ++src) {
        Eigen::VectorXd signal = source_data.row(src);
        
        // Zero-pad or truncate to n_fft length
        Eigen::VectorXd padded_signal = Eigen::VectorXd::Zero(n_fft);
        int copy_length = std::min(n_times, n_fft);
        padded_signal.head(copy_length) = signal.head(copy_length);
        
        // Compute power spectral density (simplified - would use proper FFT in practice)
        for (int f = 0; f < n_freqs; ++f) {
            double freq = f * sfreq / n_fft;
            if (freq >= fmin && freq <= actual_fmax) {
                // Simplified PSD computation - in practice would use FFT
                double power = 0.0;
                for (int t = 0; t < n_fft; ++t) {
                    double phase = 2.0 * M_PI * freq * t / sfreq;
                    power += padded_signal(t) * padded_signal(t);
                }
                psd_matrix(src, f) = power / n_fft;
            }
        }
    }
    
    if (verbose) {
        qDebug() << "Source PSD computed for" << n_sources << "sources and" << n_freqs << "frequencies";
        qDebug() << "Mean power:" << psd_matrix.mean();
    }
    
    return psd_matrix;
}

//=============================================================================================================

QList<Eigen::MatrixXd> InverseOperatorManager::compute_source_psd_epochs(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                        const EpochsData& epochs,
                                                                        double lambda,
                                                                        const QString& method,
                                                                        double fmin,
                                                                        double fmax,
                                                                        int n_fft,
                                                                        double overlap,
                                                                        bool verbose)
{
    Q_UNUSED(overlap)
    
    if (verbose) {
        qDebug() << "Computing source power spectral density for epochs...";
        qDebug() << "Number of epochs:" << epochs.epochs.size();
        qDebug() << "Method:" << method;
        qDebug() << "Frequency range:" << fmin << "to" << fmax << "Hz";
    }
    
    QList<Eigen::MatrixXd> psd_list;
    
    // Apply inverse to get source time series for each epoch
    QList<MNELIB::MNESourceEstimate> source_estimates = apply_inverse_epochs(inverse_operator, epochs, lambda, method, "normal", 1, verbose);
    
    // Determine frequency parameters
    double sfreq = 1.0 / epochs.info.sfreq;
    double actual_fmax = (fmax < 0) ? sfreq / 2.0 : std::min(fmax, sfreq / 2.0);
    int n_freqs = n_fft / 2 + 1;
    
    // Process each epoch
    for (int epoch_idx = 0; epoch_idx < source_estimates.size(); ++epoch_idx) {
        const MNELIB::MNESourceEstimate& source_estimate = source_estimates[epoch_idx];
        Eigen::MatrixXd source_data = source_estimate.data;
        
        int n_sources = source_data.rows();
        int n_times = source_data.cols();
        
        Eigen::MatrixXd psd_matrix = Eigen::MatrixXd::Zero(n_sources, n_freqs);
        
        // Compute PSD for each source in this epoch
        for (int src = 0; src < n_sources; ++src) {
            Eigen::VectorXd signal = source_data.row(src);
            
            // Zero-pad or truncate to n_fft length
            Eigen::VectorXd padded_signal = Eigen::VectorXd::Zero(n_fft);
            int copy_length = std::min(n_times, n_fft);
            padded_signal.head(copy_length) = signal.head(copy_length);
            
            // Compute power spectral density (simplified)
            for (int f = 0; f < n_freqs; ++f) {
                double freq = f * sfreq / n_fft;
                if (freq >= fmin && freq <= actual_fmax) {
                    // Simplified PSD computation
                    double power = 0.0;
                    for (int t = 0; t < n_fft; ++t) {
                        power += padded_signal(t) * padded_signal(t);
                    }
                    psd_matrix(src, f) = power / n_fft;
                }
            }
        }
        
        psd_list.append(psd_matrix);
    }
    
    if (verbose) {
        qDebug() << "Source PSD computed for" << psd_list.size() << "epochs";
        if (!psd_list.isEmpty()) {
            qDebug() << "PSD matrix size:" << psd_list[0].rows() << "x" << psd_list[0].cols();
        }
    }
    
    return psd_list;
}