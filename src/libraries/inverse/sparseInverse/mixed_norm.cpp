#include "mixed_norm.h"
#include <Eigen/LU>
#include <Eigen/QR>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace INVERSELIB
{

//=============================================================================================================

MixedNorm::MixedNorm(const Eigen::MatrixXd& leadfield,
                     const OptimizationParams& opt_params,
                     const TFParams& tf_params)
: m_optParams(opt_params)
, m_tfParams(tf_params)
, m_matLeadfield(leadfield)
, m_matLeadfieldOrig(leadfield)
, m_bConverged(false)
, m_iIterations(0)
, m_iNSensors(leadfield.rows())
, m_iNSources(leadfield.cols())
, m_iNTimes(0)
{
    // Normalize leadfield if requested
    if (m_optParams.normalize) {
        normalizeLeadfield();
    }
}

//=============================================================================================================

MixedNorm::OptimizationParams MixedNorm::getDefaultOptimizationParams()
{
    OptimizationParams params;
    params.alpha = 0.1;
    params.l1_ratio = 0.5;
    params.max_iterations = 1000;
    params.tolerance = 1e-6;
    params.normalize = true;
    params.positive = false;
    params.step_size = 1.0;
    params.adaptive_step = true;
    return params;
}

//=============================================================================================================

MixedNorm::TFParams MixedNorm::getDefaultTFParams()
{
    TFParams params;
    params.use_tf_mixed_norm = false;
    params.tf_alpha = 0.01;
    params.n_tfmxne_iter = 10;
    params.debias = true;
    params.solver = "auto";
    return params;
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::solve(const Eigen::MatrixXd& data)
{
    if (data.rows() != m_iNSensors) {
        std::cerr << "MixedNorm::solve: Data dimensions mismatch." << std::endl;
        return Eigen::MatrixXd();
    }
    
    m_iNTimes = data.cols();
    m_vecCostHistory.clear();
    m_bConverged = false;
    m_iIterations = 0;
    
    Eigen::MatrixXd solution;
    
    // Choose solver based on problem size and parameters
    if (m_tfParams.solver == "cd" || 
        (m_tfParams.solver == "auto" && m_iNSources < 1000)) {
        solution = solveCoordinateDescent(data);
    } else if (m_tfParams.solver == "bcd" || 
               (m_tfParams.solver == "auto" && m_iNSources >= 1000)) {
        solution = solveBlockCoordinateDescent(data);
    } else {
        solution = solveProximalGradient(data);
    }
    
    // Debias if requested
    if (m_tfParams.debias) {
        solution = debiasolution(solution, data);
    }
    
    return solution;
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::solveTF(const Eigen::MatrixXd& data)
{
    // Time-frequency mixed norm implementation
    m_tfParams.use_tf_mixed_norm = true;
    
    Eigen::MatrixXd solution = solve(data);
    
    // Additional TF-MxNE iterations
    for (int iter = 0; iter < m_tfParams.n_tfmxne_iter; ++iter) {
        // Update weights based on current solution
        Eigen::VectorXd weights = Eigen::VectorXd::Ones(m_iNSources);
        
        for (int i = 0; i < m_iNSources; ++i) {
            double norm = solution.row(i).norm();
            if (norm > 1e-12) {
                weights(i) = 1.0 / (norm + m_tfParams.tf_alpha);
            }
        }
        
        // Solve weighted problem
        Eigen::MatrixXd weighted_leadfield = m_matLeadfield;
        for (int i = 0; i < m_iNSources; ++i) {
            weighted_leadfield.col(i) *= weights(i);
        }
        
        // Update leadfield temporarily
        Eigen::MatrixXd temp_leadfield = m_matLeadfield;
        m_matLeadfield = weighted_leadfield;
        
        solution = solve(data);
        
        // Scale back solution
        for (int i = 0; i < m_iNSources; ++i) {
            solution.row(i) *= weights(i);
        }
        
        // Restore original leadfield
        m_matLeadfield = temp_leadfield;
    }
    
    return solution;
}

//=============================================================================================================

double MixedNorm::getSparsityLevel(double threshold) const
{
    if (m_matSolution.size() == 0) {
        return 0.0;
    }
    
    int sparse_count = 0;
    for (int i = 0; i < m_iNSources; ++i) {
        if (m_matSolution.row(i).norm() < threshold) {
            sparse_count++;
        }
    }
    
    return (double)sparse_count / (double)m_iNSources;
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::solveCoordinateDescent(const Eigen::MatrixXd& data)
{
    std::cout << "Using coordinate descent solver..." << std::endl;
    
    // Initialize solution
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(m_iNSources, m_iNTimes);
    
    // Precompute Gram matrix
    Eigen::MatrixXd G = m_matLeadfield.transpose() * m_matLeadfield;
    Eigen::MatrixXd Gy = m_matLeadfield.transpose() * data;
    
    // Extract diagonal for efficient updates
    Eigen::VectorXd G_diag = G.diagonal();
    
    for (int iter = 0; iter < m_optParams.max_iterations; ++iter) {
        Eigen::MatrixXd X_old = X;
        
        // Coordinate descent updates
        for (int j = 0; j < m_iNSources; ++j) {
            if (G_diag(j) < 1e-12) continue;
            
            // Compute residual for source j
            Eigen::VectorXd residual = Gy.row(j).transpose();
            for (int k = 0; k < m_iNSources; ++k) {
                if (k != j) {
                    residual -= G(j, k) * X.row(k).transpose();
                }
            }
            
            // Update with soft thresholding
            double threshold = m_optParams.alpha * m_optParams.l1_ratio;
            for (int t = 0; t < m_iNTimes; ++t) {
                double update = residual(t) / G_diag(j);
                X(j, t) = softThreshold(update, threshold);
                
                // Enforce positivity if requested
                if (m_optParams.positive && X(j, t) < 0) {
                    X(j, t) = 0.0;
                }
            }
        }
        
        // Check convergence
        if (checkConvergence(X, X_old)) {
            m_bConverged = true;
            std::cout << "Coordinate descent converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        // Compute and store cost
        double cost = computeCostFunction(data, X);
        m_vecCostHistory.push_back(cost);
        
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Cost: " << cost << std::endl;
        }
        
        m_iIterations = iter + 1;
    }
    
    m_matSolution = X;
    return X;
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::solveBlockCoordinateDescent(const Eigen::MatrixXd& data)
{
    std::cout << "Using block coordinate descent solver..." << std::endl;
    
    // Initialize solution
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(m_iNSources, m_iNTimes);
    
    // Precompute matrices
    Eigen::MatrixXd G = m_matLeadfield.transpose() * m_matLeadfield;
    Eigen::MatrixXd Gy = m_matLeadfield.transpose() * data;
    
    for (int iter = 0; iter < m_optParams.max_iterations; ++iter) {
        Eigen::MatrixXd X_old = X;
        
        // Block coordinate descent updates
        for (int j = 0; j < m_iNSources; ++j) {
            // Compute residual for source j
            Eigen::VectorXd residual = Gy.row(j).transpose();
            for (int k = 0; k < m_iNSources; ++k) {
                if (k != j) {
                    residual -= G(j, k) * X.row(k).transpose();
                }
            }
            
            // Group soft thresholding
            double threshold = m_optParams.alpha * m_optParams.l1_ratio;
            Eigen::VectorXd update = residual / G(j, j);
            X.row(j) = groupSoftThreshold(update, threshold).transpose();
            
            // Enforce positivity if requested
            if (m_optParams.positive) {
                X.row(j) = X.row(j).cwiseMax(0.0);
            }
        }
        
        // Check convergence
        if (checkConvergence(X, X_old)) {
            m_bConverged = true;
            std::cout << "Block coordinate descent converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        // Compute and store cost
        double cost = computeCostFunction(data, X);
        m_vecCostHistory.push_back(cost);
        
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Cost: " << cost << std::endl;
        }
        
        m_iIterations = iter + 1;
    }
    
    m_matSolution = X;
    return X;
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::solveProximalGradient(const Eigen::MatrixXd& data)
{
    std::cout << "Using proximal gradient solver..." << std::endl;
    
    // Initialize solution
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(m_iNSources, m_iNTimes);
    
    // Precompute matrices
    Eigen::MatrixXd G = m_matLeadfield.transpose() * m_matLeadfield;
    Eigen::MatrixXd Gy = m_matLeadfield.transpose() * data;
    
    // Estimate Lipschitz constant more robustly
    double L = G.diagonal().maxCoeff(); // Use max diagonal element
    if (L < 1e-12) {
        L = 1.0;
    }
    double step_size = 0.9 / L; // Use 0.9 for stability
    
    for (int iter = 0; iter < m_optParams.max_iterations; ++iter) {
        Eigen::MatrixXd X_old = X;
        
        // Gradient step
        Eigen::MatrixXd gradient = G * X - Gy;
        Eigen::MatrixXd Y = X - step_size * gradient;
        
        // Proximal operator (soft thresholding)
        double threshold = step_size * m_optParams.alpha * m_optParams.l1_ratio;
        for (int i = 0; i < m_iNSources; ++i) {
            for (int t = 0; t < m_iNTimes; ++t) {
                X(i, t) = softThreshold(Y(i, t), threshold);
                
                // Enforce positivity if requested
                if (m_optParams.positive && X(i, t) < 0) {
                    X(i, t) = 0.0;
                }
            }
        }
        
        // Adaptive step size
        if (m_optParams.adaptive_step && iter > 0) {
            double cost_new = computeCostFunction(data, X);
            double cost_old = m_vecCostHistory.back();
            
            if (cost_new > cost_old) {
                step_size *= 0.8; // Decrease step size
            } else if (iter % 10 == 0) {
                step_size *= 1.1; // Increase step size occasionally
                step_size = std::min(step_size, m_optParams.step_size / L);
            }
        }
        
        // Check convergence
        if (checkConvergence(X, X_old)) {
            m_bConverged = true;
            std::cout << "Proximal gradient converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        // Compute and store cost
        double cost = computeCostFunction(data, X);
        m_vecCostHistory.push_back(cost);
        
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Cost: " << cost 
                      << ", Step size: " << step_size << std::endl;
        }
        
        m_iIterations = iter + 1;
    }
    
    m_matSolution = X;
    return X;
}

//=============================================================================================================

double MixedNorm::softThreshold(double x, double threshold) const
{
    if (x > threshold) {
        return x - threshold;
    } else if (x < -threshold) {
        return x + threshold;
    } else {
        return 0.0;
    }
}

//=============================================================================================================

Eigen::VectorXd MixedNorm::groupSoftThreshold(const Eigen::VectorXd& x, double threshold) const
{
    double norm = x.norm();
    if (norm <= threshold) {
        return Eigen::VectorXd::Zero(x.size());
    } else {
        return x * (1.0 - threshold / norm);
    }
}

//=============================================================================================================

double MixedNorm::computeElasticNetPenalty(const Eigen::MatrixXd& X) const
{
    double l1_penalty = 0.0;
    double l2_penalty = 0.0;
    
    for (int i = 0; i < m_iNSources; ++i) {
        double row_norm = X.row(i).norm();
        l1_penalty += row_norm; // Group L1 norm
        l2_penalty += row_norm * row_norm; // Group L2 norm squared
    }
    
    return m_optParams.alpha * (m_optParams.l1_ratio * l1_penalty + 
                               0.5 * (1.0 - m_optParams.l1_ratio) * l2_penalty);
}

//=============================================================================================================

double MixedNorm::computeCostFunction(const Eigen::MatrixXd& data, const Eigen::MatrixXd& X) const
{
    // Data fidelity term
    Eigen::MatrixXd residual = data - m_matLeadfield * X;
    double data_fidelity = 0.5 * residual.squaredNorm();
    
    // Regularization term
    double regularization = computeElasticNetPenalty(X);
    
    return data_fidelity + regularization;
}

//=============================================================================================================

bool MixedNorm::checkConvergence(const Eigen::MatrixXd& X_new, const Eigen::MatrixXd& X_old) const
{
    if (m_iIterations == 0) {
        return false;
    }
    
    // Check relative change in solution
    double diff_norm = (X_new - X_old).norm();
    double solution_norm = X_old.norm();
    
    if (solution_norm < 1e-12) {
        return diff_norm < m_optParams.tolerance;
    }
    
    double relative_change = diff_norm / solution_norm;
    return relative_change < m_optParams.tolerance;
}

//=============================================================================================================

void MixedNorm::normalizeLeadfield()
{
    m_vecNormFactors = Eigen::VectorXd::Zero(m_iNSources);
    
    for (int i = 0; i < m_iNSources; ++i) {
        double norm = m_matLeadfield.col(i).norm();
        if (norm > 1e-12) {
            m_vecNormFactors(i) = norm;
            m_matLeadfield.col(i) /= norm;
        } else {
            m_vecNormFactors(i) = 1.0;
        }
    }
}

//=============================================================================================================

Eigen::MatrixXd MixedNorm::debiasolution(const Eigen::MatrixXd& X, const Eigen::MatrixXd& data) const
{
    // Find active sources (non-zero)
    std::vector<int> active_sources;
    double threshold = 1e-6;
    
    for (int i = 0; i < m_iNSources; ++i) {
        if (X.row(i).norm() > threshold) {
            active_sources.push_back(i);
        }
    }
    
    if (active_sources.empty()) {
        return X;
    }
    
    // Create reduced leadfield with active sources only
    Eigen::MatrixXd G_active(m_iNSensors, active_sources.size());
    for (size_t i = 0; i < active_sources.size(); ++i) {
        G_active.col(i) = m_matLeadfield.col(active_sources[i]);
    }
    
    // Solve least squares problem for active sources
    Eigen::MatrixXd X_debiased = Eigen::MatrixXd::Zero(m_iNSources, m_iNTimes);
    
    try {
        // Use QR decomposition for numerical stability
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(G_active);
        Eigen::MatrixXd X_active = qr.solve(data);
        
        // Put back into full solution
        for (size_t i = 0; i < active_sources.size(); ++i) {
            X_debiased.row(active_sources[i]) = X_active.row(i);
        }
    } catch (const std::exception& e) {
        std::cerr << "MixedNorm::debiasolution: Failed to debias: " << e.what() << std::endl;
        return X; // Return original solution
    }
    
    return X_debiased;
}

//=============================================================================================================
// Convenience functions
//=============================================================================================================

Eigen::MatrixXd mixed_norm(const Eigen::MatrixXd& leadfield,
                          const Eigen::MatrixXd& data,
                          double alpha,
                          double l1_ratio,
                          const MixedNorm::OptimizationParams& opt_params)
{
    MixedNorm::OptimizationParams params = opt_params;
    params.alpha = alpha;
    params.l1_ratio = l1_ratio;
    
    MixedNorm solver(leadfield, params);
    return solver.solve(data);
}

//=============================================================================================================

Eigen::MatrixXd tf_mixed_norm(const Eigen::MatrixXd& leadfield,
                             const Eigen::MatrixXd& data,
                             double alpha,
                             double l1_ratio,
                             const MixedNorm::TFParams& tf_params)
{
    MixedNorm::OptimizationParams opt_params;
    opt_params.alpha = 0.1;
    opt_params.l1_ratio = 0.5;
    opt_params.max_iterations = 1000;
    opt_params.tolerance = 1e-6;
    opt_params.normalize = true;
    opt_params.positive = false;
    opt_params.step_size = 1.0;
    opt_params.adaptive_step = true;
    
    opt_params.alpha = alpha;
    opt_params.l1_ratio = l1_ratio;
    
    MixedNorm::TFParams params = tf_params;
    params.use_tf_mixed_norm = true;
    
    MixedNorm solver(leadfield, opt_params, params);
    return solver.solveTF(data);
}

} // namespace INVERSELIB