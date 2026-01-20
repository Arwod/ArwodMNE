#include "gamma_map.h"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace INVERSELIB
{

//=============================================================================================================

GammaMAP::GammaMAP(const Eigen::MatrixXd& leadfield,
                   const Hyperparameters& hyperparams,
                   const ConvergenceCriteria& convergence)
: m_hyperparams(hyperparams)
, m_convergence(convergence)
, m_matLeadfield(leadfield)
, m_dNoiseVariance(hyperparams.noise_variance)
, m_bConverged(false)
, m_iIterations(0)
, m_bFitted(false)
, m_iNSensors(leadfield.rows())
, m_iNSources(leadfield.cols())
, m_iNTimes(0)
{
    // Initialize source variances
    m_vecSourceVariances = Eigen::VectorXd::Constant(m_iNSources, m_hyperparams.alpha / m_hyperparams.beta);
    
    // Initialize precision matrix
    updatePrecisionMatrix();
}

//=============================================================================================================

GammaMAP::Hyperparameters GammaMAP::getDefaultHyperparameters()
{
    Hyperparameters params;
    params.alpha = 1.0;
    params.beta = 1.0;
    params.noise_variance = 1.0;
    params.update_noise = true;
    params.use_ard = true;
    return params;
}

//=============================================================================================================

GammaMAP::ConvergenceCriteria GammaMAP::getDefaultConvergenceCriteria()
{
    ConvergenceCriteria criteria;
    criteria.tolerance = 1e-6;
    criteria.max_iterations = 100;
    criteria.check_cost_function = true;
    criteria.cost_tolerance = 1e-8;
    return criteria;
}

//=============================================================================================================

bool GammaMAP::fit(const Eigen::MatrixXd& data)
{
    if (data.rows() != m_iNSensors) {
        std::cerr << "GammaMAP::fit: Data dimensions mismatch. Expected " 
                  << m_iNSensors << " sensors, got " << data.rows() << std::endl;
        return false;
    }
    
    m_iNTimes = data.cols();
    m_vecCostHistory.clear();
    m_bConverged = false;
    m_iIterations = 0;
    
    // Initialize posterior mean and covariance
    m_matPosteriorMean = Eigen::MatrixXd::Zero(m_iNSources, m_iNTimes);
    
    double cost_old = std::numeric_limits<double>::infinity();
    
    std::cout << "Starting Gamma-MAP optimization..." << std::endl;
    
    for (int iter = 0; iter < m_convergence.max_iterations; ++iter) {
        double cost_new = performEMIteration(data);
        m_vecCostHistory.push_back(cost_new);
        
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << ", Cost: " << cost_new << std::endl;
        }
        
        // Check convergence
        if (checkConvergence(cost_new, cost_old)) {
            m_bConverged = true;
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        cost_old = cost_new;
        m_iIterations = iter + 1;
    }
    
    if (!m_bConverged) {
        std::cout << "Warning: Algorithm did not converge after " 
                  << m_convergence.max_iterations << " iterations." << std::endl;
    }
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

Eigen::MatrixXd GammaMAP::apply(const Eigen::MatrixXd& data) const
{
    if (!m_bFitted) {
        std::cerr << "GammaMAP::apply: Model not fitted. Call fit() first." << std::endl;
        return Eigen::MatrixXd();
    }
    
    if (data.rows() != m_iNSensors) {
        std::cerr << "GammaMAP::apply: Data dimensions mismatch." << std::endl;
        return Eigen::MatrixXd();
    }
    
    // Compute posterior mean: mu = Sigma * G^T * (noise_var)^-1 * data
    // where Sigma is the posterior covariance
    
    Eigen::MatrixXd source_estimates(m_iNSources, data.cols());
    
    // Efficient computation using the precision matrix
    Eigen::MatrixXd temp = m_matLeadfield.transpose() * data / m_dNoiseVariance;
    source_estimates = m_matPosteriorCov * temp;
    
    return source_estimates;
}

//=============================================================================================================

Eigen::MatrixXd GammaMAP::fitApply(const Eigen::MatrixXd& data)
{
    if (!fit(data)) {
        return Eigen::MatrixXd();
    }
    return apply(data);
}

//=============================================================================================================

double GammaMAP::getSparsityLevel(double threshold) const
{
    if (!m_bFitted) {
        return 0.0;
    }
    
    int sparse_count = 0;
    for (int i = 0; i < m_vecSourceVariances.size(); ++i) {
        if (m_vecSourceVariances(i) < threshold) {
            sparse_count++;
        }
    }
    
    return (double)sparse_count / (double)m_vecSourceVariances.size();
}

//=============================================================================================================

double GammaMAP::performEMIteration(const Eigen::MatrixXd& data)
{
    // E-step: Update posterior statistics
    eStep(data);
    
    // M-step: Update hyperparameters
    mStep(data);
    
    // Update precision matrix with new hyperparameters
    updatePrecisionMatrix();
    
    // Compute and return cost function
    return computeCostFunction(data);
}

//=============================================================================================================

void GammaMAP::eStep(const Eigen::MatrixXd& data)
{
    // Update posterior covariance and mean
    computePosteriorCovariance();
    
    // Compute posterior mean: mu = Sigma * G^T * (noise_var)^-1 * data
    Eigen::MatrixXd temp = m_matLeadfield.transpose() * data / m_dNoiseVariance;
    m_matPosteriorMean = m_matPosteriorCov * temp;
}

//=============================================================================================================

void GammaMAP::mStep(const Eigen::MatrixXd& data)
{
    // Update source variances (hyperparameters)
    if (m_hyperparams.use_ard) {
        // Automatic Relevance Determination update
        for (int i = 0; i < m_iNSources; ++i) {
            // Update based on posterior mean and covariance
            double posterior_var = m_matPosteriorCov(i, i);
            double posterior_mean_sq = m_matPosteriorMean.row(i).squaredNorm() / m_iNTimes;
            
            // Gamma prior update
            double new_variance = (m_hyperparams.alpha + m_iNTimes / 2.0) / 
                                 (m_hyperparams.beta + 0.5 * (posterior_mean_sq + posterior_var));
            
            m_vecSourceVariances(i) = std::max(new_variance, 1e-12); // Avoid numerical issues
        }
    }
    
    // Update noise variance if requested
    if (m_hyperparams.update_noise) {
        Eigen::MatrixXd residual = data - m_matLeadfield * m_matPosteriorMean;
        double residual_norm = residual.squaredNorm();
        
        // Add trace term for posterior covariance
        double trace_term = (m_matLeadfield * m_matPosteriorCov * m_matLeadfield.transpose()).trace();
        
        m_dNoiseVariance = (residual_norm + trace_term) / (m_iNSensors * m_iNTimes);
        m_dNoiseVariance = std::max(m_dNoiseVariance, 1e-12); // Avoid numerical issues
    }
}

//=============================================================================================================

double GammaMAP::computeCostFunction(const Eigen::MatrixXd& data) const
{
    // Compute negative log-likelihood
    double cost = 0.0;
    
    // Data likelihood term
    Eigen::MatrixXd residual = data - m_matLeadfield * m_matPosteriorMean;
    cost += residual.squaredNorm() / (2.0 * m_dNoiseVariance);
    
    // Log determinant terms
    cost += 0.5 * m_iNTimes * m_iNSensors * std::log(2.0 * M_PI * m_dNoiseVariance);
    
    // Prior terms (Gamma priors on source variances)
    for (int i = 0; i < m_iNSources; ++i) {
        if (m_vecSourceVariances(i) > 1e-12) {
            cost += 0.5 * m_iNTimes * std::log(2.0 * M_PI * m_vecSourceVariances(i));
            cost += m_matPosteriorMean.row(i).squaredNorm() / (2.0 * m_vecSourceVariances(i));
            cost += 0.5 * m_matPosteriorCov(i, i) / m_vecSourceVariances(i);
        }
    }
    
    // Hyperprior terms
    for (int i = 0; i < m_iNSources; ++i) {
        if (m_vecSourceVariances(i) > 1e-12) {
            cost -= (m_hyperparams.alpha - 1.0) * std::log(m_vecSourceVariances(i));
            cost += m_hyperparams.beta * m_vecSourceVariances(i);
        }
    }
    
    return cost;
}

//=============================================================================================================

bool GammaMAP::checkConvergence(double cost_new, double cost_old) const
{
    if (m_iIterations == 0) {
        return false;
    }
    
    // Check relative change in cost function
    if (m_convergence.check_cost_function) {
        double relative_change = std::abs(cost_new - cost_old) / std::abs(cost_old);
        if (relative_change < m_convergence.cost_tolerance) {
            return true;
        }
    }
    
    // Check change in hyperparameters (could be added here)
    
    return false;
}

//=============================================================================================================

void GammaMAP::updatePrecisionMatrix()
{
    // Precision matrix: Lambda = diag(1/source_variances)
    m_matPrecision = Eigen::MatrixXd::Zero(m_iNSources, m_iNSources);
    
    for (int i = 0; i < m_iNSources; ++i) {
        if (m_vecSourceVariances(i) > 1e-12) {
            m_matPrecision(i, i) = 1.0 / m_vecSourceVariances(i);
        } else {
            m_matPrecision(i, i) = 1e12; // Very large precision for near-zero variance
        }
    }
}

//=============================================================================================================

void GammaMAP::computePosteriorCovariance()
{
    // Posterior covariance: Sigma = (G^T * G / noise_var + Lambda)^-1
    // where Lambda is the precision matrix (diagonal)
    
    Eigen::MatrixXd gram_matrix = m_matLeadfield.transpose() * m_matLeadfield / m_dNoiseVariance;
    Eigen::MatrixXd precision_total = gram_matrix + m_matPrecision;
    
    // Compute inverse using LU decomposition
    try {
        m_matPosteriorCov = precision_total.inverse();
    } catch (const std::exception& e) {
        std::cerr << "GammaMAP::computePosteriorCovariance: Failed to invert precision matrix: " 
                  << e.what() << std::endl;
        
        // Fallback: Use pseudo-inverse or regularization
        Eigen::MatrixXd regularized = precision_total + 1e-6 * Eigen::MatrixXd::Identity(m_iNSources, m_iNSources);
        m_matPosteriorCov = regularized.inverse();
    }
}

//=============================================================================================================
// Convenience function
//=============================================================================================================

Eigen::MatrixXd gamma_map(const Eigen::MatrixXd& leadfield,
                         const Eigen::MatrixXd& data,
                         const GammaMAP::Hyperparameters& hyperparams,
                         const GammaMAP::ConvergenceCriteria& convergence)
{
    GammaMAP solver(leadfield, hyperparams, convergence);
    return solver.fitApply(data);
}

} // namespace INVERSELIB