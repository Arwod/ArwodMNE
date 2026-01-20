#include "csp.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace DECODINGLIB
{

//=============================================================================================================
// CSP Implementation
//=============================================================================================================

CSP::CSP(int n_components, bool norm_trace, bool log, RegularizationMethod reg_method, double reg_param)
: m_iNComponents(n_components)
, m_bNormTrace(norm_trace)
, m_bLog(log)
, m_regMethod(reg_method)
, m_dRegParam(reg_param)
, m_strComponentSelection("extremes")
{
}

//=============================================================================================================

Eigen::MatrixXd CSP::computeCovariance(const Eigen::MatrixXd& epoch) const
{
    // epoch is Channels x Time
    // Centering is assumed to be done or not required for band-passed data
    Eigen::MatrixXd centered = epoch.colwise() - epoch.rowwise().mean();
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(epoch.cols() - 1);
    
    if (m_bNormTrace) {
        double trace = cov.trace();
        if (trace > 1e-12) {
            cov /= trace;
        }
    }
    
    return regularizeCovariance(cov);
}

//=============================================================================================================

Eigen::MatrixXd CSP::regularizeCovariance(const Eigen::MatrixXd& cov) const
{
    Eigen::MatrixXd reg_cov = cov;
    
    switch (m_regMethod) {
        case RegularizationMethod::None:
            break;
            
        case RegularizationMethod::Diagonal:
        {
            double trace = cov.trace();
            reg_cov += m_dRegParam * trace / cov.rows() * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
            break;
        }
        
        case RegularizationMethod::Shrinkage:
        {
            double trace = cov.trace();
            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
            reg_cov = (1.0 - m_dRegParam) * cov + m_dRegParam * (trace / cov.rows()) * identity;
            break;
        }
        
        case RegularizationMethod::LedoitWolf:
        {
            // Simplified Ledoit-Wolf shrinkage
            double trace = cov.trace();
            double mu = trace / cov.rows();
            
            // Estimate optimal shrinkage parameter (simplified version)
            double alpha2 = 0.0;
            double delta = 0.0;
            
            for (int i = 0; i < cov.rows(); ++i) {
                for (int j = 0; j < cov.cols(); ++j) {
                    if (i == j) {
                        delta += (cov(i, j) - mu) * (cov(i, j) - mu);
                    } else {
                        alpha2 += cov(i, j) * cov(i, j);
                        delta += cov(i, j) * cov(i, j);
                    }
                }
            }
            
            double shrinkage = std::min(1.0, alpha2 / delta);
            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
            reg_cov = (1.0 - shrinkage) * cov + shrinkage * mu * identity;
            break;
        }
    }
    
    return reg_cov;
}

//=============================================================================================================

std::vector<int> CSP::selectComponents(const Eigen::VectorXd& eigenvalues) const
{
    std::vector<int> indices;
    int n_channels = eigenvalues.size();
    int n_pick = std::min(m_iNComponents, n_channels);
    
    if (m_strComponentSelection == "extremes") {
        // Take components from both extremes
        int n_bottom = n_pick / 2;
        int n_top = n_pick - n_bottom;
        
        // Eigenvalues are in ascending order
        // Top eigenvalues (largest)
        for (int i = 0; i < n_top; ++i) {
            indices.push_back(n_channels - 1 - i);
        }
        // Bottom eigenvalues (smallest)
        for (int i = 0; i < n_bottom; ++i) {
            indices.push_back(i);
        }
    } else if (m_strComponentSelection == "top") {
        // Take only top eigenvalues
        for (int i = 0; i < n_pick; ++i) {
            indices.push_back(n_channels - 1 - i);
        }
    } else if (m_strComponentSelection == "bottom") {
        // Take only bottom eigenvalues
        for (int i = 0; i < n_pick; ++i) {
            indices.push_back(i);
        }
    }
    
    return indices;
}

//=============================================================================================================

bool CSP::fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels)
{
    if (epochs.empty() || epochs.size() != labels.size()) {
        std::cerr << "CSP::fit: Empty epochs or size mismatch." << std::endl;
        return false;
    }

    int n_channels = epochs[0].rows();
    
    // Find unique labels
    std::vector<int> unique_labels = labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    
    if (unique_labels.size() != 2) {
        std::cerr << "CSP::fit: Only binary classification is supported. Found " << unique_labels.size() << " classes." << std::endl;
        return false;
    }
    
    Eigen::MatrixXd cov_a = Eigen::MatrixXd::Zero(n_channels, n_channels);
    Eigen::MatrixXd cov_b = Eigen::MatrixXd::Zero(n_channels, n_channels);
    int n_a = 0;
    int n_b = 0;
    
    int label_a = unique_labels[0];
    
    for (size_t i = 0; i < epochs.size(); ++i) {
        if (epochs[i].rows() != n_channels) {
            std::cerr << "CSP::fit: Epoch " << i << " has wrong channel count." << std::endl;
            return false;
        }
        
        Eigen::MatrixXd cov = computeCovariance(epochs[i]);
        
        if (labels[i] == label_a) {
            cov_a += cov;
            n_a++;
        } else {
            cov_b += cov;
            n_b++;
        }
    }
    
    if (n_a > 0) cov_a /= (double)n_a;
    if (n_b > 0) cov_b /= (double)n_b;
    
    // Solve Generalized Eigenvalue Problem
    Eigen::MatrixXd cov_sum = cov_a + cov_b;
    
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(cov_a, cov_sum);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "CSP::fit: Eigenvalue decomposition failed." << std::endl;
        return false;
    }
    
    m_vecEigenValues = es.eigenvalues();
    m_matFilters = es.eigenvectors();
    
    // Select components
    m_vecSelectedIndices = selectComponents(m_vecEigenValues);
    
    // Calculate patterns
    try {
        m_matPatterns = m_matFilters.inverse().transpose();
    } catch (const std::exception& e) {
        std::cerr << "CSP::fit: Failed to compute patterns: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

//=============================================================================================================

Eigen::MatrixXd CSP::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (epochs.empty() || m_vecSelectedIndices.empty()) {
        return Eigen::MatrixXd();
    }
    
    int n_epochs = epochs.size();
    int n_components = m_vecSelectedIndices.size();
    
    // Build projection matrix
    Eigen::MatrixXd W_selected(m_matFilters.rows(), n_components);
    for (int i = 0; i < n_components; ++i) {
        W_selected.col(i) = m_matFilters.col(m_vecSelectedIndices[i]);
    }
    
    Eigen::MatrixXd features(n_epochs, n_components);
    
    for (int i = 0; i < n_epochs; ++i) {
        // Project data
        Eigen::MatrixXd Z = W_selected.transpose() * epochs[i];
        
        // Compute variance for each component
        for (int k = 0; k < n_components; ++k) {
            double var = Z.row(k).squaredNorm() / (double)Z.cols();
            
            if (m_bLog) {
                features(i, k) = std::log(std::max(var, 1e-12)); // Avoid log(0)
            } else {
                features(i, k) = var;
            }
        }
    }
    
    return features;
}

//=============================================================================================================

Eigen::MatrixXd CSP::fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                 const std::vector<int>& labels)
{
    if (!fit(epochs, labels)) {
        return Eigen::MatrixXd();
    }
    return transform(epochs);
}

//=============================================================================================================
// SPoC Implementation
//=============================================================================================================

SPoC::SPoC(int n_components, bool norm_trace, bool log, double reg_param)
: m_iNComponents(n_components)
, m_bNormTrace(norm_trace)
, m_bLog(log)
, m_dRegParam(reg_param)
{
}

//=============================================================================================================

Eigen::MatrixXd SPoC::computeCovariance(const Eigen::MatrixXd& epoch) const
{
    Eigen::MatrixXd centered = epoch.colwise() - epoch.rowwise().mean();
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(epoch.cols() - 1);
    
    if (m_bNormTrace) {
        double trace = cov.trace();
        if (trace > 1e-12) {
            cov /= trace;
        }
    }
    
    // Add regularization
    if (m_dRegParam > 0.0) {
        double trace = cov.trace();
        cov += m_dRegParam * trace / cov.rows() * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
    }
    
    return cov;
}

//=============================================================================================================

Eigen::MatrixXd SPoC::computeWeightedCovariance(const std::vector<Eigen::MatrixXd>& epochs,
                                               const Eigen::VectorXd& weights) const
{
    if (epochs.empty() || weights.size() != epochs.size()) {
        return Eigen::MatrixXd();
    }
    
    int n_channels = epochs[0].rows();
    Eigen::MatrixXd weighted_cov = Eigen::MatrixXd::Zero(n_channels, n_channels);
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < epochs.size(); ++i) {
        Eigen::MatrixXd cov = computeCovariance(epochs[i]);
        weighted_cov += weights(i) * cov;
        weight_sum += weights(i);
    }
    
    if (weight_sum > 1e-12) {
        weighted_cov /= weight_sum;
    }
    
    return weighted_cov;
}

//=============================================================================================================

bool SPoC::fit(const std::vector<Eigen::MatrixXd>& epochs, const Eigen::VectorXd& target)
{
    if (epochs.empty() || target.size() != epochs.size()) {
        std::cerr << "SPoC::fit: Empty epochs or size mismatch." << std::endl;
        return false;
    }
    
    int n_channels = epochs[0].rows();
    
    // Normalize target values to zero mean and unit variance
    double target_mean = target.mean();
    Eigen::VectorXd centered_target = target.array() - target_mean;
    double target_std = std::sqrt(centered_target.squaredNorm() / (target.size() - 1));
    
    if (target_std < 1e-12) {
        std::cerr << "SPoC::fit: Target has zero variance." << std::endl;
        return false;
    }
    
    Eigen::VectorXd normalized_target = centered_target / target_std;
    
    // Compute average covariance matrix
    Eigen::MatrixXd cov_avg = Eigen::MatrixXd::Zero(n_channels, n_channels);
    for (size_t i = 0; i < epochs.size(); ++i) {
        cov_avg += computeCovariance(epochs[i]);
    }
    cov_avg /= (double)epochs.size();
    
    // Compute target-weighted covariance matrix
    Eigen::VectorXd weights = normalized_target.array().abs();
    Eigen::MatrixXd cov_weighted = computeWeightedCovariance(epochs, weights);
    
    // Solve generalized eigenvalue problem
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(cov_weighted, cov_avg);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "SPoC::fit: Eigenvalue decomposition failed." << std::endl;
        return false;
    }
    
    m_vecEigenValues = es.eigenvalues();
    m_matFilters = es.eigenvectors();
    
    // Take the top components (largest eigenvalues)
    // Eigenvalues are in ascending order, so take the last n_components
    int n_pick = std::min(m_iNComponents, n_channels);
    Eigen::MatrixXd selected_filters(n_channels, n_pick);
    for (int i = 0; i < n_pick; ++i) {
        selected_filters.col(i) = m_matFilters.col(n_channels - 1 - i);
    }
    m_matFilters = selected_filters;
    
    // Calculate patterns
    try {
        m_matPatterns = m_matFilters.inverse().transpose();
    } catch (const std::exception& e) {
        std::cerr << "SPoC::fit: Failed to compute patterns: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

//=============================================================================================================

Eigen::MatrixXd SPoC::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (epochs.empty() || m_matFilters.cols() == 0) {
        return Eigen::MatrixXd();
    }
    
    int n_epochs = epochs.size();
    int n_components = m_matFilters.cols();
    
    Eigen::MatrixXd features(n_epochs, n_components);
    
    for (int i = 0; i < n_epochs; ++i) {
        // Project data
        Eigen::MatrixXd Z = m_matFilters.transpose() * epochs[i];
        
        // Compute variance for each component
        for (int k = 0; k < n_components; ++k) {
            double var = Z.row(k).squaredNorm() / (double)Z.cols();
            
            if (m_bLog) {
                features(i, k) = std::log(std::max(var, 1e-12)); // Avoid log(0)
            } else {
                features(i, k) = var;
            }
        }
    }
    
    return features;
}

//=============================================================================================================

Eigen::MatrixXd SPoC::fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                  const Eigen::VectorXd& target)
{
    if (!fit(epochs, target)) {
        return Eigen::MatrixXd();
    }
    return transform(epochs);
}

} // namespace DECODINGLIB
