#include "spatial_filtering.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace DECODINGLIB
{

//=============================================================================================================
// BaseSpatialFilter Implementation
//=============================================================================================================

std::vector<Eigen::MatrixXd> BaseSpatialFilter::fitTransform(const std::vector<Eigen::MatrixXd>& epochs)
{
    if (!fit(epochs)) {
        return std::vector<Eigen::MatrixXd>();
    }
    return transform(epochs);
}

//=============================================================================================================
// SupervisedSpatialFilter Implementation
//=============================================================================================================

bool SupervisedSpatialFilter::fit(const std::vector<Eigen::MatrixXd>& epochs)
{
    // Default implementation: call supervised fit with empty labels
    std::vector<int> empty_labels;
    return fit(epochs, empty_labels);
}

//=============================================================================================================

std::vector<Eigen::MatrixXd> SupervisedSpatialFilter::fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                                                  const std::vector<int>& labels)
{
    if (!fit(epochs, labels)) {
        return std::vector<Eigen::MatrixXd>();
    }
    return transform(epochs);
}

//=============================================================================================================
// UnsupervisedSpatialFilter Implementation
//=============================================================================================================

UnsupervisedSpatialFilter::UnsupervisedSpatialFilter(Method method, int n_components, double reg_param)
: m_method(method)
, m_iNComponents(n_components)
, m_dRegParam(reg_param)
, m_bFitted(false)
{
}

//=============================================================================================================

bool UnsupervisedSpatialFilter::fit(const std::vector<Eigen::MatrixXd>& epochs)
{
    if (epochs.empty()) {
        std::cerr << "UnsupervisedSpatialFilter::fit: Empty epochs." << std::endl;
        return false;
    }
    
    switch (m_method) {
        case Method::PCA:
            return fitPCA(epochs);
        case Method::Whitening:
            return fitWhitening(epochs);
        case Method::Laplacian:
            return fitLaplacian(epochs);
        case Method::ICA:
            std::cerr << "UnsupervisedSpatialFilter::fit: ICA not implemented yet." << std::endl;
            return false;
        default:
            std::cerr << "UnsupervisedSpatialFilter::fit: Unknown method." << std::endl;
            return false;
    }
}

//=============================================================================================================

std::vector<Eigen::MatrixXd> UnsupervisedSpatialFilter::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (!m_bFitted || epochs.empty()) {
        return std::vector<Eigen::MatrixXd>();
    }
    
    std::vector<Eigen::MatrixXd> filtered_epochs;
    filtered_epochs.reserve(epochs.size());
    
    for (const auto& epoch : epochs) {
        // Apply spatial filter: W^T * X
        Eigen::MatrixXd filtered = m_matFilters.transpose() * epoch;
        filtered_epochs.push_back(filtered);
    }
    
    return filtered_epochs;
}

//=============================================================================================================

bool UnsupervisedSpatialFilter::fitPCA(const std::vector<Eigen::MatrixXd>& epochs)
{
    // Compute covariance matrix
    Eigen::MatrixXd cov = computeCovarianceMatrix(epochs);
    int n_channels = cov.rows();
    
    // Limit components to available channels
    int n_components = std::min(m_iNComponents, n_channels);
    
    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "UnsupervisedSpatialFilter::fitPCA: Eigenvalue decomposition failed." << std::endl;
        return false;
    }
    
    // Eigenvalues are in ascending order, we want the largest
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    
    // Select top components
    m_matFilters = Eigen::MatrixXd(n_channels, n_components);
    m_vecExplainedVariance = Eigen::VectorXd(n_components);
    
    double total_variance = eigenvalues.sum();
    
    for (int i = 0; i < n_components; ++i) {
        int idx = n_channels - 1 - i; // Reverse order for largest eigenvalues
        m_matFilters.col(i) = eigenvectors.col(idx);
        m_vecExplainedVariance(i) = eigenvalues(idx) / total_variance;
    }
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

bool UnsupervisedSpatialFilter::fitWhitening(const std::vector<Eigen::MatrixXd>& epochs)
{
    // Compute covariance matrix
    Eigen::MatrixXd cov = computeCovarianceMatrix(epochs);
    int n_channels = cov.rows();
    
    // Add regularization
    if (m_dRegParam > 0.0) {
        double trace = cov.trace();
        cov += m_dRegParam * trace / n_channels * Eigen::MatrixXd::Identity(n_channels, n_channels);
    }
    
    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "UnsupervisedSpatialFilter::fitWhitening: Eigenvalue decomposition failed." << std::endl;
        return false;
    }
    
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    
    // Compute whitening matrix: W = D^(-1/2) * V^T
    // where D is diagonal matrix of eigenvalues, V is eigenvectors
    Eigen::VectorXd inv_sqrt_eigenvalues = eigenvalues.cwiseSqrt().cwiseInverse();
    
    m_matFilters = eigenvectors * inv_sqrt_eigenvalues.asDiagonal() * eigenvectors.transpose();
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

bool UnsupervisedSpatialFilter::fitLaplacian(const std::vector<Eigen::MatrixXd>& epochs)
{
    if (epochs.empty()) {
        return false;
    }
    
    int n_channels = epochs[0].rows();
    
    // Simple Laplacian filter (nearest neighbor approximation)
    // This is a simplified version - in practice, you'd use electrode positions
    m_matFilters = Eigen::MatrixXd::Identity(n_channels, n_channels);
    
    // Apply simple discrete Laplacian (center - average of neighbors)
    for (int i = 1; i < n_channels - 1; ++i) {
        m_matFilters(i, i-1) = -0.25;
        m_matFilters(i, i) = 1.0;
        m_matFilters(i, i+1) = -0.25;
    }
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

Eigen::MatrixXd UnsupervisedSpatialFilter::computeCovarianceMatrix(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (epochs.empty()) {
        return Eigen::MatrixXd();
    }
    
    int n_channels = epochs[0].rows();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n_channels, n_channels);
    int total_samples = 0;
    
    // Compute mean across all epochs
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_channels);
    for (const auto& epoch : epochs) {
        mean += epoch.rowwise().mean();
        total_samples += epoch.cols();
    }
    mean /= epochs.size();
    
    // Compute covariance
    for (const auto& epoch : epochs) {
        Eigen::MatrixXd centered = epoch.colwise() - mean;
        cov += centered * centered.transpose();
    }
    
    cov /= (total_samples - 1);
    return cov;
}

//=============================================================================================================
// XdawnSpatialFilter Implementation
//=============================================================================================================

XdawnSpatialFilter::XdawnSpatialFilter(int n_components, double reg_param)
: m_iNComponents(n_components)
, m_dRegParam(reg_param)
, m_bFitted(false)
{
}

//=============================================================================================================

bool XdawnSpatialFilter::fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels)
{
    if (epochs.empty() || epochs.size() != labels.size()) {
        std::cerr << "XdawnSpatialFilter::fit: Empty epochs or size mismatch." << std::endl;
        return false;
    }
    
    int n_channels = epochs[0].rows();
    int n_times = epochs[0].cols();
    
    // Separate target and non-target epochs
    std::vector<Eigen::MatrixXd> target_epochs;
    std::vector<Eigen::MatrixXd> nontarget_epochs;
    
    for (size_t i = 0; i < epochs.size(); ++i) {
        if (labels[i] == 1) {
            target_epochs.push_back(epochs[i]);
        } else {
            nontarget_epochs.push_back(epochs[i]);
        }
    }
    
    if (target_epochs.empty()) {
        std::cerr << "XdawnSpatialFilter::fit: No target epochs found." << std::endl;
        return false;
    }
    
    // Compute average evoked response for targets
    m_matEvoked = Eigen::MatrixXd::Zero(n_channels, n_times);
    for (const auto& epoch : target_epochs) {
        m_matEvoked += epoch;
    }
    m_matEvoked /= target_epochs.size();
    
    // Compute covariance matrices
    Eigen::MatrixXd cov_signal = Eigen::MatrixXd::Zero(n_channels, n_channels);
    Eigen::MatrixXd cov_noise = Eigen::MatrixXd::Zero(n_channels, n_channels);
    
    // Signal covariance (from evoked response)
    cov_signal = m_matEvoked * m_matEvoked.transpose();
    
    // Noise covariance (from all epochs)
    for (const auto& epoch : epochs) {
        Eigen::MatrixXd centered = epoch.colwise() - epoch.rowwise().mean();
        cov_noise += centered * centered.transpose();
    }
    cov_noise /= (epochs.size() * (n_times - 1));
    
    // Add regularization
    if (m_dRegParam > 0.0) {
        double trace = cov_noise.trace();
        cov_noise += m_dRegParam * trace / n_channels * Eigen::MatrixXd::Identity(n_channels, n_channels);
    }
    
    // Solve generalized eigenvalue problem
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(cov_signal, cov_noise);
    
    if (es.info() != Eigen::Success) {
        std::cerr << "XdawnSpatialFilter::fit: Eigenvalue decomposition failed." << std::endl;
        return false;
    }
    
    // Select top components
    int n_components = std::min(m_iNComponents, n_channels);
    m_matFilters = Eigen::MatrixXd(n_channels, n_components);
    
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    
    // Take largest eigenvalues (ascending order, so take from end)
    for (int i = 0; i < n_components; ++i) {
        int idx = n_channels - 1 - i;
        m_matFilters.col(i) = eigenvectors.col(idx);
    }
    
    // Compute patterns
    try {
        m_matPatterns = m_matFilters.inverse().transpose();
    } catch (const std::exception& e) {
        std::cerr << "XdawnSpatialFilter::fit: Failed to compute patterns: " << e.what() << std::endl;
        return false;
    }
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

std::vector<Eigen::MatrixXd> XdawnSpatialFilter::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (!m_bFitted || epochs.empty()) {
        return std::vector<Eigen::MatrixXd>();
    }
    
    std::vector<Eigen::MatrixXd> filtered_epochs;
    filtered_epochs.reserve(epochs.size());
    
    for (const auto& epoch : epochs) {
        // Apply spatial filter: W^T * X
        Eigen::MatrixXd filtered = m_matFilters.transpose() * epoch;
        filtered_epochs.push_back(filtered);
    }
    
    return filtered_epochs;
}

//=============================================================================================================
// SurfaceLaplacianFilter Implementation
//=============================================================================================================

SurfaceLaplacianFilter::SurfaceLaplacianFilter(int m_order, double lambda)
: m_iOrder(m_order)
, m_dLambda(lambda)
, m_bFitted(false)
{
}

//=============================================================================================================

bool SurfaceLaplacianFilter::fit(const std::vector<Eigen::MatrixXd>& epochs)
{
    if (epochs.empty()) {
        std::cerr << "SurfaceLaplacianFilter::fit: Empty epochs." << std::endl;
        return false;
    }
    
    int n_channels = epochs[0].rows();
    
    if (m_matPositions.rows() != n_channels) {
        std::cerr << "SurfaceLaplacianFilter::fit: Electrode positions not set or size mismatch." << std::endl;
        return false;
    }
    
    computeLaplacianMatrix();
    m_bFitted = true;
    return true;
}

//=============================================================================================================

std::vector<Eigen::MatrixXd> SurfaceLaplacianFilter::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (!m_bFitted || epochs.empty()) {
        return std::vector<Eigen::MatrixXd>();
    }
    
    std::vector<Eigen::MatrixXd> filtered_epochs;
    filtered_epochs.reserve(epochs.size());
    
    for (const auto& epoch : epochs) {
        // Apply Laplacian filter
        Eigen::MatrixXd filtered = m_matLaplacian * epoch;
        filtered_epochs.push_back(filtered);
    }
    
    return filtered_epochs;
}

//=============================================================================================================

void SurfaceLaplacianFilter::setElectrodePositions(const Eigen::MatrixXd& positions)
{
    m_matPositions = positions;
}

//=============================================================================================================

void SurfaceLaplacianFilter::computeLaplacianMatrix()
{
    int n_channels = m_matPositions.rows();
    
    // Compute spline interpolation matrix
    Eigen::MatrixXd G = computeSplineMatrix(m_matPositions);
    
    // Add regularization
    G += m_dLambda * Eigen::MatrixXd::Identity(n_channels, n_channels);
    
    // Compute Laplacian matrix
    try {
        m_matLaplacian = G.inverse();
    } catch (const std::exception& e) {
        std::cerr << "SurfaceLaplacianFilter::computeLaplacianMatrix: Failed to invert matrix: " << e.what() << std::endl;
        // Fallback to identity
        m_matLaplacian = Eigen::MatrixXd::Identity(n_channels, n_channels);
    }
}

//=============================================================================================================

Eigen::MatrixXd SurfaceLaplacianFilter::computeSplineMatrix(const Eigen::MatrixXd& positions) const
{
    int n_channels = positions.rows();
    Eigen::MatrixXd G(n_channels, n_channels);
    
    // Compute spherical spline interpolation matrix
    for (int i = 0; i < n_channels; ++i) {
        for (int j = 0; j < n_channels; ++j) {
            if (i == j) {
                G(i, j) = 1.0;
            } else {
                // Compute distance between electrodes
                Eigen::Vector3d diff = positions.row(i) - positions.row(j);
                double dist = diff.norm();
                
                // Spherical spline function (simplified)
                if (dist > 1e-12) {
                    G(i, j) = std::pow(dist, m_iOrder);
                } else {
                    G(i, j) = 0.0;
                }
            }
        }
    }
    
    return G;
}

} // namespace DECODINGLIB