#include "csp.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <cmath>

namespace DECODINGLIB
{

//=============================================================================================================

CSP::CSP(int n_components, bool norm_trace, bool log, bool cov_est)
: m_iNComponents(n_components)
, m_bNormTrace(norm_trace)
, m_bLog(log)
, m_bCovEst(cov_est)
{
}

//=============================================================================================================

Eigen::MatrixXd CSP::computeCovariance(const Eigen::MatrixXd& epoch) const
{
    // epoch is Channels x Time
    // Centering is assumed to be done or not required for band-passed data?
    // CSP usually assumes zero-mean signals (band-passed).
    // We can subtract mean per row just in case.
    
    Eigen::MatrixXd centered = epoch.colwise() - epoch.rowwise().mean();
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(epoch.cols() - 1);
    
    if (m_bNormTrace) {
        cov /= cov.trace();
    }
    
    return cov;
}

//=============================================================================================================

bool CSP::fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels)
{
    if (epochs.empty() || epochs.size() != labels.size()) {
        std::cerr << "CSP::fit: Empty epochs or size mismatch." << std::endl;
        return false;
    }

    int n_channels = epochs[0].rows();
    
    // 1. Compute average covariance matrices for each class
    // Assuming binary classification with labels 0 and 1 (or any two distinct values)
    
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
    // int label_b = unique_labels[1];
    
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
    
    // 2. Solve Generalized Eigenvalue Problem
    // cov_a * w = lambda * (cov_a + cov_b) * w
    // This makes eigenvalues sum to 1 for the two classes if solved properly.
    // Small lambda -> cov_a small, cov_b large (Class B)
    // Large lambda -> cov_a large, cov_b small (Class A)
    
    Eigen::MatrixXd cov_sum = cov_a + cov_b;
    
    // Regularize cov_sum to ensure invertibility?
    // MNE-Python adds a small epsilon * average trace if reg is enabled.
    // For now, we assume data is full rank.
    
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(cov_a, cov_sum);
    
    m_vecEigenValues = es.eigenvalues();
    m_matFilters = es.eigenvectors(); // Columns are eigenvectors
    
    // Calculate Patterns: A = (W^T)^-1 = W^-T = (W^-1)^T
    // Since W^T * C_sum * W = I, W is not necessarily orthogonal, so W^-1 != W^T.
    // A = (W^-1)^T. 
    // In Eigen, we can invert m_matFilters.
    m_matPatterns = m_matFilters.inverse().transpose();
    
    // Transpose filters to match MNE-Python convention if needed?
    // MNE-Python: filters_ (n_channels, n_channels)
    // transform: np.dot(filters_.T, X)
    // So filters_ columns are the filters.
    // Eigen eigenvectors are columns. So m_matFilters matches MNE-Python.
    
    return true;
}

//=============================================================================================================

Eigen::MatrixXd CSP::transform(const std::vector<Eigen::MatrixXd>& epochs) const
{
    if (epochs.empty()) {
        return Eigen::MatrixXd();
    }
    
    int n_epochs = epochs.size();
    int n_channels = m_matFilters.rows();
    
    // Select filters
    // Eigen returns eigenvalues in ascending order.
    // We want the extreme components.
    // n_components usually 4 -> take 0, 1 (Smallest) and N-1, N-2 (Largest)
    
    std::vector<int> pick_indices;
    int n_pick = m_iNComponents;
    if (n_pick > n_channels) n_pick = n_channels;
    
    // Logic to pick indices: [0, 1, ..., k-1] + [N-k, ..., N-1]
    // Note: If n_pick is odd, we might favor one side. MNE-Python usually assumes even or takes equal from both.
    
    int n_bottom = n_pick / 2;
    int n_top = n_pick - n_bottom;
    
    // Eigenvalues are ascending.
    // Index 0: Smallest lambda (Class B dominant)
    // Index N-1: Largest lambda (Class A dominant)
    
    // Indices from top (Largest)
    for (int i = 0; i < n_top; ++i) {
        pick_indices.push_back(n_channels - 1 - i);
    }
    // Indices from bottom (Smallest)
    for (int i = 0; i < n_bottom; ++i) {
        pick_indices.push_back(i);
    }
    
    // Build projection matrix W_pick (n_channels x n_pick)
    Eigen::MatrixXd W_pick(n_channels, n_pick);
    for (int i = 0; i < n_pick; ++i) {
        W_pick.col(i) = m_matFilters.col(pick_indices[i]);
    }
    
    // Features matrix: n_epochs x n_pick
    Eigen::MatrixXd features(n_epochs, n_pick);
    
    for (int i = 0; i < n_epochs; ++i) {
        // Project: Z = W^T * X
        // X is (n_channels, n_times)
        // W_pick is (n_channels, n_pick)
        // Z should be (n_pick, n_times)
        // Z = W_pick.transpose() * X
        
        Eigen::MatrixXd Z = W_pick.transpose() * epochs[i];
        
        // Compute Variance (mean of squared signal, assuming zero mean)
        // Or row-wise variance properly.
        // MNE-Python: (Z**2).mean(axis=1)
        
        for (int k = 0; k < n_pick; ++k) {
            double var = Z.row(k).squaredNorm() / (double)Z.cols(); // mean of squares
            // If we didn't remove mean in Z, this is second moment, not variance.
            // But CSP filters usually output zero-mean signals if input is zero-mean.
            
            if (m_bLog) {
                features(i, k) = std::log(var);
            } else {
                features(i, k) = var;
            }
        }
    }
    
    return features;
}

} // namespace DECODINGLIB
