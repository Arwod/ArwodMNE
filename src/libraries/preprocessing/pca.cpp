#include "pca.h"
#include <Eigen/Eigenvalues>
#include <iostream>

namespace PREPROCESSINGLIB {

PCA::PCA(int n_components, bool whiten)
    : m_n_components(n_components)
    , m_whiten(whiten)
{
}

void PCA::fit(const Eigen::MatrixXd& data)
{
    // data is (n_channels x n_samples)
    int n_channels = data.rows();
    int n_samples = data.cols();
    
    // 1. Compute Mean
    m_mean = data.rowwise().mean();
    
    // 2. Center data
    Eigen::MatrixXd centered = data.colwise() - m_mean;
    
    // 3. Compute Covariance
    // C = (X * X^T) / (n - 1)
    Eigen::MatrixXd cov = (centered * centered.transpose()) / (double)(n_samples - 1);
    
    // 4. Eigendecomposition
    // SelfAdjointEigenSolver is efficient for symmetric matrices.
    // It returns eigenvalues in ascending order.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    
    Eigen::VectorXd eigenvalues = es.eigenvalues(); // Ascending
    Eigen::MatrixXd eigenvectors = es.eigenvectors(); // Columns are eigenvectors
    
    // 5. Sort descending
    // We want top k components
    int n_eigs = eigenvalues.size();
    
    if (m_n_components <= 0 || m_n_components > n_channels) {
        m_n_components = n_channels;
    }
    
    m_explained_variance.resize(m_n_components);
    m_components.resize(m_n_components, n_channels);
    
    for (int i = 0; i < m_n_components; ++i) {
        // Take from end (largest eigenvalues)
        int idx = n_eigs - 1 - i;
        m_explained_variance[i] = eigenvalues[idx];
        m_components.row(i) = eigenvectors.col(idx);
    }
    
    // 6. Compute Whitening Matrix
    // K = diag(1/sqrt(eigenvalues)) * U^T
    // U^T is m_components
    if (m_whiten) {
        m_whitening_matrix.resize(m_n_components, n_channels);
        for (int i = 0; i < m_n_components; ++i) {
            double lambda = m_explained_variance[i];
            // Avoid division by zero, though lambda should be >= 0 for covariance
            double scale = (lambda > 1e-15) ? 1.0 / std::sqrt(lambda) : 0.0;
            m_whitening_matrix.row(i) = m_components.row(i) * scale;
        }
    } else {
        // If not whitening, "whitening matrix" is just projection matrix?
        // Or identity? Usually PCA transform is just rotation.
        // But to keep API consistent, maybe we shouldn't call it whitening matrix if whiten=false.
        // But for transform(), we usually use this matrix.
        // If whiten=false, transform = U^T * X.
        m_whitening_matrix = m_components;
    }
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& data) const
{
    // X_centered = X - mean
    // result = K * X_centered
    Eigen::MatrixXd centered = data.colwise() - m_mean;
    return m_whitening_matrix * centered;
}

Eigen::MatrixXd PCA::inverse_transform(const Eigen::MatrixXd& data) const
{
    // data is (n_components x n_samples)
    // reconstructed = K_inv * data + mean
    
    // If whitened: K = S^{-0.5} * U^T
    // K_inv = U * S^{0.5}
    // Actually: X_hat = U * S^{0.5} * Y + mean (if Y was whitened)
    // If not whitened: X_hat = U * Y + mean
    
    Eigen::MatrixXd inv_matrix;
    
    if (m_whiten) {
        // m_whitening_matrix row i is u_i / sqrt(lambda_i)
        // We want u_i * sqrt(lambda_i)
        // Since m_components row i is u_i
        // We can reconstruct K_inv manually
        inv_matrix.resize(m_components.cols(), m_components.rows()); // n_channels x n_components
        for(int i=0; i<m_n_components; ++i) {
            inv_matrix.col(i) = m_components.row(i).transpose() * std::sqrt(m_explained_variance[i]);
        }
    } else {
        // U
        inv_matrix = m_components.transpose();
    }
    
    return (inv_matrix * data).colwise() + m_mean;
}

} // NAMESPACE
