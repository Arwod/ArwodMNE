#ifndef PCA_H
#define PCA_H

#include "preprocessing_global.h"
#include <Eigen/Core>
#include <Eigen/SVD>

namespace PREPROCESSINGLIB {

class PREPROCESSINGSHARED_EXPORT PCA
{
public:
    PCA(int n_components = 0, bool whiten = true);
    
    /**
     * Fit PCA on data.
     * 
     * @param[in] data Input data (n_channels x n_times).
     *                 PCA is performed over channels (spatial PCA).
     */
    void fit(const Eigen::MatrixXd& data);
    
    /**
     * Transform data (Project to principal components).
     * 
     * @param[in] data Input data (n_channels x n_times).
     * @return Transformed data (n_components x n_times).
     */
    Eigen::MatrixXd transform(const Eigen::MatrixXd& data) const;
    
    /**
     * Inverse transform data.
     * 
     * @param[in] data Transformed data (n_components x n_times).
     * @return Reconstructed data (n_channels x n_times).
     */
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& data) const;
    
    // Getters
    Eigen::MatrixXd get_components() const { return m_components; } // (n_components x n_channels)
    Eigen::VectorXd get_explained_variance() const { return m_explained_variance; }
    Eigen::MatrixXd get_whitening_matrix() const { return m_whitening_matrix; } // (n_components x n_channels)

private:
    int m_n_components;
    bool m_whiten;
    
    Eigen::VectorXd m_mean;             // (n_channels)
    Eigen::MatrixXd m_components;       // (n_components x n_channels) - Eigenvectors (U^T)
    Eigen::VectorXd m_explained_variance;
    Eigen::MatrixXd m_whitening_matrix; // K
};

} // NAMESPACE

#endif // PCA_H
