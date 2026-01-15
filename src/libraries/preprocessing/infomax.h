#ifndef INFOMAX_H
#define INFOMAX_H

#include "preprocessing_global.h"
#include <Eigen/Core>
#include <string>

namespace PREPROCESSINGLIB {

/**
 * @brief Infomax ICA algorithm implementation
 * 
 * Implements the Infomax algorithm for Independent Component Analysis
 * based on Bell & Sejnowski (1995) and extended Infomax.
 */
class PREPROCESSINGSHARED_EXPORT Infomax
{
public:
    Infomax(int n_components = 0,
            bool extended = true,
            double learning_rate = 0.001,
            int max_iter = 200,
            double tol = 1e-6,
            int random_state = 0);

    /**
     * Fit Infomax ICA on whitened data.
     * 
     * @param[in] data Whitened input data (n_components x n_samples).
     */
    void fit(const Eigen::MatrixXd& data);
    
    /**
     * Transform data using learned unmixing matrix.
     * 
     * @param[in] data Input data (n_components x n_samples).
     * @return Independent components (n_components x n_samples).
     */
    Eigen::MatrixXd transform(const Eigen::MatrixXd& data) const;
    
    /**
     * Inverse transform from sources back to mixed data.
     * 
     * @param[in] sources Independent components (n_components x n_samples).
     * @return Reconstructed mixed data (n_components x n_samples).
     */
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& sources) const;

    // Getters
    Eigen::MatrixXd get_unmixing_matrix() const { return m_W; }
    Eigen::MatrixXd get_mixing_matrix() const { return m_A; }
    
    // Check if algorithm converged
    bool converged() const { return m_converged; }
    int get_n_iter() const { return m_n_iter; }

private:
    /**
     * Sigmoid activation function for standard Infomax.
     */
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) const;
    
    /**
     * Extended Infomax activation function.
     */
    Eigen::MatrixXd extended_activation(const Eigen::MatrixXd& x, const Eigen::VectorXd& signs) const;
    
    /**
     * Update signs for extended Infomax based on kurtosis.
     */
    Eigen::VectorXd update_signs(const Eigen::MatrixXd& sources) const;
    
    /**
     * Compute kurtosis for each component.
     */
    Eigen::VectorXd compute_kurtosis(const Eigen::MatrixXd& sources) const;

    int m_n_components;
    bool m_extended;           // Use extended Infomax
    double m_learning_rate;
    int m_max_iter;
    double m_tol;
    int m_random_state;
    
    Eigen::MatrixXd m_W;       // Unmixing matrix
    Eigen::MatrixXd m_A;       // Mixing matrix (inverse of W)
    
    bool m_converged;
    int m_n_iter;
};

} // NAMESPACE

#endif // INFOMAX_H