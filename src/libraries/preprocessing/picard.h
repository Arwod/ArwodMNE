#ifndef PICARD_H
#define PICARD_H

#include "preprocessing_global.h"
#include <Eigen/Core>
#include <string>

namespace PREPROCESSINGLIB {

/**
 * @brief Picard ICA algorithm implementation
 * 
 * Implements the Picard algorithm for Independent Component Analysis
 * based on Ablin et al. (2018) - faster convergence than FastICA.
 */
class PREPROCESSINGSHARED_EXPORT Picard
{
public:
    Picard(int n_components = 0,
           const std::string& fun = "logcosh",
           double ortho = true,
           int max_iter = 100,
           double tol = 1e-7,
           int random_state = 0);

    /**
     * Fit Picard ICA on whitened data.
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
     * Compute the contrast function and its derivatives.
     */
    void compute_contrast(const Eigen::MatrixXd& Y, 
                         Eigen::MatrixXd& psi, 
                         Eigen::VectorXd& psiy) const;
    
    /**
     * Orthogonalization step using Gram-Schmidt.
     */
    Eigen::MatrixXd orthogonalize(const Eigen::MatrixXd& W) const;
    
    /**
     * Line search for optimal step size.
     */
    double line_search(const Eigen::MatrixXd& W, 
                      const Eigen::MatrixXd& direction,
                      const Eigen::MatrixXd& data) const;
    
    /**
     * Compute objective function value.
     */
    double compute_objective(const Eigen::MatrixXd& W, const Eigen::MatrixXd& data) const;

    int m_n_components;
    std::string m_fun;         // Contrast function: "logcosh", "exp", "cube"
    bool m_ortho;              // Use orthogonalization
    int m_max_iter;
    double m_tol;
    int m_random_state;
    
    Eigen::MatrixXd m_W;       // Unmixing matrix
    Eigen::MatrixXd m_A;       // Mixing matrix (inverse of W)
    
    bool m_converged;
    int m_n_iter;
    
    // Contrast function parameters
    double m_alpha;            // Parameter for logcosh
};

} // NAMESPACE

#endif // PICARD_H