#ifndef COVARIANCE_H
#define COVARIANCE_H

#include "../inverse_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>

namespace INVERSELIB {

class INVERSESHARED_EXPORT Covariance
{
public:
    Covariance();
    Covariance(const Eigen::MatrixXd& data, const std::vector<std::string>& names = {});
    ~Covariance() = default;

    // Data Members
    Eigen::MatrixXd data;           // The covariance matrix (n_chan x n_chan)
    std::vector<std::string> names; // Channel names
    int nfree;                      // Degrees of freedom (number of samples used)
    double loglik;                  // Log-likelihood (optional)
    std::string method;             // Method used (e.g., "empirical")
    bool is_empty;                  // Flag for empty covariance

    /**
     * Compute empirical covariance matrix from a single data matrix.
     * C = (1 / (N - 1)) * (X - mean) * (X - mean)^T
     * 
     * @param[in] data Input data (n_channels x n_samples).
     * @return Covariance object.
     */
    static Covariance compute_empirical(const Eigen::MatrixXd& data, const std::vector<std::string>& names = {});

    /**
     * Compute empirical covariance matrix from multiple epochs.
     * 
     * @param[in] epochs Vector of data matrices (n_channels x n_times).
     * @param[in] names Channel names.
     * @return Covariance object.
     */
    static Covariance compute_empirical(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<std::string>& names = {});
    
    /**
     * Regularize covariance matrix.
     * C_reg = C + lambda * I (simplest Tikhonov)
     * 
     * @param[in] lambda Regularization parameter (absolute value added to diagonal).
     * @return New regularized Covariance object.
     */
    Covariance regularize(double lambda) const;
    
    /**
     * Regularize covariance matrix using scaling factor.
     * lambda = reg * average_eigenvalue
     * 
     * @param[in] reg Regularization scaling factor (e.g. 0.05).
     * @return New regularized Covariance object.
     */
    Covariance regularize_scaled(double reg) const;
};

} // NAMESPACE

#endif // COVARIANCE_H
