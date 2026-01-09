#ifndef COVARIANCE_H
#define COVARIANCE_H

#include "../inverse_global.h"
#include <Eigen/Core>

namespace INVERSELIB {

class INVERSESHARED_EXPORT Covariance
{
public:
    /**
     * Compute empirical covariance matrix.
     * C = (1 / (N - 1)) * (X - mean) * (X - mean)^T
     * 
     * @param[in] data Input data (n_channels x n_samples).
     * @return Covariance matrix (n_channels x n_channels).
     */
    static Eigen::MatrixXd compute_empirical(const Eigen::MatrixXd& data);
    
    /**
     * Regularize covariance matrix.
     * C_reg = C + lambda * I (simplest Tikhonov)
     * 
     * @param[in] cov Input covariance matrix.
     * @param[in] lambda Regularization parameter.
     * @return Regularized covariance matrix.
     */
    static Eigen::MatrixXd regularize(const Eigen::MatrixXd& cov, double lambda);
};

} // NAMESPACE

#endif // COVARIANCE_H
