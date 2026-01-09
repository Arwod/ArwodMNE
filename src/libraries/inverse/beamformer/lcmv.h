#ifndef LCMV_H
#define LCMV_H

#include "../inverse_global.h"
#include <Eigen/Core>
#include <vector>

namespace INVERSELIB {

class INVERSESHARED_EXPORT LCMV
{
public:
    /**
     * Compute LCMV Spatial Filter weights.
     * 
     * @param[in] leadfield Forward solution leadfield (n_channels x n_sources).
     *                      Usually n_sources is small (e.g. 3 for vector dipole at one location, or 1 for fixed).
     *                      If computing for whole brain, this function is usually called per source location.
     * @param[in] data_cov Data covariance matrix (n_channels x n_channels).
     * @param[in] reg Regularization parameter (e.g. 0.05 * trace(C) / n).
     * @param[in] pick_ori Orientation selection ("vector", "max-power", "normal").
     *                     For now, assume "vector" (keep all source components).
     * @return Weights (n_sources x n_channels).
     */
    static Eigen::MatrixXd compute_weights(const Eigen::MatrixXd& leadfield, 
                                           const Eigen::MatrixXd& data_cov, 
                                           double reg = 0.05);
    
    /**
     * Apply LCMV weights to data.
     * 
     * @param[in] weights Spatial filter weights (n_sources x n_channels).
     * @param[in] data Input data (n_channels x n_times).
     * @return Source estimates (n_sources x n_times).
     */
    static Eigen::MatrixXd apply(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& data);
};

} // NAMESPACE

#endif // LCMV_H
