#ifndef LCMV_H
#define LCMV_H

#include "../inverse_global.h"
#include "covariance.h"
#include <Eigen/Core>
#include <vector>
#include <string>

namespace INVERSELIB {

struct INVERSESHARED_EXPORT BeamformerWeights {
    Eigen::MatrixXd weights; // (n_sources x n_channels)
    std::vector<std::string> ch_names;
    std::string pick_ori;
    bool weight_norm;
    Eigen::MatrixXd whitening_mat; // Optional whitening matrix used
    
    BeamformerWeights() : weight_norm(false) {}
};

class INVERSESHARED_EXPORT LCMV
{
public:
    /**
     * @brief Compute LCMV beamformer.
     * 
     * @param leadfield (n_channels x n_sources*n_ori)
     * @param data_cov Data covariance (the C matrix in LCMV).
     * @param noise_cov Noise covariance (optional, for weight normalization).
     * @param reg Regularization factor (0.05).
     * @param pick_ori "vector", "max-power", "normal".
     * @param weight_norm "unit-noise-gain", "none".
     * @param n_ori Number of orientations per source (default 3 for vector, 1 for fixed/normal).
     */
    static BeamformerWeights make_lcmv(
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov = Covariance(),
        double reg = 0.05,
        const std::string& pick_ori = "vector",
        const std::string& weight_norm = "none",
        int n_ori = 3
    );

    static Eigen::MatrixXd apply(const BeamformerWeights& weights, const Eigen::MatrixXd& data);

    // Deprecated interface
    static Eigen::MatrixXd compute_weights(const Eigen::MatrixXd& leadfield, 
                                           const Eigen::MatrixXd& data_cov, 
                                           double reg = 0.05);
};

} // NAMESPACE

#endif // LCMV_H
