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
    std::string weight_norm;
    Eigen::MatrixXd whitening_mat; // Optional whitening matrix used
    Eigen::MatrixXd max_power_ori; // For max-power orientation (n_sources x 3)
    double depth_param;            // Depth weighting parameter used
    std::string inversion_method;  // "matrix" or "single"
    bool is_free_ori;             // Whether free orientation was used
    int n_sources;                // Number of source locations
    int rank;                     // Rank of data covariance matrix
    
    BeamformerWeights() : weight_norm("none"), depth_param(0.0), 
                         inversion_method("matrix"), is_free_ori(true), 
                         n_sources(0), rank(0) {}
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
     * @param weight_norm "unit-noise-gain", "unit-noise-gain-invariant", "nai", "neural-activity-index", "none".
     * @param n_ori Number of orientations per source (default 3 for vector, 1 for fixed/normal).
     * @param depth Depth weighting parameter (optional, for depth bias compensation).
     * @param reduce_rank Whether to reduce rank of covariance matrix.
     * @param inversion Inversion method: "matrix" or "single".
     */
    static BeamformerWeights make_lcmv(
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov = Covariance(),
        double reg = 0.05,
        const std::string& pick_ori = "vector",
        const std::string& weight_norm = "unit-noise-gain-invariant",
        int n_ori = 3,
        double depth = 0.0,
        bool reduce_rank = false,
        const std::string& inversion = "matrix"
    );

    /**
     * @brief Compute resolution matrix for LCMV beamformer.
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     * @param weights Beamformer weights from make_lcmv.
     * @param n_ori Number of orientations per source.
     * @return Resolution matrix (n_sources x n_sources).
     */
    static Eigen::MatrixXd compute_resolution_matrix(
        const Eigen::MatrixXd& leadfield,
        const BeamformerWeights& weights,
        int n_ori = 3
    );

    /**
     * @brief Apply depth weighting to leadfield matrix.
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     * @param depth Depth weighting parameter (0.0 = no weighting, 0.8 = standard).
     * @param source_positions Source positions (n_sources x 3) in head coordinates.
     * @return Depth-weighted leadfield matrix.
     */
    static Eigen::MatrixXd apply_depth_weighting(
        const Eigen::MatrixXd& leadfield,
        double depth,
        const Eigen::MatrixXd& source_positions,
        int n_ori = 3
    );

    /**
     * @brief Compute neural activity index (NAI) normalization.
     * 
     * @param weights Beamformer weights.
     * @param leadfield Forward solution leadfield.
     * @param noise_cov Noise covariance matrix.
     * @param n_ori Number of orientations per source.
     * @return NAI normalized weights.
     */
    static BeamformerWeights apply_nai_normalization(
        const BeamformerWeights& weights,
        const Eigen::MatrixXd& leadfield,
        const Covariance& noise_cov,
        int n_ori = 3
    );

    /**
     * @brief Compute neural activity index (NAI) normalization - alternative name.
     * 
     * @param weights Beamformer weights.
     * @param leadfield Forward solution leadfield.
     * @param noise_cov Noise covariance matrix.
     * @param n_ori Number of orientations per source.
     * @return NAI normalized weights.
     */
    static BeamformerWeights apply_neural_activity_index_normalization(
        const BeamformerWeights& weights,
        const Eigen::MatrixXd& leadfield,
        const Covariance& noise_cov,
        int n_ori = 3
    );

    /**
     * @brief Compute beamformer weights using matrix inversion method.
     * 
     * @param leadfield Forward solution leadfield.
     * @param data_cov Data covariance matrix.
     * @param noise_cov Noise covariance matrix.
     * @param reg Regularization parameter.
     * @param weight_norm Weight normalization method.
     * @param n_ori Number of orientations per source.
     * @return Beamformer weights computed using matrix inversion.
     */
    static BeamformerWeights compute_matrix_inversion_weights(
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov,
        double reg,
        const std::string& weight_norm,
        int n_ori,
        const std::string& pick_ori
    );

    /**
     * @brief Compute beamformer weights using single dipole method.
     * 
     * @param leadfield Forward solution leadfield.
     * @param data_cov Data covariance matrix.
     * @param noise_cov Noise covariance matrix.
     * @param reg Regularization parameter.
     * @param weight_norm Weight normalization method.
     * @param n_ori Number of orientations per source.
     * @return Beamformer weights computed using single dipole method.
     */
    static BeamformerWeights compute_single_dipole_weights(
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov,
        double reg,
        const std::string& weight_norm,
        int n_ori,
        const std::string& pick_ori
    );

    /**
     * @brief Compute unit-noise-gain-invariant (UNGI) normalization.
     * 
     * @param weights Beamformer weights.
     * @param leadfield Forward solution leadfield.
     * @param noise_cov Noise covariance matrix.
     * @param n_ori Number of orientations per source.
     * @return UNGI normalized weights.
     */
    static BeamformerWeights apply_ungi_normalization(
        const BeamformerWeights& weights,
        const Eigen::MatrixXd& leadfield,
        const Covariance& noise_cov,
        int n_ori = 3
    );

    static Eigen::MatrixXd apply(const BeamformerWeights& weights, const Eigen::MatrixXd& data);

    // Deprecated interface
    static Eigen::MatrixXd compute_weights(const Eigen::MatrixXd& leadfield, 
                                           const Eigen::MatrixXd& data_cov, 
                                           double reg = 0.05);

private:
    /**
     * @brief Apply orientation selection and weight normalization.
     * 
     * @param weights Raw beamformer weights (n_dipoles x n_channels).
     * @param leadfield Forward solution leadfield.
     * @param data_cov Data covariance matrix.
     * @param noise_cov Noise covariance matrix.
     * @param weight_norm Weight normalization method.
     * @param n_ori Number of orientations per source.
     * @param pick_ori Orientation selection method.
     * @return Processed beamformer weights.
     */
    static BeamformerWeights apply_orientation_selection_and_normalization(
        const Eigen::MatrixXd& weights,
        const Eigen::MatrixXd& leadfield,
        const Covariance& data_cov,
        const Covariance& noise_cov,
        const std::string& weight_norm,
        int n_ori,
        const std::string& pick_ori
    );
};

} // NAMESPACE

#endif // LCMV_H
