#ifndef DICS_H
#define DICS_H

#include "../inverse_global.h"
#include <tfr/csd.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>

namespace INVERSELIB {

/**
 * @brief DICS Beamformer Filter structure
 * 
 * Contains all information needed to apply DICS beamformer filters.
 */
struct DicsFilter {
    std::vector<Eigen::MatrixXcd> weights;      // Filter weights for each frequency
    std::vector<double> frequencies;            // Frequencies corresponding to weights
    std::string pick_ori;                       // Orientation selection method
    std::string weight_norm;                    // Weight normalization method
    std::string inversion;                      // Inversion method ("single" or "matrix")
    bool real_filter;                           // Whether real filters were used
    int n_sources;                              // Number of sources
    int n_channels;                             // Number of channels
    bool is_free_ori;                           // Whether free orientation was used
    
    DicsFilter() : pick_ori("vector"), weight_norm("none"), inversion("matrix"), 
                   real_filter(true), n_sources(0), n_channels(0), is_free_ori(false) {}
};

class INVERSESHARED_EXPORT DICS
{
public:
    /**
     * @brief Create DICS beamformer filters (equivalent to make_dics in MNE Python).
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     * @param csd Cross-Spectral Density object.
     * @param reg Regularization parameter (default 0.05).
     * @param noise_csd Optional noise CSD for whitening (default nullptr).
     * @param pick_ori Orientation selection: "vector", "max-power", "normal" (default "vector").
     * @param weight_norm Weight normalization: "none", "unit-noise-gain", "unit-noise-gain-invariant" (default "none").
     * @param inversion Inversion method: "single", "matrix" (default "matrix").
     * @param real_filter Whether to use real part of CSD (default true).
     * @param n_ori Number of orientations per source (default 1).
     * @return DicsFilter object containing all filter information.
     */
    static DicsFilter make_dics(const Eigen::MatrixXd& leadfield,
                                const TFRLIB::CSD& csd,
                                double reg = 0.05,
                                const TFRLIB::CSD* noise_csd = nullptr,
                                const std::string& pick_ori = "vector",
                                const std::string& weight_norm = "none",
                                const std::string& inversion = "matrix",
                                bool real_filter = true,
                                int n_ori = 1);

    /**
     * @brief Apply DICS beamformer to CSD data for source power estimation.
     * 
     * @param csd Cross-Spectral Density object.
     * @param filters Pre-computed DICS filters.
     * @return Source power matrix (n_sources x n_freqs) and frequencies vector.
     */
    static std::pair<Eigen::MatrixXd, std::vector<double>> apply_dics_csd(
        const TFRLIB::CSD& csd,
        const DicsFilter& filters);

    /**
     * @brief Compute Source Power using DICS beamformer.
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     *                  Assuming sources are grouped (e.g. 3 cols per source if n_ori=3).
     *                  If n_ori is 1 (fixed or normal), step is 1.
     * @param csd       Cross-Spectral Density object.
     * @param reg       Regularization parameter (e.g. 0.05).
     * @param n_ori     Number of orientations per source (default 1).
     * @param real_filter Whether to use real part of CSD to compute filters (default true).
     * @return Source Power Matrix (n_sources x n_freqs).
     *         n_sources = leadfield.cols() / n_ori.
     */
    static Eigen::MatrixXd compute_source_power(const Eigen::MatrixXd& leadfield, 
                                                const TFRLIB::CSD& csd, 
                                                double reg = 0.05,
                                                int n_ori = 1,
                                                bool real_filter = true);

    /**
     * @brief Apply DICS beamformer to epochs data.
     * 
     * @param epochs Input epochs data (vector of matrices, each n_channels x n_times).
     * @param filters Pre-computed DICS filters.
     * @return Source space epochs (vector of matrices, each n_sources x n_times).
     */
    static std::vector<Eigen::MatrixXcd> apply_dics_epochs(
        const std::vector<Eigen::MatrixXd>& epochs,
        const DicsFilter& filters);

    /**
     * @brief Apply DICS beamformer to raw/evoked data.
     * 
     * @param data Input data (n_channels x n_times).
     * @param filters Pre-computed DICS filters.
     * @return Source space data (n_sources x n_times) - complex for preserving phase.
     */
    static Eigen::MatrixXcd apply_dics_raw(
        const Eigen::MatrixXd& data,
        const DicsFilter& filters);

    /**
     * @brief Apply DICS beamformer to time-frequency epochs data.
     * 
     * @param epochs_tfr Input time-frequency epochs (vector of 3D arrays: n_channels x n_freqs x n_times).
     * @param filters Pre-computed DICS filters.
     * @return Source space time-frequency data (vector of 3D arrays: n_sources x n_freqs x n_times).
     */
    static std::vector<std::vector<Eigen::MatrixXcd>> apply_dics_tfr_epochs(
        const std::vector<std::vector<Eigen::MatrixXcd>>& epochs_tfr,
        const DicsFilter& filters);

private:
    /**
     * @brief Compute DICS filters for later application.
     * 
     * @param leadfield Forward solution leadfield (n_channels x n_sources*n_ori).
     * @param csd Cross-Spectral Density object.
     * @param reg Regularization parameter.
     * @param noise_csd Optional noise CSD for whitening.
     * @param n_ori Number of orientations per source.
     * @param real_filter Whether to use real part of CSD for filter computation.
     * @param pick_ori Orientation selection: "vector", "max-power", "normal".
     * @param weight_norm Weight normalization method.
     * @param inversion Inversion method: "single" or "matrix".
     * @return DICS filters (n_sources x n_channels x n_freqs) - complex filters.
     */
    static std::vector<Eigen::MatrixXcd> compute_dics_filters(
        const Eigen::MatrixXd& leadfield,
        const TFRLIB::CSD& csd,
        double reg = 0.05,
        const TFRLIB::CSD* noise_csd = nullptr,
        int n_ori = 1,
        bool real_filter = true,
        const std::string& pick_ori = "vector",
        const std::string& weight_norm = "none",
        const std::string& inversion = "matrix");

    /**
     * @brief Apply weight normalization to beamformer filters.
     * 
     * @param filters Input filters to normalize.
     * @param leadfield Forward solution leadfield.
     * @param csd Cross-Spectral Density object.
     * @param noise_csd Optional noise CSD.
     * @param weight_norm Normalization method.
     * @param n_ori Number of orientations per source.
     * @return Normalized filters.
     */
    static std::vector<Eigen::MatrixXcd> apply_weight_normalization(
        const std::vector<Eigen::MatrixXcd>& filters,
        const Eigen::MatrixXd& leadfield,
        const TFRLIB::CSD& csd,
        const TFRLIB::CSD* noise_csd,
        const std::string& weight_norm,
        int n_ori);

    /**
     * @brief Compute whitening matrix from noise CSD.
     * 
     * @param noise_csd Noise Cross-Spectral Density.
     * @param reg Regularization parameter.
     * @return Whitening matrix.
     */
    static Eigen::MatrixXcd compute_whitener(const TFRLIB::CSD& noise_csd, double reg);

    /**
     * @brief Optimize frequency domain complex filtering.
     * 
     * @param filters Pre-computed DICS filters.
     * @param data_fft FFT of input data (n_channels x n_freqs x n_times).
     * @return Source space data in frequency domain (n_sources x n_freqs x n_times).
     */
    static std::vector<Eigen::MatrixXcd> apply_filters_fft(
        const std::vector<Eigen::MatrixXcd>& filters,
        const std::vector<Eigen::MatrixXcd>& data_fft);
};

} // NAMESPACE

#endif // DICS_H
