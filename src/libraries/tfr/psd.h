#ifndef PSD_H
#define PSD_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <utility>

namespace TFRLIB {

class TFRSHARED_EXPORT PSD
{
public:
    /**
     * Compute Power Spectral Density using Welch's method.
     *
     * @param[in] data      Input data (n_channels x n_times).
     * @param[in] sfreq     Sampling frequency.
     * @param[in] n_fft     Length of FFT used.
     * @param[in] n_overlap Number of points of overlap between segments.
     * @param[in] n_per_seg Length of each segment. If 0, n_per_seg = n_fft.
     * @param[in] window    Window to use (e.g. "hamming").
     * @return              Pair of (PSDs, freqs).
     *                      PSDs: n_channels x n_freqs (where n_freqs = n_fft/2 + 1)
     *                      Freqs: VectorXd of frequencies.
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> psd_welch(const Eigen::MatrixXd& data,
                                                                 double sfreq,
                                                                 int n_fft = 256,
                                                                 int n_overlap = 0,
                                                                 int n_per_seg = 0,
                                                                 const std::string& window = "hamming");

    /**
     * Compute Power Spectral Density using multitaper method.
     *
     * @param[in] data      Input data (n_channels x n_times).
     * @param[in] sfreq     Sampling frequency.
     * @param[in] bandwidth Frequency bandwidth. If 0.0, defaults to 4.0 / n_times * sfreq.
     * @param[in] adaptive  Use adaptive weighting (not implemented, defaults to false).
     * @param[in] low_bias  Only use tapers with eigenvalues > 0.9 (defaults to true).
     * @return              Pair of (PSDs, freqs).
     *                      PSDs: n_channels x n_freqs
     *                      Freqs: VectorXd of frequencies.
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> psd_multitaper(const Eigen::MatrixXd& data,
                                                                      double sfreq,
                                                                      double bandwidth = 0.0,
                                                                      bool adaptive = false,
                                                                      bool low_bias = true);

    /**
     * Enhanced Welch's method with optimized windowing and overlap processing.
     *
     * @param[in] data          Input data array (n_channels x n_times).
     * @param[in] sfreq         Sampling frequency.
     * @param[in] fmin          Minimum frequency of interest.
     * @param[in] fmax          Maximum frequency of interest.
     * @param[in] n_fft         Length of FFT used.
     * @param[in] n_overlap     Number of points of overlap between segments.
     * @param[in] n_per_seg     Length of each segment.
     * @param[in] window        Window function name.
     * @param[in] detrend       Detrending method ("constant", "linear", "none").
     * @param[in] scaling       Scaling method ("density", "spectrum").
     * @return                  Pair of (PSDs, freqs) within frequency range.
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> psd_array_welch(const Eigen::MatrixXd& data,
                                                                       double sfreq,
                                                                       double fmin = 0.0,
                                                                       double fmax = -1.0,
                                                                       int n_fft = 256,
                                                                       int n_overlap = -1,
                                                                       int n_per_seg = 0,
                                                                       const std::string& window = "hamming",
                                                                       const std::string& detrend = "constant",
                                                                       const std::string& scaling = "density");

    /**
     * Enhanced multitaper method with optimized taper selection and weighting.
     *
     * @param[in] data          Input data array (n_channels x n_times).
     * @param[in] sfreq         Sampling frequency.
     * @param[in] fmin          Minimum frequency of interest.
     * @param[in] fmax          Maximum frequency of interest.
     * @param[in] bandwidth     Frequency bandwidth.
     * @param[in] adaptive      Use adaptive weighting.
     * @param[in] low_bias      Only use high-eigenvalue tapers.
     * @param[in] normalization Normalization method ("full", "length").
     * @return                  Pair of (PSDs, freqs) within frequency range.
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> psd_array_multitaper(const Eigen::MatrixXd& data,
                                                                            double sfreq,
                                                                            double fmin = 0.0,
                                                                            double fmax = -1.0,
                                                                            double bandwidth = 0.0,
                                                                            bool adaptive = true,
                                                                            bool low_bias = true,
                                                                            const std::string& normalization = "full");

    /**
     * Generate optimized window function.
     *
     * @param[in] window_name   Name of window function.
     * @param[in] n_samples     Number of samples.
     * @param[in] beta          Beta parameter for Kaiser window.
     * @return                  Window coefficients.
     */
    static Eigen::VectorXd generate_window(const std::string& window_name, 
                                          int n_samples, 
                                          double beta = 8.6);

    /**
     * Apply detrending to signal segment.
     *
     * @param[in,out] segment   Signal segment to detrend.
     * @param[in] method        Detrending method ("constant", "linear", "none").
     */
    static void apply_detrend(Eigen::VectorXd& segment, const std::string& method = "constant");

    /**
     * Compute optimal overlap for given window and desired overlap ratio.
     *
     * @param[in] n_per_seg     Segment length.
     * @param[in] overlap_ratio Desired overlap ratio (0.0 to 1.0).
     * @return                  Optimal overlap in samples.
     */
    static int compute_optimal_overlap(int n_per_seg, double overlap_ratio = 0.5);
};

} // NAMESPACE

#endif // PSD_H
