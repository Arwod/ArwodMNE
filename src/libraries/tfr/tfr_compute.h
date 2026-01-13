#ifndef TFR_COMPUTE_H
#define TFR_COMPUTE_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>

namespace TFRLIB {

class TFRSHARED_EXPORT TFRCompute
{
public:
    /**
     * Compute Time-Frequency Representation using Morlet wavelets.
     *
     * @param[in] data      Input data (n_channels x n_times).
     * @param[in] sfreq     Sampling frequency.
     * @param[in] freqs     Frequencies of interest.
     * @param[in] n_cycles  Number of cycles (fixed float).
     * @param[in] use_fft   Use FFT for convolution (default true).
     * @param[in] decim     Decimation factor (default 1).
     * @return              TFR power (n_channels x n_freqs x n_times).
     *                      Note: Currently returns power only.
     */
    static std::vector<std::vector<Eigen::VectorXd>> tfr_morlet(const Eigen::MatrixXd& data,
                                                                double sfreq,
                                                                const Eigen::VectorXd& freqs,
                                                                double n_cycles = 7.0,
                                                                bool use_fft = true,
                                                                int decim = 1);

    /**
     * Compute Time-Frequency Representation using Morlet wavelets (enhanced version).
     *
     * @param[in] data      Input data (n_channels x n_times).
     * @param[in] sfreq     Sampling frequency.
     * @param[in] freqs     Frequencies of interest.
     * @param[in] n_cycles  Number of cycles (can be vector for variable cycles per frequency).
     * @param[in] use_fft   Use FFT for convolution (default true).
     * @param[in] output    Output type: "complex", "power", "phase", "avg_power", "itc".
     * @param[in] decim     Decimation factor (default 1).
     * @return              TFR data in requested format.
     */
    static std::vector<std::vector<Eigen::VectorXcd>> tfr_morlet_enhanced(const Eigen::MatrixXd& data,
                                                                          double sfreq,
                                                                          const Eigen::VectorXd& freqs,
                                                                          const Eigen::VectorXd& n_cycles,
                                                                          bool use_fft = true,
                                                                          const std::string& output = "complex",
                                                                          int decim = 1);

    /**
     * Compute Time-Frequency Representation using multitaper method.
     *
     * @param[in] data          Input data (n_channels x n_times).
     * @param[in] sfreq         Sampling frequency.
     * @param[in] freqs         Frequencies of interest.
     * @param[in] n_cycles      Number of cycles for each frequency.
     * @param[in] time_bandwidth Time-bandwidth product (default 4.0).
     * @param[in] use_fft       Use FFT for convolution (default true).
     * @param[in] output        Output type: "complex", "power", "phase".
     * @param[in] decim         Decimation factor (default 1).
     * @return                  TFR data in requested format.
     */
    static std::vector<std::vector<Eigen::VectorXcd>> tfr_multitaper(const Eigen::MatrixXd& data,
                                                                     double sfreq,
                                                                     const Eigen::VectorXd& freqs,
                                                                     const Eigen::VectorXd& n_cycles,
                                                                     double time_bandwidth = 4.0,
                                                                     bool use_fft = true,
                                                                     const std::string& output = "complex",
                                                                     int decim = 1);
};

} // NAMESPACE

#endif // TFR_COMPUTE_H
