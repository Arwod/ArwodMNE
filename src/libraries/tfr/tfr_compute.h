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
};

} // NAMESPACE

#endif // TFR_COMPUTE_H
