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
};

} // NAMESPACE

#endif // PSD_H
