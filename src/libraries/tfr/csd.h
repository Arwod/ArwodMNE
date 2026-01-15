#ifndef CSD_H
#define CSD_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <complex>

namespace TFRLIB {

class TFRSHARED_EXPORT CSD {
public:
    CSD();
    ~CSD() = default;

    // Data Holders
    std::vector<double> freqs;
    std::vector<Eigen::MatrixXcd> data; // Vector of CSD matrices (n_chan x n_chan) per freq
    int n_fft;

    /**
     * @brief Compute CSD using multitaper.
     * 
     * @param epochs Input epochs (std::vector<MatrixXd>). Each Matrix is (n_channels x n_times).
     * @param sfreq Sampling frequency.
     * @param tmin Time start (relative to epoch, not used for cropping currently, just for info).
     * @param tmax Time end.
     * @param fmin Min frequency.
     * @param fmax Max frequency.
     * @param bandwidth Bandwidth for multitaper.
     * @param adaptive Use adaptive weighting.
     * @param low_bias Use low bias tapers.
     * @return CSD object.
     */
    static CSD compute_multitaper(
        const std::vector<Eigen::MatrixXd>& epochs, 
        double sfreq, 
        double tmin = 0.0, double tmax = 0.0,
        double fmin = 0.0, double fmax = 100.0,
        double bandwidth = 0.0,
        bool adaptive = false,
        bool low_bias = true);
        
    // Helper to get matrix at specific freq (nearest)
    Eigen::MatrixXcd get_data(double freq) const;
};

} // NAMESPACE

#endif // CSD_H
