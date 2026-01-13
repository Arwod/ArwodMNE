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

    /**
     * @brief Compute CSD using Morlet wavelets.
     * 
     * @param epochs Input epochs (std::vector<MatrixXd>). Each Matrix is (n_channels x n_times).
     * @param frequencies Frequencies of interest.
     * @param sfreq Sampling frequency.
     * @param tmin Time start (relative to epoch).
     * @param tmax Time end.
     * @param n_cycles Number of cycles for each frequency.
     * @param use_fft Use FFT for convolution.
     * @param decim Decimation factor.
     * @return CSD object.
     */
    static CSD compute_morlet(
        const std::vector<Eigen::MatrixXd>& epochs,
        const Eigen::VectorXd& frequencies,
        double sfreq,
        double tmin = 0.0, double tmax = 0.0,
        const Eigen::VectorXd& n_cycles = Eigen::VectorXd(),
        bool use_fft = true,
        int decim = 1);

    /**
     * @brief Compute CSD using Fourier method.
     * 
     * @param epochs Input epochs (std::vector<MatrixXd>). Each Matrix is (n_channels x n_times).
     * @param sfreq Sampling frequency.
     * @param fmin Min frequency.
     * @param fmax Max frequency.
     * @param tmin Time start (relative to epoch).
     * @param tmax Time end.
     * @param n_fft FFT length.
     * @param overlap Overlap between segments.
     * @return CSD object.
     */
    static CSD compute_fourier(
        const std::vector<Eigen::MatrixXd>& epochs,
        double sfreq,
        double fmin = 0.0, double fmax = 100.0,
        double tmin = 0.0, double tmax = 0.0,
        int n_fft = -1,
        double overlap = 0.5);

    /**
     * @brief Verify Hermitian property of CSD matrices.
     * 
     * @param tolerance Tolerance for Hermitian check.
     * @return True if all CSD matrices are Hermitian within tolerance.
     */
    bool verify_hermitian(double tolerance = 1e-10) const;

    /**
     * @brief Get the condition number of CSD matrix at specific frequency.
     * 
     * @param freq Frequency of interest.
     * @return Condition number.
     */
    double get_condition_number(double freq) const;
        
    // Helper to get matrix at specific freq (nearest)
    Eigen::MatrixXcd get_data(double freq) const;
};

} // NAMESPACE

#endif // CSD_H
