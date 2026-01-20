#ifndef TFR_UTILS_H
#define TFR_UTILS_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>

namespace TFRLIB {

class TFRSHARED_EXPORT TFRUtils
{
public:
    /**
     * Creates a Morlet wavelet.
     *
     * @param[in] sfreq     Sampling frequency.
     * @param[in] freqs     Frequencies to compute Morlet wavelets for.
     * @param[in] n_cycles  Number of cycles.
     * @param[in] sigma     Sigma of the envelope (optional, default usually n_cycles / (2 * pi * freq)).
     *                      If 0, it is calculated from n_cycles.
     * @param[in] zero_mean Make sure the wavelet has a mean of zero.
     * @return              List of Morlet wavelets (one per frequency).
     */
    static std::vector<Eigen::VectorXcd> morlet(double sfreq,
                                                const Eigen::VectorXd& freqs,
                                                double n_cycles = 7.0,
                                                double sigma = 0.0,
                                                bool zero_mean = false);

    /**
     * Creates Morlet wavelets with variable cycles per frequency.
     *
     * @param[in] sfreq     Sampling frequency.
     * @param[in] freqs     Frequencies to compute Morlet wavelets for.
     * @param[in] n_cycles  Number of cycles per frequency (vector).
     * @param[in] sigma     Sigma of the envelope (optional).
     * @param[in] zero_mean Make sure the wavelet has a mean of zero.
     * @return              List of Morlet wavelets (one per frequency).
     */
    static std::vector<Eigen::VectorXcd> morlet_variable(double sfreq,
                                                         const Eigen::VectorXd& freqs,
                                                         const Eigen::VectorXd& n_cycles,
                                                         double sigma = 0.0,
                                                         bool zero_mean = false);

    /**
     * Creates DPSS (Slepian) windows.
     *
     * @param[in] N     Sequence length.
     * @param[in] nw    Time-bandwidth product.
     * @param[in] k_max Number of windows to return.
     * @return          Pair of (Windows [N x k_max], Eigenvalues [k_max]).
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> dpss_windows(int N, double nw, int k_max);

};

} // NAMESPACE

#endif // TFR_UTILS_H
