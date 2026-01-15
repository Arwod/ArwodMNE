#ifndef DICS_H
#define DICS_H

#include "../inverse_global.h"
#include <tfr/csd.h>
#include <Eigen/Core>
#include <vector>

namespace INVERSELIB {

class INVERSESHARED_EXPORT DICS
{
public:
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
};

} // NAMESPACE

#endif // DICS_H
