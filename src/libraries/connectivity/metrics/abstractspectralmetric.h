#ifndef ABSTRACTSPECTRALMETRIC_H
#define ABSTRACTSPECTRALMETRIC_H

#include "../connectivity_global.h"
#include "abstractmetric.h"
#include <tfr/tfr_utils.h>
#include <unsupported/Eigen/FFT>
#include <QPair>
#include <QVector>
#include <Eigen/Core>

namespace CONNECTIVITYLIB {

class CONNECTIVITYSHARED_EXPORT AbstractSpectralMetric : public AbstractMetric
{
public:
    AbstractSpectralMetric();
    virtual ~AbstractSpectralMetric() = default;

protected:
    /**
     * @brief Computes tapered spectra for a single trial.
     * 
     * @param trialData     Input data (n_channels x n_samples)
     * @param tapers        Tapers matrix (n_tapers x n_samples)
     * @param n_fft         FFT length
     * @param n_freqs       Number of frequencies to keep (usually n_fft/2 + 1)
     * @param weights       Taper weights (eigenvalues)
     * 
     * @return List of matrices (one per channel), each (n_tapers x n_freqs)
     */
    static QVector<Eigen::MatrixXcd> computeTaperedSpectra(
        const Eigen::MatrixXd& trialData,
        const Eigen::MatrixXd& tapers,
        int n_fft,
        int n_freqs,
        const Eigen::VectorXd& weights);

    /**
     * @brief Generates tapers using mne_tfr.
     */
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> generateTapers(
        int n_times, const QString& windowType, int n_tapers = -1, double nw = 4.0);
};

} // NAMESPACE

#endif // ABSTRACTSPECTRALMETRIC_H
