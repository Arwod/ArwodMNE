#include "abstractspectralmetric.h"
#include <utils/spectral.h> // Fallback or if needed, but we use tfr_utils mostly
#include <QDebug>

using namespace Eigen;
using namespace TFRLIB;

namespace CONNECTIVITYLIB {

AbstractSpectralMetric::AbstractSpectralMetric()
{
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> AbstractSpectralMetric::generateTapers(
    int n_times, const QString& windowType, int n_tapers, double nw)
{
    if (windowType == "multitaper") {
        // Use TFRUtils from Phase 11
        if (n_tapers == -1) {
            n_tapers = int(2 * nw - 1);
        }
        return TFRUtils::dpss_windows(n_times, nw, n_tapers);
    } else {
        // Use Hanning/Hamming from existing utils (or implement here)
        // For now, let's rely on UTILSLIB::Spectral as a fallback for non-multitaper
        // or just reimplement simple Hanning here to avoid utilslib dependency if we want.
        // But the original code used UTILSLIB::Spectral::generateTapers.
        // Let's assume we want to use TFRUtils for DPSS.
        // For Hanning, we can generate it easily.
        
        MatrixXd tapers(1, n_times);
        VectorXd weights = VectorXd::Ones(1);
        
        if (windowType == "hanning") {
             // Simple Hanning window
             for (int i = 0; i < n_times; ++i) {
                 tapers(0, i) = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (n_times - 1)));
             }
        } else {
            // Default to boxcar or Hamming
             tapers.setOnes();
        }
        
        // Normalize L2
        tapers.row(0) /= tapers.row(0).norm();
        
        return std::make_pair(tapers, weights);
    }
}

QVector<Eigen::MatrixXcd> AbstractSpectralMetric::computeTaperedSpectra(
    const Eigen::MatrixXd& trialData,
    const Eigen::MatrixXd& tapers,
    int n_fft,
    int n_freqs,
    const Eigen::VectorXd& weights)
{
    Q_UNUSED(weights); // Weights used later in CSD average usually
    
    int n_channels = trialData.rows();
    int n_tapers = tapers.rows();
    
    QVector<Eigen::MatrixXcd> tapSpectra(n_channels);
    
    FFT<double> fft;
    fft.SetFlag(fft.HalfSpectrum);
    
    // Resize output
    for(int i = 0; i < n_channels; ++i) {
        tapSpectra[i].resize(n_tapers, n_freqs);
    }
    
    for (int i = 0; i < n_channels; ++i) {
        // Center data
        RowVectorXd rowData = trialData.row(i);
        rowData.array() -= rowData.mean();
        
        for (int k = 0; k < n_tapers; ++k) {
            RowVectorXd taperedData;
            // Zero pad
            if (rowData.cols() < n_fft) {
                taperedData.setZero(n_fft);
                taperedData.head(rowData.cols()) = rowData.cwiseProduct(tapers.row(k));
            } else {
                taperedData = rowData.cwiseProduct(tapers.row(k));
            }
            
            // FFT
            RowVectorXcd freqData(n_freqs);
            std::vector<std::complex<double>> tmpOut;
            std::vector<double> tmpIn(taperedData.data(), taperedData.data() + taperedData.size());
            
            fft.fwd(tmpOut, tmpIn);
            
            // Map to Eigen
            for(int f = 0; f < n_freqs; ++f) {
                tapSpectra[i](k, f) = tmpOut[f];
            }
        }
    }
    
    return tapSpectra;
}

} // NAMESPACE
