#include "tfr_compute.h"
#include "tfr_utils.h"
#include <utils/mnemath.h>
#include <iostream>

namespace TFRLIB {

std::vector<std::vector<Eigen::VectorXd>> TFRCompute::tfr_morlet(const Eigen::MatrixXd& data,
                                                                 double sfreq,
                                                                 const Eigen::VectorXd& freqs,
                                                                 double n_cycles,
                                                                 bool use_fft,
                                                                 int decim)
{
    // 1. Generate Wavelets
    std::vector<Eigen::VectorXcd> wavelets = TFRUtils::morlet(sfreq, freqs, n_cycles);
    
    int n_channels = data.rows();
    int n_times = data.cols();
    int n_freqs = freqs.size();
    
    // Output structure: [channel][freq] -> time_series
    std::vector<std::vector<Eigen::VectorXd>> power(n_channels, std::vector<Eigen::VectorXd>(n_freqs));
    
    // 2. Convolve
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        
        for (int f = 0; f < n_freqs; ++f) {
            Eigen::VectorXcd W = wavelets[f];
            
            // Convolution
            // MNEMath::convolve is real-only right now?
            // I implemented VectorXd convolve(VectorXd, VectorXd).
            // But wavelet is complex. Signal is real.
            // Result is complex.
            // I need complex convolution.
            
            // MNEMath::convolve needs to support complex or I need to implement it here or extend MNEMath.
            // Ideally extend MNEMath.
            
            // For now, I can do: conv(sig, real(W)) + j*conv(sig, imag(W))
            // Because convolution is linear.
            // (f * (g_r + j*g_i)) = f*g_r + j*(f*g_i)
            
            Eigen::VectorXd W_real = W.real();
            Eigen::VectorXd W_imag = W.imag();
            
            Eigen::VectorXd conv_r = UTILSLIB::MNEMath::convolve(signal, W_real, std::string("same"));
            Eigen::VectorXd conv_i = UTILSLIB::MNEMath::convolve(signal, W_imag, std::string("same"));
            
            // Compute Power: |z|^2 = r^2 + i^2
            Eigen::VectorXd p(conv_r.size());
            for (int t = 0; t < p.size(); ++t) {
                p[t] = conv_r[t]*conv_r[t] + conv_i[t]*conv_i[t];
            }
            
            // Decimation?
            if (decim > 1) {
                // Simple decimation (picking every k-th sample)
                // MNE typically filters before decimating to avoid aliasing if decimating raw data,
                // but here we are decimating the TFR result which is smooth?
                // MNE tfr_morlet docs: "Decimation factor. Returns tfr[..., ::decim]."
                // So it just slices.
                int n_out = (p.size() + decim - 1) / decim;
                Eigen::VectorXd p_decim(n_out);
                for (int i = 0; i < n_out; ++i) {
                    if (i*decim < p.size())
                        p_decim[i] = p[i*decim];
                }
                power[ch][f] = p_decim;
            } else {
                power[ch][f] = p;
            }
        }
    }
    
    return power;
}

} // NAMESPACE
