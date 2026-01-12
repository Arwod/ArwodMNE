#include "csd.h"
#include "tfr_utils.h"
#include <unsupported/Eigen/FFT>
#include <iostream>
#include <cmath>

namespace TFRLIB {

CSD::CSD() : n_fft(0)
{
}

Eigen::MatrixXcd CSD::get_data(double freq) const
{
    if (freqs.empty()) return Eigen::MatrixXcd();
    
    // Find nearest
    int idx = 0;
    double min_diff = std::abs(freq - freqs[0]);
    for (int i = 1; i < freqs.size(); ++i) {
        double diff = std::abs(freq - freqs[i]);
        if (diff < min_diff) {
            min_diff = diff;
            idx = i;
        }
    }
    return data[idx];
}

CSD CSD::compute_multitaper(const std::vector<Eigen::MatrixXd>& epochs, 
                            double sfreq, 
                            double tmin, double tmax,
                            double fmin, double fmax,
                            double bandwidth,
                            bool adaptive,
                            bool low_bias)
{
    CSD csd;
    if (epochs.empty()) return csd;
    
    int n_channels = epochs[0].rows();
    int n_times = epochs[0].cols();
    
    if (bandwidth == 0.0) {
        bandwidth = 4.0 / ((double)n_times / sfreq);
    }
    
    double nw = bandwidth * n_times / (2.0 * sfreq);
    int k_max = (int)(2.0 * nw) - 1;
    if (k_max < 1) k_max = 1;
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> dpss = TFRUtils::dpss_windows(n_times, nw, k_max);
    const Eigen::MatrixXd& tapers = dpss.first;
    const Eigen::VectorXd& eigen_vals = dpss.second;
    
    std::vector<int> valid_tapers;
    if (low_bias) {
        for (int i = 0; i < k_max; ++i) {
            if (eigen_vals[i] > 0.9) valid_tapers.push_back(i);
        }
    } else {
        for (int i = 0; i < k_max; ++i) valid_tapers.push_back(i);
    }
    
    if (valid_tapers.empty()) valid_tapers.push_back(0); 
    int n_tapers = valid_tapers.size();
    
    int n_fft = n_times; // Usually next power of 2
    csd.n_fft = n_fft;
    int n_freqs_all = n_fft / 2 + 1;
    
    // Identify freq indices to keep
    std::vector<int> freq_indices;
    for (int i = 0; i < n_freqs_all; ++i) {
        double f = (double)i * sfreq / (double)n_fft;
        if (f >= fmin && f <= fmax) {
            freq_indices.push_back(i);
            csd.freqs.push_back(f);
        }
    }
    
    int n_freqs_kept = freq_indices.size();
    csd.data.resize(n_freqs_kept, Eigen::MatrixXcd::Zero(n_channels, n_channels));
    
    Eigen::FFT<double> fft;
    
    for (const auto& epoch : epochs) {
        for (int k_idx : valid_tapers) {
            Eigen::VectorXd window = tapers.col(k_idx);
            
            // Temporary storage for this taper's FFTs at relevant frequencies
            // X_f[i] is vector of size n_channels for freq i (where i is index in freq_indices)
            std::vector<Eigen::VectorXcd> X_f(n_freqs_kept); 
            for (int i=0; i<n_freqs_kept; ++i) {
                X_f[i].resize(n_channels);
            }
            
            for (int ch = 0; ch < n_channels; ++ch) {
                Eigen::VectorXd signal = epoch.row(ch);
                double mean = signal.mean();
                signal.array() -= mean;
                
                Eigen::VectorXd windowed_sig = signal.array() * window.array();
                
                std::vector<double> time_domain(n_fft, 0.0);
                for(int i=0; i<n_times; ++i) time_domain[i] = windowed_sig[i];
                
                std::vector<std::complex<double>> freq_domain(n_fft);
                fft.fwd(freq_domain, time_domain);
                
                for (int i = 0; i < n_freqs_kept; ++i) {
                    int f_idx = freq_indices[i];
                    
                    // Scaling
                    double factor = 2.0 / sfreq;
                    if (n_fft % 2 == 0) {
                        if (f_idx == 0 || f_idx == n_fft/2) factor /= 2.0;
                    } else {
                        if (f_idx == 0) factor /= 2.0;
                    }
                    
                    X_f[i](ch) = freq_domain[f_idx] * std::sqrt(factor);
                }
            }
            
            // Accumulate
            for (int i = 0; i < n_freqs_kept; ++i) {
                csd.data[i] += X_f[i] * X_f[i].adjoint();
            }
        }
    }
    
    // Normalize
    double n_epochs = (double)epochs.size();
    double total_weight = n_epochs * (double)n_tapers;
    
    for (int i = 0; i < n_freqs_kept; ++i) {
        csd.data[i] /= total_weight;
    }
    
    return csd;
}

} // NAMESPACE
