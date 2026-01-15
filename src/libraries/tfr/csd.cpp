#include "csd.h"
#include "tfr_utils.h"
#include <unsupported/Eigen/FFT>
#include <iostream>
#include <cmath>
#include <QtConcurrent/QtConcurrent>

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

struct CSDWorkItem {
    const Eigen::MatrixXd* epoch;
    const Eigen::MatrixXd* tapers;
    const std::vector<int>* valid_tapers;
    const std::vector<int>* freq_indices;
    double sfreq;
    int n_fft;
    int n_freqs_kept;
    int n_channels;
    int n_times;
};

// Map function
std::vector<Eigen::MatrixXcd> compute_epoch_csd(const CSDWorkItem& item) {
    std::vector<Eigen::MatrixXcd> epoch_csd(item.n_freqs_kept);
    for (int i = 0; i < item.n_freqs_kept; ++i) {
        epoch_csd[i] = Eigen::MatrixXcd::Zero(item.n_channels, item.n_channels);
    }
    
    Eigen::FFT<double> fft;
    
    for (int k_idx : *item.valid_tapers) {
        Eigen::VectorXd window = item.tapers->col(k_idx);
        
        // Temporary storage for this taper's FFTs at relevant frequencies
        std::vector<Eigen::VectorXcd> X_f(item.n_freqs_kept); 
        for (int i=0; i<item.n_freqs_kept; ++i) {
            X_f[i].resize(item.n_channels);
        }
        
        std::vector<double> time_domain(item.n_fft, 0.0);
        std::vector<std::complex<double>> freq_domain(item.n_fft);

        for (int ch = 0; ch < item.n_channels; ++ch) {
            Eigen::VectorXd signal = item.epoch->row(ch);
            double mean = signal.mean();
            signal.array() -= mean;
            
            Eigen::VectorXd windowed_sig = signal.array() * window.array();
            
            // Zero pad if needed (n_fft > n_times)
            std::fill(time_domain.begin(), time_domain.end(), 0.0);
            for(int i=0; i<item.n_times; ++i) time_domain[i] = windowed_sig[i];
            
            fft.fwd(freq_domain, time_domain);
            
            for (int i = 0; i < item.n_freqs_kept; ++i) {
                int f_idx = (*item.freq_indices)[i];
                
                // Scaling
                double factor = 2.0 / item.sfreq;
                if (item.n_fft % 2 == 0) {
                    if (f_idx == 0 || f_idx == item.n_fft/2) factor /= 2.0;
                } else {
                    if (f_idx == 0) factor /= 2.0;
                }
                
                X_f[i](ch) = freq_domain[f_idx] * std::sqrt(factor);
            }
        }
        
        // Accumulate taper result to epoch result
        for (int i = 0; i < item.n_freqs_kept; ++i) {
            epoch_csd[i] += X_f[i] * X_f[i].adjoint();
        }
    }
    return epoch_csd;
}

// Reduce function
void accumulate_csd(std::vector<Eigen::MatrixXcd>& result, const std::vector<Eigen::MatrixXcd>& intermediate) {
    if (result.empty()) {
        result = intermediate;
    } else {
        // Assume sizes match
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += intermediate[i];
        }
    }
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
    
    // Prepare work items
    std::vector<CSDWorkItem> work_items;
    work_items.reserve(epochs.size());
    for(const auto& epoch : epochs) {
        CSDWorkItem item;
        item.epoch = &epoch;
        item.tapers = &tapers;
        item.valid_tapers = &valid_tapers;
        item.freq_indices = &freq_indices;
        item.sfreq = sfreq;
        item.n_fft = n_fft;
        item.n_freqs_kept = n_freqs_kept;
        item.n_channels = n_channels;
        item.n_times = n_times;
        work_items.push_back(item);
    }
    
    // Execute parallel map-reduce
    csd.data = QtConcurrent::blockingMappedReduced(work_items, compute_epoch_csd, accumulate_csd);
    
    if (csd.data.empty()) {
         csd.data.resize(n_freqs_kept, Eigen::MatrixXcd::Zero(n_channels, n_channels));
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
