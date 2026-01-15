#include "psd.h"
#include "tfr_utils.h"
#include <utils/mnemath.h>
#include <unsupported/Eigen/FFT>
#include <iostream>

namespace TFRLIB {

std::pair<Eigen::MatrixXd, Eigen::VectorXd> PSD::psd_multitaper(const Eigen::MatrixXd& data,
                                                                double sfreq,
                                                                double bandwidth,
                                                                bool adaptive,
                                                                bool low_bias)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    if (bandwidth == 0.0) {
        bandwidth = 4.0 / ((double)n_times / sfreq);
    }
    
    double nw = bandwidth * n_times / (2.0 * sfreq);
    int k_max = (int)(2.0 * nw) - 1;
    if (k_max < 1) k_max = 1;
    
    // Get DPSS windows
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
    int n_fft = n_times; 
    int n_freqs = n_fft / 2 + 1;
    
    Eigen::MatrixXd psds = Eigen::MatrixXd::Zero(n_channels, n_freqs);
    Eigen::FFT<double> fft;
    
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        double mean = signal.mean();
        signal.array() -= mean;
        
        for (int k_idx : valid_tapers) {
            Eigen::VectorXd window = tapers.col(k_idx);
            Eigen::VectorXd windowed_sig = signal.array() * window.array();
            
            std::vector<double> time_domain(n_fft, 0.0);
            for(int i=0; i<n_times; ++i) time_domain[i] = windowed_sig[i];
            
            std::vector<std::complex<double>> freq_domain(n_fft);
            fft.fwd(freq_domain, time_domain);
            
            for (int f = 0; f < n_freqs; ++f) {
                double abs_val = std::abs(freq_domain[f]);
                double p = abs_val * abs_val;
                
                if (n_fft % 2 == 1) {
                    if (f > 0) p *= 2.0;
                } else {
                    if (f > 0 && f < n_freqs - 1) p *= 2.0;
                }
                
                p /= sfreq;
                
                psds(ch, f) += p;
            }
        }
        
        psds.row(ch) /= (double)n_tapers;
    }
    
    Eigen::VectorXd freqs(n_freqs);
    for(int i=0; i<n_freqs; ++i) {
        freqs[i] = (double)i * sfreq / (double)n_fft;
    }
    
    return std::make_pair(psds, freqs);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> PSD::psd_welch(const Eigen::MatrixXd& data,
                                                           double sfreq,
                                                           int n_fft,
                                                           int n_overlap,
                                                           int n_per_seg,
                                                           const std::string& window_name)
{
    if (n_per_seg <= 0) n_per_seg = n_fft;
    
    int n_channels = data.rows();
    int n_samples = data.cols();
    
    // Step (stride)
    int step = n_per_seg - n_overlap;
    if (step <= 0) step = 1; // Prevent infinite loop
    
    // Window
    Eigen::VectorXd win;
    if (window_name == "hamming") {
        win = UTILSLIB::MNEMath::hamming(n_per_seg);
    } else if (window_name == "hanning") {
        win = UTILSLIB::MNEMath::hanning(n_per_seg);
    } else if (window_name == "blackman") {
        win = UTILSLIB::MNEMath::blackman(n_per_seg);
    } else {
        // Default rectangular
        win = Eigen::VectorXd::Ones(n_per_seg);
    }
    
    double win_sum_sq = win.squaredNorm();
    double scale = 1.0 / (sfreq * win_sum_sq);
    
    int n_freqs = n_fft / 2 + 1;
    Eigen::MatrixXd psds = Eigen::MatrixXd::Zero(n_channels, n_freqs);
    
    Eigen::FFT<double> fft;
    
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        int seg_count = 0;
        
        for (int start = 0; start <= n_samples - n_per_seg; start += step) {
            Eigen::VectorXd segment = signal.segment(start, n_per_seg);
            
            // Detrend (constant)
            double mean = segment.mean();
            segment.array() -= mean;
            
            // Apply window
            segment.array() *= win.array();
            
            // Zero pad to n_fft
            std::vector<double> time_domain(n_fft, 0.0);
            for(int i=0; i<n_per_seg; ++i) time_domain[i] = segment[i];
            
            // FFT
            std::vector<std::complex<double>> freq_domain(n_fft);
            fft.fwd(freq_domain, time_domain);
            
            // Compute periodogram |X|^2
            for(int f=0; f<n_freqs; ++f) {
                double abs_val = std::abs(freq_domain[f]);
                double p = abs_val * abs_val * scale;
                
                // One-sided scaling
                if (n_fft % 2 == 1) {
                    if (f > 0) p *= 2.0;
                } else {
                    if (f > 0 && f < n_freqs - 1) p *= 2.0;
                }

                psds(ch, f) += p;
            }
            seg_count++;
        }
        
        if (seg_count > 0) {
            psds.row(ch) /= (double)seg_count;
        }
    }
    
    // Frequencies
    Eigen::VectorXd freqs(n_freqs);
    for(int i=0; i<n_freqs; ++i) {
        freqs[i] = (double)i * sfreq / (double)n_fft;
    }
    
    return std::make_pair(psds, freqs);
}

} // NAMESPACE
