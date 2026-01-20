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

std::pair<Eigen::MatrixXd, Eigen::VectorXd> TFRLIB::PSD::psd_array_welch(const Eigen::MatrixXd& data,
                                                                       double sfreq,
                                                                       double fmin,
                                                                       double fmax,
                                                                       int n_fft,
                                                                       int n_overlap,
                                                                       int n_per_seg,
                                                                       const std::string& window,
                                                                       const std::string& detrend,
                                                                       const std::string& scaling)
{
    int n_channels = data.rows();
    int n_samples = data.cols();
    
    // Set default parameters
    if (n_per_seg <= 0) n_per_seg = n_fft;
    if (n_overlap < 0) n_overlap = compute_optimal_overlap(n_per_seg, 0.5);
    if (fmax < 0) fmax = sfreq / 2.0;
    
    // Generate optimized window
    Eigen::VectorXd win = generate_window(window, n_per_seg);
    
    // Compute window normalization factors
    double win_sum_sq = win.squaredNorm();
    double scale_factor;
    if (scaling == "density") {
        scale_factor = 1.0 / (sfreq * win_sum_sq);
    } else { // spectrum
        scale_factor = 1.0 / (win.sum() * win.sum());
    }
    
    // Compute step size
    int step = n_per_seg - n_overlap;
    if (step <= 0) step = 1;
    
    // Frequency vector
    int n_freqs_all = n_fft / 2 + 1;
    Eigen::VectorXd all_freqs(n_freqs_all);
    for (int i = 0; i < n_freqs_all; ++i) {
        all_freqs[i] = static_cast<double>(i) * sfreq / static_cast<double>(n_fft);
    }
    
    // Find frequency indices within range
    std::vector<int> freq_indices;
    for (int i = 0; i < n_freqs_all; ++i) {
        if (all_freqs[i] >= fmin && all_freqs[i] <= fmax) {
            freq_indices.push_back(i);
        }
    }
    
    int n_freqs_kept = freq_indices.size();
    Eigen::MatrixXd psds = Eigen::MatrixXd::Zero(n_channels, n_freqs_kept);
    Eigen::VectorXd freqs(n_freqs_kept);
    for (int i = 0; i < n_freqs_kept; ++i) {
        freqs[i] = all_freqs[freq_indices[i]];
    }
    
    Eigen::FFT<double> fft;
    
    // Process each channel
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        int seg_count = 0;
        
        // Process overlapping segments
        for (int start = 0; start <= n_samples - n_per_seg; start += step) {
            Eigen::VectorXd segment = signal.segment(start, n_per_seg);
            
            // Apply detrending
            apply_detrend(segment, detrend);
            
            // Apply window
            segment.array() *= win.array();
            
            // Zero pad to n_fft
            std::vector<double> time_domain(n_fft, 0.0);
            for (int i = 0; i < n_per_seg; ++i) {
                time_domain[i] = segment[i];
            }
            
            // Compute FFT
            std::vector<std::complex<double>> freq_domain(n_fft);
            fft.fwd(freq_domain, time_domain);
            
            // Compute periodogram for selected frequencies
            for (int f_idx = 0; f_idx < n_freqs_kept; ++f_idx) {
                int f = freq_indices[f_idx];
                double abs_val = std::abs(freq_domain[f]);
                double p = abs_val * abs_val * scale_factor;
                
                // One-sided scaling for positive frequencies
                if (scaling == "density") {
                    if (n_fft % 2 == 1) {
                        if (f > 0) p *= 2.0;
                    } else {
                        if (f > 0 && f < n_freqs_all - 1) p *= 2.0;
                    }
                }
                
                psds(ch, f_idx) += p;
            }
            seg_count++;
        }
        
        // Average over segments
        if (seg_count > 0) {
            psds.row(ch) /= static_cast<double>(seg_count);
        }
    }
    
    return std::make_pair(psds, freqs);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> TFRLIB::PSD::psd_array_multitaper(const Eigen::MatrixXd& data,
                                                                      double sfreq,
                                                                      double fmin,
                                                                      double fmax,
                                                                      double bandwidth,
                                                                      bool adaptive,
                                                                      bool low_bias,
                                                                      const std::string& normalization)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    // Set default parameters
    if (bandwidth == 0.0) {
        bandwidth = 4.0 / (static_cast<double>(n_times) / sfreq);
    }
    if (fmax < 0) fmax = sfreq / 2.0;
    
    // Compute multitaper parameters
    double nw = bandwidth * n_times / (2.0 * sfreq);
    int k_max = static_cast<int>(2.0 * nw) - 1;
    if (k_max < 1) k_max = 1;
    
    // Generate DPSS tapers
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> dpss = TFRUtils::dpss_windows(n_times, nw, k_max);
    const Eigen::MatrixXd& tapers = dpss.first;
    const Eigen::VectorXd& eigen_vals = dpss.second;
    
    // Select valid tapers
    std::vector<int> valid_tapers;
    std::vector<double> taper_weights;
    
    for (int i = 0; i < k_max; ++i) {
        if (!low_bias || eigen_vals[i] > 0.9) {
            valid_tapers.push_back(i);
            
            if (adaptive) {
                // Adaptive weighting based on eigenvalue
                double weight = eigen_vals[i] / (1.0 - eigen_vals[i] + 1e-10);
                taper_weights.push_back(weight);
            } else {
                taper_weights.push_back(1.0);
            }
        }
    }
    
    if (valid_tapers.empty()) {
        valid_tapers.push_back(0);
        taper_weights.push_back(1.0);
    }
    
    // Normalize weights
    if (adaptive) {
        double weight_sum = 0.0;
        for (double w : taper_weights) weight_sum += w;
        for (double& w : taper_weights) w /= weight_sum;
    } else {
        double uniform_weight = 1.0 / taper_weights.size();
        for (double& w : taper_weights) w = uniform_weight;
    }
    
    // Frequency setup
    int n_fft = n_times;
    int n_freqs_all = n_fft / 2 + 1;
    
    // Find frequency indices within range
    std::vector<int> freq_indices;
    Eigen::VectorXd freqs;
    
    for (int i = 0; i < n_freqs_all; ++i) {
        double f = static_cast<double>(i) * sfreq / static_cast<double>(n_fft);
        if (f >= fmin && f <= fmax) {
            freq_indices.push_back(i);
            freqs.conservativeResize(freqs.size() + 1);
            freqs[freqs.size() - 1] = f;
        }
    }
    
    int n_freqs_kept = freq_indices.size();
    Eigen::MatrixXd psds = Eigen::MatrixXd::Zero(n_channels, n_freqs_kept);
    
    Eigen::FFT<double> fft;
    
    // Process each channel
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        
        // Remove mean
        double mean = signal.mean();
        signal.array() -= mean;
        
        // Process each taper
        for (size_t t_idx = 0; t_idx < valid_tapers.size(); ++t_idx) {
            int taper_idx = valid_tapers[t_idx];
            double weight = taper_weights[t_idx];
            
            Eigen::VectorXd window = tapers.col(taper_idx);
            Eigen::VectorXd windowed_sig = signal.array() * window.array();
            
            // Apply normalization
            if (normalization == "full") {
                double norm_factor = window.norm();
                if (norm_factor > 0) windowed_sig /= norm_factor;
            } else if (normalization == "length") {
                windowed_sig /= std::sqrt(static_cast<double>(n_times));
            }
            
            // Prepare for FFT
            std::vector<double> time_domain(n_fft, 0.0);
            for (int i = 0; i < n_times; ++i) {
                time_domain[i] = windowed_sig[i];
            }
            
            // Compute FFT
            std::vector<std::complex<double>> freq_domain(n_fft);
            fft.fwd(freq_domain, time_domain);
            
            // Accumulate power spectrum
            for (int f_idx = 0; f_idx < n_freqs_kept; ++f_idx) {
                int f = freq_indices[f_idx];
                double abs_val = std::abs(freq_domain[f]);
                double p = abs_val * abs_val;
                
                // One-sided scaling
                if (n_fft % 2 == 1) {
                    if (f > 0) p *= 2.0;
                } else {
                    if (f > 0 && f < n_freqs_all - 1) p *= 2.0;
                }
                
                // Scale by sampling frequency
                p /= sfreq;
                
                psds(ch, f_idx) += weight * p;
            }
        }
    }
    
    return std::make_pair(psds, freqs);
}

Eigen::VectorXd TFRLIB::PSD::generate_window(const std::string& window_name, int n_samples, double beta)
{
    Eigen::VectorXd window(n_samples);
    
    if (window_name == "hamming") {
        for (int i = 0; i < n_samples; ++i) {
            window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (n_samples - 1));
        }
    } else if (window_name == "hanning" || window_name == "hann") {
        for (int i = 0; i < n_samples; ++i) {
            window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (n_samples - 1)));
        }
    } else if (window_name == "blackman") {
        for (int i = 0; i < n_samples; ++i) {
            double x = 2.0 * M_PI * i / (n_samples - 1);
            window[i] = 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);
        }
    } else if (window_name == "kaiser") {
        // Kaiser window with beta parameter
        // Custom implementation of modified Bessel function I0
        auto bessel_i0 = [](double x) -> double {
            double sum = 1.0;
            double term = 1.0;
            double x_half_sq = (x * 0.5) * (x * 0.5);
            
            for (int k = 1; k < 50; ++k) {
                term *= x_half_sq / (k * k);
                sum += term;
                if (term < 1e-12 * sum) break;
            }
            return sum;
        };
        
        double alpha = (n_samples - 1) / 2.0;
        double i0_beta = bessel_i0(beta);
        
        for (int i = 0; i < n_samples; ++i) {
            double x = (i - alpha) / alpha;
            double arg = beta * std::sqrt(std::max(0.0, 1.0 - x * x));
            window[i] = bessel_i0(arg) / i0_beta;
        }
    } else if (window_name == "tukey") {
        // Tukey window (tapered cosine)
        double alpha = 0.5; // Taper fraction
        int taper_samples = static_cast<int>(alpha * n_samples / 2.0);
        
        for (int i = 0; i < n_samples; ++i) {
            if (i < taper_samples) {
                window[i] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * i / (alpha * n_samples) - 1.0)));
            } else if (i >= n_samples - taper_samples) {
                window[i] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * (i - n_samples + taper_samples) / (alpha * n_samples) + 1.0)));
            } else {
                window[i] = 1.0;
            }
        }
    } else {
        // Default: rectangular window
        window.setOnes();
    }
    
    return window;
}

void TFRLIB::PSD::apply_detrend(Eigen::VectorXd& segment, const std::string& method)
{
    if (method == "constant" || method == "mean") {
        // Remove mean
        double mean = segment.mean();
        segment.array() -= mean;
    } else if (method == "linear") {
        // Remove linear trend
        int n = segment.size();
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        
        for (int i = 0; i < n; ++i) {
            sum_x += i;
            sum_y += segment[i];
            sum_xy += i * segment[i];
            sum_x2 += i * i;
        }
        
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;
        
        for (int i = 0; i < n; ++i) {
            segment[i] -= (slope * i + intercept);
        }
    }
    // For "none", do nothing
}

int TFRLIB::PSD::compute_optimal_overlap(int n_per_seg, double overlap_ratio)
{
    int overlap = static_cast<int>(n_per_seg * overlap_ratio);
    
    // Ensure overlap is reasonable
    if (overlap >= n_per_seg) {
        overlap = n_per_seg - 1;
    }
    if (overlap < 0) {
        overlap = 0;
    }
    
    return overlap;
}

} // NAMESPACE