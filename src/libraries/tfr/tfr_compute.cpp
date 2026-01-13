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
std::vector<std::vector<Eigen::VectorXcd>> TFRLIB::TFRCompute::tfr_morlet_enhanced(const Eigen::MatrixXd& data,
                                                                                   double sfreq,
                                                                                   const Eigen::VectorXd& freqs,
                                                                                   const Eigen::VectorXd& n_cycles,
                                                                                   bool use_fft,
                                                                                   const std::string& output,
                                                                                   int decim)
{
    // Validate input
    if (n_cycles.size() != freqs.size()) {
        throw std::invalid_argument("n_cycles must have the same size as freqs");
    }

    // Generate wavelets with variable cycles
    std::vector<Eigen::VectorXcd> wavelets = TFRUtils::morlet_variable(sfreq, freqs, n_cycles);
    
    int n_channels = data.rows();
    int n_times = data.cols();
    int n_freqs = freqs.size();
    
    // Output structure: [channel][freq] -> complex time_series
    std::vector<std::vector<Eigen::VectorXcd>> tfr_complex(n_channels, std::vector<Eigen::VectorXcd>(n_freqs));
    
    // Convolve each channel with each wavelet
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        
        for (int f = 0; f < n_freqs; ++f) {
            Eigen::VectorXcd W = wavelets[f];
            
            // Complex convolution
            Eigen::VectorXd W_real = W.real();
            Eigen::VectorXd W_imag = W.imag();
            
            Eigen::VectorXd conv_r = UTILSLIB::MNEMath::convolve(signal, W_real, std::string("same"));
            Eigen::VectorXd conv_i = UTILSLIB::MNEMath::convolve(signal, W_imag, std::string("same"));
            
            // Construct complex result
            Eigen::VectorXcd complex_result(conv_r.size());
            for (int t = 0; t < complex_result.size(); ++t) {
                complex_result[t] = std::complex<double>(conv_r[t], conv_i[t]);
            }
            
            // Apply decimation if requested
            if (decim > 1) {
                int n_out = (complex_result.size() + decim - 1) / decim;
                Eigen::VectorXcd decimated_result(n_out);
                for (int i = 0; i < n_out; ++i) {
                    if (i * decim < complex_result.size()) {
                        decimated_result[i] = complex_result[i * decim];
                    }
                }
                tfr_complex[ch][f] = decimated_result;
            } else {
                tfr_complex[ch][f] = complex_result;
            }
        }
    }
    
    // Convert output based on requested format
    if (output == "complex") {
        return tfr_complex;
    } else if (output == "power") {
        // Convert to power and return as complex with zero imaginary part
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int f = 0; f < n_freqs; ++f) {
                Eigen::VectorXcd& complex_data = tfr_complex[ch][f];
                for (int t = 0; t < complex_data.size(); ++t) {
                    double power = std::norm(complex_data[t]);
                    complex_data[t] = std::complex<double>(power, 0.0);
                }
            }
        }
        return tfr_complex;
    } else if (output == "phase") {
        // Convert to phase and return as complex with zero imaginary part
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int f = 0; f < n_freqs; ++f) {
                Eigen::VectorXcd& complex_data = tfr_complex[ch][f];
                for (int t = 0; t < complex_data.size(); ++t) {
                    double phase = std::arg(complex_data[t]);
                    complex_data[t] = std::complex<double>(phase, 0.0);
                }
            }
        }
        return tfr_complex;
    } else {
        // Default to complex output
        return tfr_complex;
    }
}

std::vector<std::vector<Eigen::VectorXcd>> TFRLIB::TFRCompute::tfr_multitaper(const Eigen::MatrixXd& data,
                                                                      double sfreq,
                                                                      const Eigen::VectorXd& freqs,
                                                                      const Eigen::VectorXd& n_cycles,
                                                                      double time_bandwidth,
                                                                      bool use_fft,
                                                                      const std::string& output,
                                                                      int decim)
{
    // Placeholder implementation for multitaper method
    // For now, fall back to enhanced Morlet method
    // TODO: Implement proper multitaper method using DPSS tapers
    
    std::cout << "Warning: tfr_multitaper not fully implemented, falling back to Morlet method" << std::endl;
    return tfr_morlet_enhanced(data, sfreq, freqs, n_cycles, use_fft, output, decim);
}
std::vector<std::vector<Eigen::VectorXcd>> TFRLIB::TFRCompute::tfr_stockwell(const Eigen::MatrixXd& data,
                                                                                    double sfreq,
                                                                                    double fmin,
                                                                                    double fmax,
                                                                                    int n_fft,
                                                                                    double width,
                                                                                    int decim)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    // Set default parameters
    if (fmax < 0) {
        fmax = sfreq / 2.0;
    }
    if (n_fft < 0) {
        n_fft = n_times;
    }
    
    // Ensure n_fft is at least as large as n_times
    if (n_fft < n_times) {
        n_fft = n_times;
    }
    
    // Generate frequency vector
    double df = sfreq / n_fft;
    std::vector<double> freqs;
    for (double f = fmin; f <= fmax; f += df) {
        freqs.push_back(f);
    }
    int n_freqs = freqs.size();
    
    // Output structure: [channel][freq] -> complex time_series
    std::vector<std::vector<Eigen::VectorXcd>> stft_result(n_channels, std::vector<Eigen::VectorXcd>(n_freqs));
    
    // Process each channel
    for (int ch = 0; ch < n_channels; ++ch) {
        Eigen::VectorXd signal = data.row(ch);
        
        // Pad signal to n_fft length if necessary
        Eigen::VectorXcd padded_signal = Eigen::VectorXcd::Zero(n_fft);
        for (int i = 0; i < std::min(n_times, n_fft); ++i) {
            padded_signal[i] = std::complex<double>(signal[i], 0.0);
        }
        
        // Compute FFT of the signal (placeholder - would use proper FFT library)
        // For now, we'll implement a simplified version
        
        // Process each frequency
        for (int f_idx = 0; f_idx < n_freqs; ++f_idx) {
            double freq = freqs[f_idx];
            
            // Skip DC component for Stockwell transform
            if (freq == 0.0) {
                stft_result[ch][f_idx] = Eigen::VectorXcd::Zero(n_times);
                continue;
            }
            
            // Calculate Gaussian window width
            // In Stockwell transform, window width is inversely proportional to frequency
            double sigma = width / freq;
            
            // Generate Gaussian window
            int window_length = static_cast<int>(6.0 * sigma * sfreq); // 6 sigma window
            if (window_length % 2 == 0) window_length++; // Make odd
            
            Eigen::VectorXcd window = Eigen::VectorXcd::Zero(window_length);
            int half_window = window_length / 2;
            
            for (int i = 0; i < window_length; ++i) {
                double t = (i - half_window) / sfreq;
                double gaussian = std::exp(-0.5 * t * t / (sigma * sigma));
                std::complex<double> complex_exp = std::exp(std::complex<double>(0.0, -2.0 * M_PI * freq * t));
                window[i] = gaussian * complex_exp;
            }
            
            // Normalize window
            double norm = 0.0;
            for (int i = 0; i < window_length; ++i) {
                norm += std::norm(window[i]);
            }
            if (norm > 0) {
                window /= std::sqrt(norm);
            }
            
            // Convolve signal with window (simplified convolution)
            Eigen::VectorXcd convolved = Eigen::VectorXcd::Zero(n_times);
            
            for (int t = 0; t < n_times; ++t) {
                std::complex<double> sum(0.0, 0.0);
                
                for (int w = 0; w < window_length; ++w) {
                    int signal_idx = t - half_window + w;
                    if (signal_idx >= 0 && signal_idx < n_times) {
                        sum += signal[signal_idx] * std::conj(window[w]);
                    }
                }
                
                convolved[t] = sum;
            }
            
            // Apply decimation if requested
            if (decim > 1) {
                int n_out = (n_times + decim - 1) / decim;
                Eigen::VectorXcd decimated_result(n_out);
                for (int i = 0; i < n_out; ++i) {
                    if (i * decim < n_times) {
                        decimated_result[i] = convolved[i * decim];
                    }
                }
                stft_result[ch][f_idx] = decimated_result;
            } else {
                stft_result[ch][f_idx] = convolved;
            }
        }
    }
    
    return stft_result;
}