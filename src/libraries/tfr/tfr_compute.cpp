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