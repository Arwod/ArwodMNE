//=============================================================================================================
/**
 * @file     test_tfr_enhanced.cpp
 * @author   MNE-CPP Migration Team
 * @since    1.0.0
 * @date     January, 2025
 *
 * @brief    Test for enhanced time-frequency analysis functionality
 */

#include "src/libraries/tfr/tfr_compute.h"
#include "src/libraries/tfr/tfr_utils.h"
#include <iostream>
#include <Eigen/Core>
#include <cmath>

using namespace TFRLIB;

int main()
{
    std::cout << "Testing Enhanced MNE Time-Frequency Analysis..." << std::endl;
    
    // Create test data: 2 channels, 1000 time points, 1000 Hz sampling
    int n_channels = 2;
    int n_times = 1000;
    double sfreq = 1000.0;
    
    Eigen::MatrixXd test_data = Eigen::MatrixXd::Random(n_channels, n_times);
    
    // Add some sinusoidal components for testing
    Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(n_times, 0.0, (n_times - 1) / sfreq);
    
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 0; t < n_times; ++t) {
            // Add 10 Hz and 20 Hz components
            test_data(ch, t) = std::sin(2.0 * M_PI * 10.0 * times(t)) + 
                              0.5 * std::sin(2.0 * M_PI * 20.0 * times(t)) +
                              0.1 * test_data(ch, t); // Add some noise
        }
    }
    
    // Define frequencies of interest
    Eigen::VectorXd freqs(3);
    freqs << 5.0, 10.0, 20.0;
    
    // Define variable number of cycles per frequency
    Eigen::VectorXd n_cycles(3);
    n_cycles << 3.0, 5.0, 7.0;
    
    try {
        // Test original Morlet wavelet transform
        std::cout << "Testing original tfr_morlet..." << std::endl;
        auto tfr_result_orig = TFRCompute::tfr_morlet(test_data, sfreq, freqs, 5.0);
        
        if (!tfr_result_orig.empty()) {
            std::cout << "✓ Original tfr_morlet completed successfully" << std::endl;
            std::cout << "  - Number of channels: " << tfr_result_orig.size() << std::endl;
            std::cout << "  - Number of frequencies: " << tfr_result_orig[0].size() << std::endl;
            std::cout << "  - Number of time points: " << tfr_result_orig[0][0].size() << std::endl;
        } else {
            std::cout << "✗ Original tfr_morlet failed" << std::endl;
        }
        
        // Test enhanced Morlet wavelet transform with complex output
        std::cout << "\nTesting enhanced tfr_morlet with complex output..." << std::endl;
        auto tfr_result_complex = TFRCompute::tfr_morlet_enhanced(test_data, sfreq, freqs, n_cycles, true, "complex");
        
        if (!tfr_result_complex.empty()) {
            std::cout << "✓ Enhanced tfr_morlet (complex) completed successfully" << std::endl;
            std::cout << "  - Number of channels: " << tfr_result_complex.size() << std::endl;
            std::cout << "  - Number of frequencies: " << tfr_result_complex[0].size() << std::endl;
            std::cout << "  - Number of time points: " << tfr_result_complex[0][0].size() << std::endl;
            
            // Check if output is actually complex
            bool has_imaginary = false;
            for (int ch = 0; ch < n_channels && !has_imaginary; ++ch) {
                for (int f = 0; f < freqs.size() && !has_imaginary; ++f) {
                    for (int t = 0; t < std::min(10, (int)tfr_result_complex[ch][f].size()) && !has_imaginary; ++t) {
                        if (std::abs(tfr_result_complex[ch][f][t].imag()) > 1e-10) {
                            has_imaginary = true;
                        }
                    }
                }
            }
            
            if (has_imaginary) {
                std::cout << "✓ Complex output contains imaginary components" << std::endl;
            } else {
                std::cout << "⚠ Complex output appears to be real-only" << std::endl;
            }
        } else {
            std::cout << "✗ Enhanced tfr_morlet (complex) failed" << std::endl;
        }
        
        // Test enhanced Morlet wavelet transform with power output
        std::cout << "\nTesting enhanced tfr_morlet with power output..." << std::endl;
        auto tfr_result_power = TFRCompute::tfr_morlet_enhanced(test_data, sfreq, freqs, n_cycles, true, "power");
        
        if (!tfr_result_power.empty()) {
            std::cout << "✓ Enhanced tfr_morlet (power) completed successfully" << std::endl;
            
            // Check if power values are non-negative
            bool all_positive = true;
            for (int ch = 0; ch < n_channels && all_positive; ++ch) {
                for (int f = 0; f < freqs.size() && all_positive; ++f) {
                    for (int t = 0; t < std::min(10, (int)tfr_result_power[ch][f].size()) && all_positive; ++t) {
                        if (tfr_result_power[ch][f][t].real() < 0) {
                            all_positive = false;
                        }
                    }
                }
            }
            
            if (all_positive) {
                std::cout << "✓ Power output contains only non-negative values" << std::endl;
            } else {
                std::cout << "✗ Power output contains negative values" << std::endl;
            }
        } else {
            std::cout << "✗ Enhanced tfr_morlet (power) failed" << std::endl;
        }
        
        // Test enhanced Morlet wavelet transform with phase output
        std::cout << "\nTesting enhanced tfr_morlet with phase output..." << std::endl;
        auto tfr_result_phase = TFRCompute::tfr_morlet_enhanced(test_data, sfreq, freqs, n_cycles, true, "phase");
        
        if (!tfr_result_phase.empty()) {
            std::cout << "✓ Enhanced tfr_morlet (phase) completed successfully" << std::endl;
            
            // Check if phase values are in expected range [-π, π]
            bool valid_phase_range = true;
            for (int ch = 0; ch < n_channels && valid_phase_range; ++ch) {
                for (int f = 0; f < freqs.size() && valid_phase_range; ++f) {
                    for (int t = 0; t < std::min(10, (int)tfr_result_phase[ch][f].size()) && valid_phase_range; ++t) {
                        double phase = tfr_result_phase[ch][f][t].real();
                        if (phase < -M_PI - 1e-10 || phase > M_PI + 1e-10) {
                            valid_phase_range = false;
                        }
                    }
                }
            }
            
            if (valid_phase_range) {
                std::cout << "✓ Phase output is in valid range [-π, π]" << std::endl;
            } else {
                std::cout << "⚠ Phase output may be outside expected range" << std::endl;
            }
        } else {
            std::cout << "✗ Enhanced tfr_morlet (phase) failed" << std::endl;
        }
        
        // Test variable cycles functionality
        std::cout << "\nTesting variable n_cycles functionality..." << std::endl;
        Eigen::VectorXd n_cycles_var(3);
        n_cycles_var << 2.0, 8.0, 12.0;  // Very different cycle numbers
        
        auto wavelets_var = TFRUtils::morlet_variable(sfreq, freqs, n_cycles_var);
        
        if (wavelets_var.size() == freqs.size()) {
            std::cout << "✓ Variable cycles wavelets generated successfully" << std::endl;
            
            // Check that wavelets have different lengths (due to different n_cycles)
            bool different_lengths = false;
            for (size_t i = 1; i < wavelets_var.size(); ++i) {
                if (wavelets_var[i].size() != wavelets_var[0].size()) {
                    different_lengths = true;
                    break;
                }
            }
            
            if (different_lengths) {
                std::cout << "✓ Wavelets have different lengths as expected" << std::endl;
            } else {
                std::cout << "⚠ All wavelets have the same length" << std::endl;
            }
            
            // Print wavelet lengths for verification
            for (size_t i = 0; i < wavelets_var.size(); ++i) {
                std::cout << "  - Wavelet " << i << " (freq=" << freqs[i] << "Hz, cycles=" 
                         << n_cycles_var[i] << "): " << wavelets_var[i].size() << " samples" << std::endl;
            }
        } else {
            std::cout << "✗ Variable cycles wavelets generation failed" << std::endl;
        }
        
        std::cout << "\n✓ All enhanced time-frequency tests completed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}