#include "tfr_utils.h"
#include <cmath>
#include <iostream>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace TFRLIB {

std::vector<Eigen::VectorXcd> TFRUtils::morlet(double sfreq,
                                               const Eigen::VectorXd& freqs,
                                               double n_cycles,
                                               double sigma,
                                               bool zero_mean)
{
    std::vector<Eigen::VectorXcd> wavelets;
    wavelets.reserve(freqs.size());

    for (int i = 0; i < freqs.size(); ++i) {
        double f = freqs[i];
        double sigma_t;
        
        if (sigma == 0.0) {
            sigma_t = n_cycles / (2.0 * M_PI * f);
        } else {
            // MNE logic: sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
            // But wait, in MNE python:
            // if sigma is None: sigma_t = n_cycles / (2*pi*f)
            // else: sigma_t = n_cycles / (2*pi*sigma)
            // This seems to imply 'sigma' parameter in MNE is not 'sigma_t' directly but some scaling factor?
            // Doc says: "sigma : float, default None. It controls the width of the wavelet ie its temporal resolution."
            // "If sigma is fixed the temporal resolution is fixed like for STFT and number of oscillations increases with frequency."
            // So if sigma is provided, sigma_t is derived differently?
            // Actually, looking at MNE code:
            // if sigma is None: sigma_t = n_cycles / (2.0 * np.pi * f)
            // else: sigma_t = n_cycles / (2.0 * np.pi * sigma)
            // This looks weird if sigma is supposed to be 'sigma_t'. 
            // If sigma is meant to be 'sigma in Hz' or something?
            // Actually, let's assume standard usage where sigma is None (0.0 in C++).
            // If the user passes sigma != 0, we follow MNE formula.
            sigma_t = n_cycles / (2.0 * M_PI * sigma);
        }

        // t = np.arange(0.0, 5.0 * sigma_t, 1.0 / sfreq)
        // t = np.r_[-t[::-1], t[1:]]
        
        double step = 1.0 / sfreq;
        double limit = 5.0 * sigma_t;
        
        // Count samples for [0, limit)
        // np.arange(start, stop, step) includes start, excludes stop.
        // But floating point arithmetic...
        // Let's replicate exact count.
        std::vector<double> t_half;
        for (double t = 0.0; t < limit; t += step) {
            t_half.push_back(t); 
            // Note: C++ loop condition might behave differently at boundary than numpy.arange
            // numpy arange usually stops before limit.
            // Safe bet: while (t < limit - epsilon)
        }
        
        // Construct full t
        // -t[::-1] means reverse and negate.
        // t[1:] means skip 0.
        // So we want -t_last, ..., -t_1, 0, t_1, ..., t_last
        
        int n_half = t_half.size();
        int n_total = (n_half * 2) - 1; 
        
        // However, MNE logic: t = np.r_[-t[::-1], t[1:]]
        // If t_half = [0, 0.1, 0.2]
        // -t[::-1] = [-0.2, -0.1, -0.0] (Wait, -0.0 is 0)
        // t[1:] = [0.1, 0.2]
        // Result: [-0.2, -0.1, 0, 0.1, 0.2]
        // Correct.
        
        Eigen::VectorXcd W(n_total);
        
        for (int k = 0; k < n_total; ++k) {
            double t_val;
            if (k < n_half - 1) {
                // Negative part: index 0 corresponds to -t_half[last], index n_half-2 corresponds to -t_half[1]
                // index in t_half: (n_half - 1) - k
                // Wait.
                // k=0 -> want -t_half[n_half-1]
                // k=n_half-2 -> want -t_half[1]
                t_val = -t_half[n_half - 1 - k];
            } else {
                // Positive part (including 0): index n_half-1 corresponds to 0 (t_half[0])
                // index in t_half: k - (n_half - 1)
                t_val = t_half[k - (n_half - 1)];
            }
            
            // Oscillation
            std::complex<double> oscillation = std::exp(std::complex<double>(0.0, 2.0 * M_PI * f * t_val));
            
            if (zero_mean) {
                 double real_offset = std::exp(-2.0 * std::pow(M_PI * f * sigma_t, 2));
                 oscillation -= std::complex<double>(real_offset, 0.0);
            }
            
            // Envelope
            double gaussian_envelope = std::exp(-(t_val * t_val) / (2.0 * sigma_t * sigma_t));
            
            W[k] = oscillation * gaussian_envelope;
        }
        
        // Normalization
        // W /= np.sqrt(0.5) * np.linalg.norm(W.ravel())
        double norm = W.norm(); // L2 norm
        double factor = std::sqrt(0.5) * norm;
        W /= factor;

        wavelets.push_back(W);
    }

    return wavelets;
}

} // NAMESPACE
