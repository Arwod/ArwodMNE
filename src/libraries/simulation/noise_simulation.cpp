#include "noise_simulation.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace SIMULATIONLIB
{

//=============================================================================================================

NoiseSimulation::NoiseSimulation(const FIFFLIB::FiffInfo::SPtr& info, int random_seed)
: m_pInfo(info)
, m_normalDist(0.0, 1.0)
, m_uniformDist(0.0, 1.0)
{
    // Initialize dimensions
    m_nChannels = m_pInfo->nchan;
    m_samplingFreq = m_pInfo->sfreq;
    
    // Set random seed
    if (random_seed >= 0) {
        m_randomGenerator.seed(random_seed);
    } else {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        m_randomGenerator.seed(seed);
    }
    
    // Detect channel types
    detectChannelTypes();
    
    std::cout << "NoiseSimulation initialized with " << m_nChannels << " channels" << std::endl;
    std::cout << "  - " << m_magChannels.size() << " magnetometers" << std::endl;
    std::cout << "  - " << m_gradChannels.size() << " gradiometers" << std::endl;
    std::cout << "  - " << m_eegChannels.size() << " EEG channels" << std::endl;
}

//=============================================================================================================

NoiseSimulation::ECGParams NoiseSimulation::getDefaultECGParams()
{
    ECGParams params;
    params.heart_rate = 72.0;
    params.amplitude = 5e-12;
    params.qrs_width = 0.1;
    params.p_wave_amplitude = 0.2;
    params.t_wave_amplitude = 0.3;
    params.affected_channels.clear(); // All channels
    params.add_variability = true;
    params.variability_std = 5.0;
    return params;
}

//=============================================================================================================

NoiseSimulation::EOGParams NoiseSimulation::getDefaultEOGParams()
{
    EOGParams params;
    params.blink_rate = 15.0;
    params.amplitude = 10e-12;
    params.blink_duration = 0.2;
    params.saccade_rate = 180.0;
    params.saccade_amplitude = 2e-12;
    params.frontal_channels.clear(); // Auto-detect
    params.add_slow_drifts = true;
    return params;
}

//=============================================================================================================

NoiseSimulation::CHPIParams NoiseSimulation::getDefaultCHPIParams()
{
    CHPIParams params;
    // Typical Elekta HPI frequencies
    params.frequencies = {83.0, 143.0, 203.0, 263.0, 323.0};
    params.amplitudes = {1e-11, 1e-11, 1e-11, 1e-11, 1e-11};
    
    // Default coil positions (approximate)
    params.positions = {
        Eigen::Vector3d(0.08, 0.0, 0.04),   // Left
        Eigen::Vector3d(-0.08, 0.0, 0.04),  // Right
        Eigen::Vector3d(0.0, 0.08, 0.04),   // Nasion
        Eigen::Vector3d(0.04, -0.06, 0.04), // Left back
        Eigen::Vector3d(-0.04, -0.06, 0.04) // Right back
    };
    
    params.add_movement = false;
    params.movement_amplitude = 1.0;
    params.movement_frequency = 0.1;
    return params;
}

//=============================================================================================================

NoiseSimulation::GeneralNoiseParams NoiseSimulation::getDefaultGeneralNoiseParams()
{
    GeneralNoiseParams params;
    params.white_noise_level = 1e-12;
    params.pink_noise_level = 5e-13;
    params.line_noise_freq = 50.0;
    params.line_noise_amplitude = 1e-12;
    params.harmonic_freqs = {100.0, 150.0}; // 2nd and 3rd harmonics
    params.harmonic_amps = {5e-13, 2e-13};
    params.add_channel_noise = true;
    params.random_seed = -1;
    return params;
}

//=============================================================================================================

void NoiseSimulation::addNoise(Eigen::MatrixXd& data, double noise_level)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    Eigen::MatrixXd noise = generateWhiteNoise(n_channels, n_times, noise_level);
    data += noise;
}

//=============================================================================================================

void NoiseSimulation::addECG(Eigen::MatrixXd& data, const ECGParams& params)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    double duration = n_times / m_samplingFreq;
    
    // Determine affected channels
    std::vector<int> channels = params.affected_channels;
    if (channels.empty()) {
        // Use all channels, but with different weights
        for (int ch = 0; ch < n_channels; ++ch) {
            channels.push_back(ch);
        }
    }
    
    // Calculate heart beat intervals with variability
    double base_interval = 60.0 / params.heart_rate; // seconds per beat
    std::vector<double> beat_times;
    
    double current_time = 0.0;
    while (current_time < duration) {
        beat_times.push_back(current_time);
        
        // Add heart rate variability
        double hr_variation = 0.0;
        if (params.add_variability) {
            hr_variation = m_normalDist(m_randomGenerator) * params.variability_std / 60.0;
        }
        
        current_time += base_interval + hr_variation;
    }
    
    // Generate ECG artifacts for each beat
    for (double beat_time : beat_times) {
        int beat_sample = static_cast<int>(beat_time * m_samplingFreq);
        
        // QRS complex
        int qrs_width = static_cast<int>(params.qrs_width * m_samplingFreq);
        if (beat_sample + qrs_width >= n_times) continue;
        
        Eigen::VectorXd qrs = generateGaussianWindow(qrs_width, qrs_width/2.0, qrs_width/6.0);
        qrs *= params.amplitude;
        
        // P wave (before QRS)
        int p_offset = static_cast<int>(-0.15 * m_samplingFreq); // 150ms before QRS
        int p_width = static_cast<int>(0.08 * m_samplingFreq);   // 80ms width
        if (beat_sample + p_offset >= 0 && beat_sample + p_offset + p_width < n_times) {
            Eigen::VectorXd p_wave = generateGaussianWindow(p_width, p_width/2.0, p_width/4.0);
            p_wave *= params.amplitude * params.p_wave_amplitude;
            
            for (int ch : channels) {
                double weight = 1.0 - 0.3 * (ch % 10) / 10.0; // Spatial variation
                for (int t = 0; t < p_width; ++t) {
                    data(ch, beat_sample + p_offset + t) += weight * p_wave(t);
                }
            }
        }
        
        // QRS complex
        for (int ch : channels) {
            double weight = 1.0 - 0.2 * (ch % 10) / 10.0; // Spatial variation
            for (int t = 0; t < qrs_width; ++t) {
                data(ch, beat_sample + t) += weight * qrs(t);
            }
        }
        
        // T wave (after QRS)
        int t_offset = static_cast<int>(0.25 * m_samplingFreq); // 250ms after QRS
        int t_width = static_cast<int>(0.15 * m_samplingFreq);  // 150ms width
        if (beat_sample + t_offset + t_width < n_times) {
            Eigen::VectorXd t_wave = generateGaussianWindow(t_width, t_width/2.0, t_width/3.0);
            t_wave *= params.amplitude * params.t_wave_amplitude;
            
            for (int ch : channels) {
                double weight = 1.0 - 0.4 * (ch % 10) / 10.0; // Spatial variation
                for (int t = 0; t < t_width; ++t) {
                    data(ch, beat_sample + t_offset + t) += weight * t_wave(t);
                }
            }
        }
    }
    
    std::cout << "Added ECG artifacts: " << beat_times.size() << " heartbeats" << std::endl;
}

//=============================================================================================================

void NoiseSimulation::addEOG(Eigen::MatrixXd& data, const EOGParams& params)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    double duration = n_times / m_samplingFreq;
    
    // Determine frontal channels
    std::vector<int> frontal_channels = params.frontal_channels;
    if (frontal_channels.empty()) {
        frontal_channels = m_frontalChannels;
        if (frontal_channels.empty()) {
            // Use first quarter of channels as approximation
            int n_frontal = std::min(n_channels / 4, 20);
            for (int ch = 0; ch < n_frontal; ++ch) {
                frontal_channels.push_back(ch);
            }
        }
    }
    
    // Generate blinks
    double blink_interval = 60.0 / params.blink_rate;
    std::vector<double> blink_times;
    
    double current_time = 0.0;
    while (current_time < duration) {
        blink_times.push_back(current_time);
        // Add some randomness to blink timing
        double variation = m_uniformDist(m_randomGenerator) * blink_interval * 0.5;
        current_time += blink_interval + variation;
    }
    
    // Add blink artifacts
    for (double blink_time : blink_times) {
        int blink_sample = static_cast<int>(blink_time * m_samplingFreq);
        int blink_width = static_cast<int>(params.blink_duration * m_samplingFreq);
        
        if (blink_sample + blink_width >= n_times) continue;
        
        Eigen::VectorXd blink = generateGaussianWindow(blink_width, blink_width/2.0, blink_width/4.0);
        blink *= params.amplitude;
        
        for (int ch : frontal_channels) {
            double weight = 1.0 - 0.1 * (ch % frontal_channels.size()) / frontal_channels.size();
            for (int t = 0; t < blink_width; ++t) {
                data(ch, blink_sample + t) += weight * blink(t);
            }
        }
    }
    
    // Generate saccades
    double saccade_interval = 60.0 / params.saccade_rate;
    std::vector<double> saccade_times;
    
    current_time = 0.0;
    while (current_time < duration) {
        saccade_times.push_back(current_time);
        double variation = m_uniformDist(m_randomGenerator) * saccade_interval * 0.3;
        current_time += saccade_interval + variation;
    }
    
    // Add saccade artifacts
    for (double saccade_time : saccade_times) {
        int saccade_sample = static_cast<int>(saccade_time * m_samplingFreq);
        int saccade_width = static_cast<int>(0.05 * m_samplingFreq); // 50ms
        
        if (saccade_sample + saccade_width >= n_times) continue;
        
        // Create biphasic saccade artifact
        Eigen::VectorXd saccade(saccade_width);
        for (int t = 0; t < saccade_width; ++t) {
            double phase = 2.0 * M_PI * t / saccade_width;
            saccade(t) = params.saccade_amplitude * std::sin(phase);
        }
        
        for (int ch : frontal_channels) {
            double weight = 0.5 + 0.5 * m_uniformDist(m_randomGenerator); // Random weight
            for (int t = 0; t < saccade_width; ++t) {
                data(ch, saccade_sample + t) += weight * saccade(t);
            }
        }
    }
    
    // Add slow drifts if requested
    if (params.add_slow_drifts) {
        for (int ch : frontal_channels) {
            for (int t = 0; t < n_times; ++t) {
                double time = t / m_samplingFreq;
                double drift = params.amplitude * 0.1 * std::sin(2.0 * M_PI * 0.01 * time); // 0.01 Hz drift
                data(ch, t) += drift;
            }
        }
    }
    
    std::cout << "Added EOG artifacts: " << blink_times.size() << " blinks, " 
              << saccade_times.size() << " saccades" << std::endl;
}

//=============================================================================================================

void NoiseSimulation::addCHPI(Eigen::MatrixXd& data, const CHPIParams& params)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    if (params.frequencies.empty()) return;
    
    // Ensure we have matching frequencies and amplitudes
    size_t n_coils = params.frequencies.size();
    std::vector<double> amplitudes = params.amplitudes;
    if (amplitudes.size() < n_coils) {
        amplitudes.resize(n_coils, amplitudes.empty() ? 1e-11 : amplitudes.back());
    }
    
    // Generate HPI signals for each coil
    for (size_t coil = 0; coil < n_coils; ++coil) {
        double frequency = params.frequencies[coil];
        double amplitude = amplitudes[coil];
        
        // Generate sinusoidal HPI signal
        Eigen::VectorXd hpi_signal = generateSinusoid(n_times, frequency, amplitude);
        
        // Add movement modulation if requested
        if (params.add_movement && coil < params.positions.size()) {
            for (int t = 0; t < n_times; ++t) {
                double time = t / m_samplingFreq;
                double movement = params.movement_amplitude * 1e-3 * // Convert mm to m
                                 std::sin(2.0 * M_PI * params.movement_frequency * time);
                hpi_signal(t) *= (1.0 + movement);
            }
        }
        
        // Apply spatial pattern (simplified - in reality this would use coil positions)
        for (int ch = 0; ch < n_channels; ++ch) {
            // Simplified spatial weighting based on channel index and coil
            double distance_factor = 1.0 / (1.0 + 0.1 * std::abs(ch - static_cast<int>(coil * n_channels / n_coils)));
            double weight = distance_factor * (0.5 + 0.5 * m_uniformDist(m_randomGenerator));
            
            for (int t = 0; t < n_times; ++t) {
                data(ch, t) += weight * hpi_signal(t);
            }
        }
    }
    
    std::cout << "Added cHPI artifacts: " << n_coils << " coils" << std::endl;
}

//=============================================================================================================

void NoiseSimulation::addGeneralNoise(Eigen::MatrixXd& data, const GeneralNoiseParams& params)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    // Add white noise
    if (params.white_noise_level > 0) {
        Eigen::MatrixXd white_noise = generateWhiteNoise(n_channels, n_times, params.white_noise_level);
        data += white_noise;
    }
    
    // Add pink noise
    if (params.pink_noise_level > 0) {
        Eigen::MatrixXd pink_noise = generatePinkNoise(n_channels, n_times, params.pink_noise_level);
        data += pink_noise;
    }
    
    // Add line noise
    if (params.line_noise_amplitude > 0) {
        Eigen::VectorXd line_noise = generateSinusoid(n_times, params.line_noise_freq, params.line_noise_amplitude);
        
        for (int ch = 0; ch < n_channels; ++ch) {
            // Add random phase per channel
            double phase = m_uniformDist(m_randomGenerator) * 2.0 * M_PI;
            Eigen::VectorXd channel_line_noise = generateSinusoid(n_times, params.line_noise_freq, 
                                                                 params.line_noise_amplitude, phase);
            data.row(ch) += channel_line_noise.transpose();
        }
    }
    
    // Add harmonics
    for (size_t h = 0; h < params.harmonic_freqs.size() && h < params.harmonic_amps.size(); ++h) {
        Eigen::VectorXd harmonic = generateSinusoid(n_times, params.harmonic_freqs[h], params.harmonic_amps[h]);
        
        for (int ch = 0; ch < n_channels; ++ch) {
            double phase = m_uniformDist(m_randomGenerator) * 2.0 * M_PI;
            Eigen::VectorXd channel_harmonic = generateSinusoid(n_times, params.harmonic_freqs[h], 
                                                               params.harmonic_amps[h], phase);
            data.row(ch) += channel_harmonic.transpose();
        }
    }
    
    // Add channel-specific noise
    if (params.add_channel_noise) {
        for (int ch = 0; ch < n_channels; ++ch) {
            double channel_noise_level = params.white_noise_level * (0.5 + m_uniformDist(m_randomGenerator));
            Eigen::VectorXd channel_noise = generateWhiteNoise(1, n_times, channel_noise_level).row(0);
            data.row(ch) += channel_noise.transpose();
        }
    }
}

//=============================================================================================================

void NoiseSimulation::addMuscleArtifacts(Eigen::MatrixXd& data, 
                                        double amplitude,
                                        const std::pair<double, double>& frequency_range,
                                        double burst_probability)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    double duration = n_times / m_samplingFreq;
    
    // Generate muscle bursts
    std::vector<std::pair<double, double>> bursts; // start_time, duration
    
    double current_time = 0.0;
    while (current_time < duration) {
        if (m_uniformDist(m_randomGenerator) < burst_probability) {
            double burst_duration = 0.1 + m_uniformDist(m_randomGenerator) * 0.4; // 0.1-0.5 seconds
            bursts.push_back({current_time, burst_duration});
        }
        current_time += 1.0; // Check every second
    }
    
    // Add muscle artifacts
    for (const auto& burst : bursts) {
        int start_sample = static_cast<int>(burst.first * m_samplingFreq);
        int burst_length = static_cast<int>(burst.second * m_samplingFreq);
        
        if (start_sample + burst_length >= n_times) continue;
        
        // Generate band-limited noise for muscle activity
        Eigen::MatrixXd muscle_noise = generateBandLimitedNoise(n_channels, burst_length,
                                                               frequency_range.first, frequency_range.second,
                                                               amplitude * amplitude);
        
        // Apply to temporal and parietal channels (simplified)
        std::vector<int> muscle_channels;
        for (int ch = n_channels/4; ch < 3*n_channels/4; ++ch) {
            muscle_channels.push_back(ch);
        }
        
        for (int ch : muscle_channels) {
            double weight = 0.5 + 0.5 * m_uniformDist(m_randomGenerator);
            for (int t = 0; t < burst_length; ++t) {
                data(ch, start_sample + t) += weight * muscle_noise(ch, t);
            }
        }
    }
    
    std::cout << "Added muscle artifacts: " << bursts.size() << " bursts" << std::endl;
}

//=============================================================================================================

void NoiseSimulation::addMovementArtifacts(Eigen::MatrixXd& data, double amplitude, double frequency)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    
    // Generate movement artifact as low-frequency oscillation
    Eigen::VectorXd movement = generateSinusoid(n_times, frequency, amplitude);
    
    // Add random phase and amplitude variations per channel
    for (int ch = 0; ch < n_channels; ++ch) {
        double phase = m_uniformDist(m_randomGenerator) * 2.0 * M_PI;
        double channel_amplitude = amplitude * (0.5 + 0.5 * m_uniformDist(m_randomGenerator));
        
        Eigen::VectorXd channel_movement = generateSinusoid(n_times, frequency, channel_amplitude, phase);
        data.row(ch) += channel_movement.transpose();
    }
}

//=============================================================================================================

void NoiseSimulation::addBadChannels(Eigen::MatrixXd& data,
                                    const std::vector<int>& bad_channels,
                                    const std::string& artifact_type)
{
    int n_times = data.cols();
    
    for (int ch : bad_channels) {
        if (ch >= data.rows()) continue;
        
        if (artifact_type == "flat") {
            // Set channel to zero
            data.row(ch).setZero();
        } else if (artifact_type == "noisy") {
            // Add high-amplitude noise
            Eigen::VectorXd noise = generateWhiteNoise(1, n_times, 1e-10).row(0);
            data.row(ch) = noise.transpose();
        } else if (artifact_type == "drift") {
            // Add linear drift
            for (int t = 0; t < n_times; ++t) {
                double drift = 1e-11 * t / n_times;
                data(ch, t) += drift;
            }
        }
    }
}

//=============================================================================================================

void NoiseSimulation::setRandomSeed(int seed)
{
    m_randomGenerator.seed(seed);
}

//=============================================================================================================

Eigen::MatrixXd NoiseSimulation::generateWhiteNoise(int n_channels, int n_times, double variance)
{
    Eigen::MatrixXd noise(n_channels, n_times);
    double std_dev = std::sqrt(variance);
    
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 0; t < n_times; ++t) {
            noise(ch, t) = std_dev * m_normalDist(m_randomGenerator);
        }
    }
    
    return noise;
}

//=============================================================================================================

Eigen::MatrixXd NoiseSimulation::generatePinkNoise(int n_channels, int n_times, double variance)
{
    // Simplified pink noise generation
    Eigen::MatrixXd white_noise = generateWhiteNoise(n_channels, n_times, variance);
    Eigen::MatrixXd pink_noise = white_noise;
    
    // Apply simple exponential smoothing to approximate 1/f characteristics
    double alpha = 0.05;
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 1; t < n_times; ++t) {
            pink_noise(ch, t) = alpha * pink_noise(ch, t) + (1.0 - alpha) * pink_noise(ch, t-1);
        }
    }
    
    return pink_noise;
}

//=============================================================================================================

Eigen::MatrixXd NoiseSimulation::generateBandLimitedNoise(int n_channels, int n_times, 
                                                         double low_freq, double high_freq, double variance)
{
    // Simplified band-limited noise (in practice, this would use proper filtering)
    Eigen::MatrixXd noise = generateWhiteNoise(n_channels, n_times, variance);
    
    // Apply simple filtering approximation
    double center_freq = (low_freq + high_freq) / 2.0;
    double modulation_freq = center_freq / m_samplingFreq * 2.0 * M_PI;
    
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 0; t < n_times; ++t) {
            double modulation = std::sin(modulation_freq * t);
            noise(ch, t) *= modulation;
        }
    }
    
    return noise;
}

//=============================================================================================================

Eigen::VectorXd NoiseSimulation::generateSinusoid(int n_times, double frequency, double amplitude, double phase)
{
    Eigen::VectorXd signal(n_times);
    
    for (int t = 0; t < n_times; ++t) {
        double time = t / m_samplingFreq;
        signal(t) = amplitude * std::sin(2.0 * M_PI * frequency * time + phase);
    }
    
    return signal;
}

//=============================================================================================================

Eigen::VectorXd NoiseSimulation::generateGaussianWindow(int n_times, double center, double width)
{
    Eigen::VectorXd window(n_times);
    
    for (int t = 0; t < n_times; ++t) {
        double x = (t - center) / width;
        window(t) = std::exp(-0.5 * x * x);
    }
    
    return window;
}

//=============================================================================================================

void NoiseSimulation::applySpatialPattern(Eigen::MatrixXd& data, const Eigen::VectorXd& artifact, 
                                         const std::vector<int>& channels, const std::vector<double>& weights)
{
    int n_times = artifact.size();
    
    for (size_t i = 0; i < channels.size() && i < weights.size(); ++i) {
        int ch = channels[i];
        double weight = weights[i];
        
        if (ch < data.rows()) {
            for (int t = 0; t < std::min(n_times, static_cast<int>(data.cols())); ++t) {
                data(ch, t) += weight * artifact(t);
            }
        }
    }
}

//=============================================================================================================

void NoiseSimulation::detectChannelTypes()
{
    // This is a simplified channel type detection
    // In practice, this would use the channel info from FiffInfo
    
    m_magChannels.clear();
    m_gradChannels.clear();
    m_eegChannels.clear();
    m_frontalChannels.clear();
    
    // Simple heuristic based on channel names or indices
    for (int ch = 0; ch < m_nChannels; ++ch) {
        // This is a placeholder - in reality, we'd check channel types from FiffInfo
        if (ch < m_nChannels / 3) {
            m_magChannels.push_back(ch);
        } else if (ch < 2 * m_nChannels / 3) {
            m_gradChannels.push_back(ch);
        } else {
            m_eegChannels.push_back(ch);
        }
        
        // Frontal channels (first 20% of channels as approximation)
        if (ch < m_nChannels / 5) {
            m_frontalChannels.push_back(ch);
        }
    }
}

//=============================================================================================================
// Convenience functions
//=============================================================================================================

void add_noise(Eigen::MatrixXd& data,
              const FIFFLIB::FiffInfo::SPtr& info,
              double noise_level,
              int random_seed)
{
    NoiseSimulation noise_sim(info, random_seed);
    noise_sim.addNoise(data, noise_level);
}

//=============================================================================================================

void add_ecg(Eigen::MatrixXd& data,
            const FIFFLIB::FiffInfo::SPtr& info,
            const NoiseSimulation::ECGParams& params)
{
    NoiseSimulation noise_sim(info);
    noise_sim.addECG(data, params);
}

//=============================================================================================================

void add_eog(Eigen::MatrixXd& data,
            const FIFFLIB::FiffInfo::SPtr& info,
            const NoiseSimulation::EOGParams& params)
{
    NoiseSimulation noise_sim(info);
    noise_sim.addEOG(data, params);
}

//=============================================================================================================

void add_chpi(Eigen::MatrixXd& data,
             const FIFFLIB::FiffInfo::SPtr& info,
             const NoiseSimulation::CHPIParams& params)
{
    NoiseSimulation noise_sim(info);
    noise_sim.addCHPI(data, params);
}

} // namespace SIMULATIONLIB