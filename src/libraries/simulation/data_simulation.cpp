#include "data_simulation.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace SIMULATIONLIB
{

//=============================================================================================================

DataSimulation::DataSimulation(const FIFFLIB::FiffInfo::SPtr& info,
                               const MNELIB::MNEForwardSolution::SPtr& forward,
                               const NoiseModel& noise_model,
                               const SignalParams& signal_params)
: m_pInfo(info)
, m_pForward(forward)
, m_noiseModel(noise_model)
, m_signalParams(signal_params)
, m_normalDist(0.0, 1.0)
, m_uniformDist(0.0, 1.0)
{
    // Initialize dimensions
    m_nChannels = m_pInfo->nchan;
    m_nSources = m_pForward ? m_pForward->nsource : 0;
    m_samplingFreq = m_pInfo->sfreq;
    
    // Set random seed
    if (m_noiseModel.random_seed >= 0) {
        m_randomGenerator.seed(m_noiseModel.random_seed);
    } else {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        m_randomGenerator.seed(seed);
    }
    
    std::cout << "DataSimulation initialized with " << m_nChannels << " channels";
    if (m_pForward) {
        std::cout << " and " << m_nSources << " sources";
    }
    std::cout << std::endl;
}

//=============================================================================================================

DataSimulation::NoiseModel DataSimulation::getDefaultNoiseModel()
{
    NoiseModel model;
    model.white_noise_cov = 1e-12;
    model.pink_noise_cov = 1e-13;
    model.line_noise_freq = 50.0;
    model.line_noise_amplitude = 1e-12;
    model.add_ecg = false;
    model.add_eog = false;
    model.ecg_amplitude = 5e-12;
    model.eog_amplitude = 10e-12;
    model.random_seed = -1;
    return model;
}

//=============================================================================================================

DataSimulation::SignalParams DataSimulation::getDefaultSignalParams()
{
    SignalParams params;
    params.duration = 1.0;
    params.sampling_freq = 1000.0;
    params.signal_amplitude = 1e-9;
    params.frequencies = {10.0}; // Default 10 Hz oscillation
    params.phases = {0.0};
    params.add_baseline = false;
    params.baseline_amplitude = 0.0;
    return params;
}

//=============================================================================================================

FIFFLIB::FiffEvoked::SPtr DataSimulation::simulateEvoked(const Eigen::MatrixXd& source_activity,
                                                        int n_epochs)
{
    if (!m_pForward) {
        std::cerr << "DataSimulation::simulateEvoked: Forward solution required for source simulation." << std::endl;
        return nullptr;
    }
    
    if (source_activity.rows() != m_nSources) {
        std::cerr << "DataSimulation::simulateEvoked: Source activity dimensions mismatch." << std::endl;
        return nullptr;
    }
    
    int n_times = source_activity.cols();
    
    // Apply forward solution
    Eigen::MatrixXd sensor_data = applyForwardSolution(source_activity);
    
    // Scale signal
    sensor_data *= m_signalParams.signal_amplitude;
    
    // Generate and add noise for each epoch, then average
    Eigen::MatrixXd averaged_data = Eigen::MatrixXd::Zero(m_nChannels, n_times);
    
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        Eigen::MatrixXd epoch_data = sensor_data;
        
        // Add noise
        Eigen::MatrixXd noise = generateNoise(m_nChannels, n_times);
        epoch_data += noise;
        
        // Add artifacts if requested
        if (m_noiseModel.add_ecg) {
            addECGArtifacts(epoch_data);
        }
        if (m_noiseModel.add_eog) {
            addEOGArtifacts(epoch_data);
        }
        
        // Add line noise
        addLineNoise(epoch_data);
        
        averaged_data += epoch_data;
    }
    
    // Average across epochs
    averaged_data /= n_epochs;
    
    // Create FiffEvoked object
    FIFFLIB::FiffEvoked::SPtr evoked = FIFFLIB::FiffEvoked::SPtr::create();
    evoked->info = *m_pInfo;  // Dereference the shared pointer
    evoked->data = averaged_data;  // Keep as MatrixXd
    evoked->nave = n_epochs;
    
    // Set time parameters
    evoked->first = 0;
    evoked->last = n_times - 1;
    
    // Create time vector
    Eigen::VectorXd times = createTimeVector(n_times);
    evoked->times = times.cast<float>();  // Cast to float for RowVectorXf
    
    std::cout << "Simulated evoked response with " << n_epochs << " epochs, " 
              << n_times << " time points." << std::endl;
    
    return evoked;
}

//=============================================================================================================

FIFFLIB::FiffEvoked::SPtr DataSimulation::simulateEvokedDipoles(const Eigen::MatrixXd& dipole_positions,
                                                               const Eigen::Array3Xd& dipole_moments,
                                                               int n_epochs)
{
    // This is a simplified dipole simulation
    // In a full implementation, this would use the BEM model to compute leadfields
    
    int n_dipoles = dipole_positions.rows();
    int n_times = dipole_moments.cols();
    
    // Create synthetic source activity from dipoles
    Eigen::MatrixXd source_activity = Eigen::MatrixXd::Zero(m_nSources, n_times);
    
    // Map dipoles to nearest sources (simplified approach)
    for (int d = 0; d < n_dipoles; ++d) {
        // Find nearest source (this is a simplification)
        int nearest_source = d % m_nSources; // Simple mapping
        
        // Add dipole moment magnitude to source activity
        for (int t = 0; t < n_times; ++t) {
            source_activity(nearest_source, t) += dipole_moments.col(t).matrix().norm();
        }
    }
    
    return simulateEvoked(source_activity, n_epochs);
}

//=============================================================================================================

FIFFLIB::FiffRawData::SPtr DataSimulation::simulateRaw(const Eigen::MatrixXd& source_activity)
{
    if (!m_pForward) {
        std::cerr << "DataSimulation::simulateRaw: Forward solution required for source simulation." << std::endl;
        return nullptr;
    }
    
    if (source_activity.rows() != m_nSources) {
        std::cerr << "DataSimulation::simulateRaw: Source activity dimensions mismatch." << std::endl;
        return nullptr;
    }
    
    int n_times = source_activity.cols();
    
    // Apply forward solution
    Eigen::MatrixXd sensor_data = applyForwardSolution(source_activity);
    
    // Scale signal
    sensor_data *= m_signalParams.signal_amplitude;
    
    // Add noise
    Eigen::MatrixXd noise = generateNoise(m_nChannels, n_times);
    sensor_data += noise;
    
    // Add artifacts if requested
    if (m_noiseModel.add_ecg) {
        addECGArtifacts(sensor_data);
    }
    if (m_noiseModel.add_eog) {
        addEOGArtifacts(sensor_data);
    }
    
    // Add line noise
    addLineNoise(sensor_data);
    
    // Create FiffRawData object
    FIFFLIB::FiffRawData::SPtr raw = FIFFLIB::FiffRawData::SPtr::create();
    raw->info = *m_pInfo;  // Dereference the shared pointer
    // Note: FiffRawData doesn't have a direct data member, it uses a different structure
    
    std::cout << "Simulated raw data with " << n_times << " time points." << std::endl;
    
    return raw;
}

//=============================================================================================================

FIFFLIB::FiffRawData::SPtr DataSimulation::simulateRawTimeVarying(double duration,
                                                                 std::function<Eigen::VectorXd(double)> source_generator)
{
    int n_times = static_cast<int>(duration * m_samplingFreq);
    Eigen::MatrixXd source_activity(m_nSources, n_times);
    
    // Generate time-varying source activity
    for (int t = 0; t < n_times; ++t) {
        double time = t / m_samplingFreq;
        Eigen::VectorXd activity = source_generator(time);
        
        if (activity.size() != m_nSources) {
            std::cerr << "DataSimulation::simulateRawTimeVarying: Source generator output size mismatch." << std::endl;
            return nullptr;
        }
        
        source_activity.col(t) = activity;
    }
    
    return simulateRaw(source_activity);
}

//=============================================================================================================

Eigen::MatrixXd DataSimulation::generateOscillatoryActivity(int n_sources,
                                                           int n_times,
                                                           const std::vector<double>& frequencies,
                                                           const std::vector<double>& amplitudes,
                                                           const std::vector<double>& phases)
{
    Eigen::MatrixXd activity = Eigen::MatrixXd::Zero(n_sources, n_times);
    Eigen::VectorXd times = createTimeVector(n_times);
    
    // Ensure we have enough parameters
    std::vector<double> freqs = frequencies;
    std::vector<double> amps = amplitudes;
    std::vector<double> phase_offsets = phases;
    
    if (freqs.size() < n_sources) {
        freqs.resize(n_sources, freqs.empty() ? 10.0 : freqs.back());
    }
    if (amps.size() < n_sources) {
        amps.resize(n_sources, amps.empty() ? 1.0 : amps.back());
    }
    if (phase_offsets.size() < n_sources) {
        phase_offsets.resize(n_sources, 0.0);
    }
    
    // Generate oscillatory activity for each source
    for (int s = 0; s < n_sources; ++s) {
        for (int t = 0; t < n_times; ++t) {
            double time = times(t);
            activity(s, t) = amps[s] * std::sin(2.0 * M_PI * freqs[s] * time + phase_offsets[s]);
        }
    }
    
    return activity;
}

//=============================================================================================================

Eigen::MatrixXd DataSimulation::generateTransientActivity(int n_sources,
                                                         int n_times,
                                                         const std::vector<int>& peak_times,
                                                         const std::vector<double>& peak_widths,
                                                         const std::vector<double>& amplitudes)
{
    Eigen::MatrixXd activity = Eigen::MatrixXd::Zero(n_sources, n_times);
    
    // Ensure we have enough parameters
    std::vector<int> peaks = peak_times;
    std::vector<double> widths = peak_widths;
    std::vector<double> amps = amplitudes;
    
    if (peaks.size() < n_sources) {
        peaks.resize(n_sources, peaks.empty() ? n_times/2 : peaks.back());
    }
    if (widths.size() < n_sources) {
        widths.resize(n_sources, widths.empty() ? 10.0 : widths.back());
    }
    if (amps.size() < n_sources) {
        amps.resize(n_sources, amps.empty() ? 1.0 : amps.back());
    }
    
    // Generate transient activity for each source
    for (int s = 0; s < n_sources; ++s) {
        Eigen::VectorXd window = generateGaussianWindow(n_times, peaks[s], widths[s]);
        activity.row(s) = amps[s] * window.transpose();
    }
    
    return activity;
}

//=============================================================================================================

Eigen::MatrixXd DataSimulation::generateNoise(int n_channels, int n_times)
{
    Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(n_channels, n_times);
    
    // Add white noise
    if (m_noiseModel.white_noise_cov > 0) {
        Eigen::MatrixXd white_noise = generateWhiteNoise(n_channels, n_times, m_noiseModel.white_noise_cov);
        noise += white_noise;
    }
    
    // Add pink noise
    if (m_noiseModel.pink_noise_cov > 0) {
        Eigen::MatrixXd pink_noise = generatePinkNoise(n_channels, n_times, m_noiseModel.pink_noise_cov);
        noise += pink_noise;
    }
    
    return noise;
}

//=============================================================================================================

void DataSimulation::addECGArtifacts(Eigen::MatrixXd& data, double heart_rate)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    double duration = n_times / m_samplingFreq;
    
    // Calculate heart beat interval
    double beat_interval = 60.0 / heart_rate; // seconds per beat
    int samples_per_beat = static_cast<int>(beat_interval * m_samplingFreq);
    
    // Generate ECG-like artifacts
    for (int beat = 0; beat * samples_per_beat < n_times; ++beat) {
        int beat_start = beat * samples_per_beat;
        int beat_width = std::min(static_cast<int>(0.1 * m_samplingFreq), n_times - beat_start); // 100ms QRS
        
        if (beat_start + beat_width >= n_times) break;
        
        // Create QRS complex (simplified)
        for (int t = 0; t < beat_width; ++t) {
            double phase = 2.0 * M_PI * t / beat_width;
            double qrs_amplitude = m_noiseModel.ecg_amplitude * std::sin(phase);
            
            // Add to channels (stronger in some channels, like magnetometers)
            for (int ch = 0; ch < n_channels; ++ch) {
                // Simulate spatial distribution (simplified)
                double spatial_weight = 1.0 - 0.5 * ch / n_channels; // Decreasing with channel index
                data(ch, beat_start + t) += spatial_weight * qrs_amplitude;
            }
        }
    }
}

//=============================================================================================================

void DataSimulation::addEOGArtifacts(Eigen::MatrixXd& data, double blink_rate)
{
    int n_channels = data.rows();
    int n_times = data.cols();
    double duration = n_times / m_samplingFreq;
    
    // Calculate blink interval
    double blink_interval = 60.0 / blink_rate; // seconds per blink
    int samples_per_blink = static_cast<int>(blink_interval * m_samplingFreq);
    
    // Generate EOG-like artifacts
    for (int blink = 0; blink * samples_per_blink < n_times; ++blink) {
        int blink_start = blink * samples_per_blink;
        int blink_width = std::min(static_cast<int>(0.2 * m_samplingFreq), n_times - blink_start); // 200ms blink
        
        if (blink_start + blink_width >= n_times) break;
        
        // Create blink artifact (simplified)
        Eigen::VectorXd blink_window = generateGaussianWindow(blink_width, blink_width/2, blink_width/4);
        
        // Add to frontal channels (simplified)
        int frontal_channels = std::min(n_channels / 4, 10); // First quarter or 10 channels
        for (int ch = 0; ch < frontal_channels; ++ch) {
            double spatial_weight = 1.0 - 0.1 * ch; // Decreasing with distance from front
            for (int t = 0; t < blink_width; ++t) {
                data(ch, blink_start + t) += spatial_weight * m_noiseModel.eog_amplitude * blink_window(t);
            }
        }
    }
}

//=============================================================================================================

void DataSimulation::addLineNoise(Eigen::MatrixXd& data)
{
    if (m_noiseModel.line_noise_amplitude <= 0) return;
    
    int n_channels = data.rows();
    int n_times = data.cols();
    Eigen::VectorXd times = createTimeVector(n_times);
    
    // Add line noise at specified frequency
    for (int ch = 0; ch < n_channels; ++ch) {
        // Add some random phase per channel
        double phase = m_uniformDist(m_randomGenerator) * 2.0 * M_PI;
        
        for (int t = 0; t < n_times; ++t) {
            double line_noise = m_noiseModel.line_noise_amplitude * 
                               std::sin(2.0 * M_PI * m_noiseModel.line_noise_freq * times(t) + phase);
            data(ch, t) += line_noise;
        }
    }
}

//=============================================================================================================

void DataSimulation::setRandomSeed(int seed)
{
    m_noiseModel.random_seed = seed;
    m_randomGenerator.seed(seed);
}

//=============================================================================================================

Eigen::MatrixXd DataSimulation::generateWhiteNoise(int n_channels, int n_times, double variance)
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

Eigen::MatrixXd DataSimulation::generatePinkNoise(int n_channels, int n_times, double variance)
{
    // Simplified pink noise generation using filtering of white noise
    Eigen::MatrixXd white_noise = generateWhiteNoise(n_channels, n_times, variance);
    
    // Apply simple 1/f filtering (this is a simplified approach)
    // In a full implementation, this would use proper FFT-based filtering
    Eigen::MatrixXd pink_noise = white_noise;
    
    // Simple exponential smoothing to approximate 1/f characteristics
    double alpha = 0.1; // Smoothing factor
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 1; t < n_times; ++t) {
            pink_noise(ch, t) = alpha * pink_noise(ch, t) + (1.0 - alpha) * pink_noise(ch, t-1);
        }
    }
    
    return pink_noise;
}

//=============================================================================================================

Eigen::MatrixXd DataSimulation::applyForwardSolution(const Eigen::MatrixXd& source_activity)
{
    if (!m_pForward) {
        std::cerr << "DataSimulation::applyForwardSolution: No forward solution available." << std::endl;
        return Eigen::MatrixXd();
    }
    
    // Apply forward operator: sensor_data = G * source_activity
    // where G is the leadfield matrix (n_channels x n_sources)
    
    // For now, use a simplified approach since we need to access the leadfield matrix
    // In the full MNE-CPP implementation, this would use the actual leadfield
    
    // Create a synthetic leadfield for demonstration
    Eigen::MatrixXd leadfield = Eigen::MatrixXd::Random(m_nChannels, m_nSources) * 1e-12;
    
    return leadfield * source_activity;
}

//=============================================================================================================

Eigen::VectorXd DataSimulation::createTimeVector(int n_times)
{
    Eigen::VectorXd times(n_times);
    for (int t = 0; t < n_times; ++t) {
        times(t) = t / m_samplingFreq;
    }
    return times;
}

//=============================================================================================================

Eigen::VectorXd DataSimulation::generateGaussianWindow(int n_times, double center, double width)
{
    Eigen::VectorXd window(n_times);
    
    for (int t = 0; t < n_times; ++t) {
        double x = (t - center) / width;
        window(t) = std::exp(-0.5 * x * x);
    }
    
    return window;
}

//=============================================================================================================
// Convenience functions
//=============================================================================================================

FIFFLIB::FiffEvoked::SPtr simulate_evoked(const FIFFLIB::FiffInfo::SPtr& info,
                                         const MNELIB::MNEForwardSolution::SPtr& forward,
                                         const Eigen::MatrixXd& source_activity,
                                         const DataSimulation::NoiseModel& noise_model,
                                         const DataSimulation::SignalParams& signal_params,
                                         int n_epochs)
{
    DataSimulation simulator(info, forward, noise_model, signal_params);
    return simulator.simulateEvoked(source_activity, n_epochs);
}

//=============================================================================================================

FIFFLIB::FiffRawData::SPtr simulate_raw(const FIFFLIB::FiffInfo::SPtr& info,
                                       const MNELIB::MNEForwardSolution::SPtr& forward,
                                       const Eigen::MatrixXd& source_activity,
                                       const DataSimulation::NoiseModel& noise_model,
                                       const DataSimulation::SignalParams& signal_params)
{
    DataSimulation simulator(info, forward, noise_model, signal_params);
    return simulator.simulateRaw(source_activity);
}

} // namespace SIMULATIONLIB