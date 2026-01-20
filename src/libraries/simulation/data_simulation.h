#ifndef DATA_SIMULATION_H
#define DATA_SIMULATION_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "simulation_global.h"
#include <fiff/fiff_info.h>
#include <fiff/fiff_evoked.h>
#include <fiff/fiff_raw_data.h>
#include <mne/mne_forwardsolution.h>
#include <mne/mne_sourceestimate.h>
#include <Eigen/Core>
#include <QSharedPointer>
#include <random>

//=============================================================================================================
// DEFINE NAMESPACE SIMULATIONLIB
//=============================================================================================================

namespace SIMULATIONLIB
{

//=============================================================================================================
// FORWARD DECLARATIONS
//=============================================================================================================

//=============================================================================================================
/**
 * Data simulation class for MEG/EEG signal generation
 * 
 * Provides functionality to simulate realistic MEG/EEG data with various noise models
 * and signal characteristics for testing and validation purposes.
 */
class SIMULATIONSHARED_EXPORT DataSimulation
{
public:
    typedef QSharedPointer<DataSimulation> SPtr;            /**< Shared pointer type for DataSimulation. */
    typedef QSharedPointer<const DataSimulation> ConstSPtr; /**< Const shared pointer type for DataSimulation. */

    /**
     * @brief Noise model parameters
     */
    struct NoiseModel {
        double white_noise_cov;     /**< White noise covariance (default: 1e-12) */
        double pink_noise_cov;      /**< Pink noise covariance (default: 1e-13) */
        double line_noise_freq;     /**< Line noise frequency in Hz (default: 50.0) */
        double line_noise_amplitude; /**< Line noise amplitude (default: 1e-12) */
        bool add_ecg;               /**< Add ECG artifacts (default: false) */
        bool add_eog;               /**< Add EOG artifacts (default: false) */
        double ecg_amplitude;       /**< ECG artifact amplitude (default: 5e-12) */
        double eog_amplitude;       /**< EOG artifact amplitude (default: 10e-12) */
        int random_seed;            /**< Random seed for reproducibility (default: -1, random) */
    };

    /**
     * @brief Signal parameters for simulation
     */
    struct SignalParams {
        double duration;            /**< Signal duration in seconds (default: 1.0) */
        double sampling_freq;       /**< Sampling frequency in Hz (default: 1000.0) */
        double signal_amplitude;    /**< Signal amplitude scaling (default: 1e-9) */
        std::vector<double> frequencies; /**< Signal frequencies for oscillatory components */
        std::vector<double> phases; /**< Phase offsets for oscillatory components */
        bool add_baseline;          /**< Add baseline offset (default: false) */
        double baseline_amplitude;  /**< Baseline amplitude (default: 0.0) */
    };

    //=========================================================================================================
    /**
     * Constructs a DataSimulation object.
     *
     * @param[in] info          MEG/EEG measurement info.
     * @param[in] forward       Forward solution for source simulation.
     * @param[in] noise_model   Noise model parameters.
     * @param[in] signal_params Signal parameters.
     */
    explicit DataSimulation(const FIFFLIB::FiffInfo::SPtr& info,
                           const MNELIB::MNEForwardSolution::SPtr& forward = nullptr,
                           const NoiseModel& noise_model = NoiseModel(),
                           const SignalParams& signal_params = SignalParams());

    //=========================================================================================================
    /**
     * Destroys the DataSimulation object.
     */
    ~DataSimulation() = default;

    //=========================================================================================================
    /**
     * Simulate evoked response data.
     * 
     * @param[in] source_activity   Source activity pattern (n_sources x n_times).
     * @param[in] n_epochs         Number of epochs to simulate (default: 1).
     * @return Simulated evoked data.
     */
    FIFFLIB::FiffEvoked::SPtr simulateEvoked(const Eigen::MatrixXd& source_activity,
                                            int n_epochs = 1);

    //=========================================================================================================
    /**
     * Simulate evoked response with dipole sources.
     * 
     * @param[in] dipole_positions  Dipole positions (n_dipoles x 3).
     * @param[in] dipole_moments    Dipole moments (n_dipoles x 3 x n_times).
     * @param[in] n_epochs         Number of epochs to simulate (default: 1).
     * @return Simulated evoked data.
     */
    FIFFLIB::FiffEvoked::SPtr simulateEvokedDipoles(const Eigen::MatrixXd& dipole_positions,
                                                   const Eigen::Array3Xd& dipole_moments,
                                                   int n_epochs = 1);

    //=========================================================================================================
    /**
     * Simulate raw continuous data.
     * 
     * @param[in] source_activity   Source activity pattern (n_sources x n_times).
     * @return Simulated raw data.
     */
    FIFFLIB::FiffRawData::SPtr simulateRaw(const Eigen::MatrixXd& source_activity);

    //=========================================================================================================
    /**
     * Simulate raw data with time-varying source activity.
     * 
     * @param[in] duration         Duration in seconds.
     * @param[in] source_generator Function to generate source activity at each time point.
     * @return Simulated raw data.
     */
    FIFFLIB::FiffRawData::SPtr simulateRawTimeVarying(double duration,
                                                     std::function<Eigen::VectorXd(double)> source_generator);

    //=========================================================================================================
    /**
     * Generate oscillatory source activity.
     * 
     * @param[in] n_sources        Number of sources.
     * @param[in] n_times          Number of time points.
     * @param[in] frequencies      Oscillation frequencies for each source.
     * @param[in] amplitudes       Amplitudes for each source.
     * @param[in] phases           Phase offsets for each source.
     * @return Generated source activity matrix.
     */
    Eigen::MatrixXd generateOscillatoryActivity(int n_sources,
                                              int n_times,
                                              const std::vector<double>& frequencies,
                                              const std::vector<double>& amplitudes,
                                              const std::vector<double>& phases = {});

    //=========================================================================================================
    /**
     * Generate transient source activity (e.g., ERPs).
     * 
     * @param[in] n_sources        Number of sources.
     * @param[in] n_times          Number of time points.
     * @param[in] peak_times       Peak times for each source (in samples).
     * @param[in] peak_widths      Peak widths for each source (in samples).
     * @param[in] amplitudes       Peak amplitudes for each source.
     * @return Generated source activity matrix.
     */
    Eigen::MatrixXd generateTransientActivity(int n_sources,
                                             int n_times,
                                             const std::vector<int>& peak_times,
                                             const std::vector<double>& peak_widths,
                                             const std::vector<double>& amplitudes);

    //=========================================================================================================
    /**
     * Generate realistic noise based on the noise model.
     * 
     * @param[in] n_channels       Number of channels.
     * @param[in] n_times          Number of time points.
     * @return Generated noise matrix.
     */
    Eigen::MatrixXd generateNoise(int n_channels, int n_times);

    //=========================================================================================================
    /**
     * Add ECG artifacts to the data.
     * 
     * @param[in,out] data         Data matrix to add artifacts to.
     * @param[in] heart_rate       Heart rate in BPM (default: 72).
     */
    void addECGArtifacts(Eigen::MatrixXd& data, double heart_rate = 72.0);

    //=========================================================================================================
    /**
     * Add EOG artifacts to the data.
     * 
     * @param[in,out] data         Data matrix to add artifacts to.
     * @param[in] blink_rate       Blink rate per minute (default: 15).
     */
    void addEOGArtifacts(Eigen::MatrixXd& data, double blink_rate = 15.0);

    //=========================================================================================================
    /**
     * Add line noise to the data.
     * 
     * @param[in,out] data         Data matrix to add noise to.
     */
    void addLineNoise(Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Get/Set noise model parameters.
     */
    NoiseModel getNoiseModel() const { return m_noiseModel; }
    void setNoiseModel(const NoiseModel& noise_model) { m_noiseModel = noise_model; }

    //=========================================================================================================
    /**
     * Get/Set signal parameters.
     */
    SignalParams getSignalParams() const { return m_signalParams; }
    void setSignalParams(const SignalParams& signal_params) { m_signalParams = signal_params; }

    //=========================================================================================================
    /**
     * Set random seed for reproducible simulations.
     */
    void setRandomSeed(int seed);

private:
    //=========================================================================================================
    /**
     * Initialize default noise model.
     */
    static NoiseModel getDefaultNoiseModel();

    //=========================================================================================================
    /**
     * Initialize default signal parameters.
     */
    static SignalParams getDefaultSignalParams();

    //=========================================================================================================
    /**
     * Generate white noise.
     */
    Eigen::MatrixXd generateWhiteNoise(int n_channels, int n_times, double variance);

    //=========================================================================================================
    /**
     * Generate pink (1/f) noise.
     */
    Eigen::MatrixXd generatePinkNoise(int n_channels, int n_times, double variance);

    //=========================================================================================================
    /**
     * Apply forward solution to source activity.
     */
    Eigen::MatrixXd applyForwardSolution(const Eigen::MatrixXd& source_activity);

    //=========================================================================================================
    /**
     * Create time vector for simulation.
     */
    Eigen::VectorXd createTimeVector(int n_times);

    //=========================================================================================================
    /**
     * Generate Gaussian window function.
     */
    Eigen::VectorXd generateGaussianWindow(int n_times, double center, double width);

    // Member variables
    FIFFLIB::FiffInfo::SPtr m_pInfo;                    // Measurement info
    MNELIB::MNEForwardSolution::SPtr m_pForward;       // Forward solution
    NoiseModel m_noiseModel;                           // Noise model parameters
    SignalParams m_signalParams;                       // Signal parameters
    
    // Random number generation
    std::mt19937 m_randomGenerator;                    // Random number generator
    std::normal_distribution<double> m_normalDist;     // Normal distribution
    std::uniform_real_distribution<double> m_uniformDist; // Uniform distribution
    
    // Cached values
    int m_nChannels;                                   // Number of channels
    int m_nSources;                                    // Number of sources
    double m_samplingFreq;                             // Sampling frequency
};

//=============================================================================================================
/**
 * Convenience function to simulate evoked response.
 * 
 * @param[in] info              Measurement info.
 * @param[in] forward           Forward solution.
 * @param[in] source_activity   Source activity pattern.
 * @param[in] noise_model       Noise model parameters.
 * @param[in] signal_params     Signal parameters.
 * @param[in] n_epochs         Number of epochs.
 * @return Simulated evoked data.
 */
SIMULATIONSHARED_EXPORT FIFFLIB::FiffEvoked::SPtr simulate_evoked(
    const FIFFLIB::FiffInfo::SPtr& info,
    const MNELIB::MNEForwardSolution::SPtr& forward,
    const Eigen::MatrixXd& source_activity,
    const DataSimulation::NoiseModel& noise_model = DataSimulation::NoiseModel(),
    const DataSimulation::SignalParams& signal_params = DataSimulation::SignalParams(),
    int n_epochs = 1);

//=============================================================================================================
/**
 * Convenience function to simulate raw data.
 * 
 * @param[in] info              Measurement info.
 * @param[in] forward           Forward solution.
 * @param[in] source_activity   Source activity pattern.
 * @param[in] noise_model       Noise model parameters.
 * @param[in] signal_params     Signal parameters.
 * @return Simulated raw data.
 */
SIMULATIONSHARED_EXPORT FIFFLIB::FiffRawData::SPtr simulate_raw(
    const FIFFLIB::FiffInfo::SPtr& info,
    const MNELIB::MNEForwardSolution::SPtr& forward,
    const Eigen::MatrixXd& source_activity,
    const DataSimulation::NoiseModel& noise_model = DataSimulation::NoiseModel(),
    const DataSimulation::SignalParams& signal_params = DataSimulation::SignalParams());

} // namespace SIMULATIONLIB

#endif // DATA_SIMULATION_H