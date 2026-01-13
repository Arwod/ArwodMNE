#ifndef NOISE_SIMULATION_H
#define NOISE_SIMULATION_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "simulation_global.h"
#include <fiff/fiff_info.h>
#include <Eigen/Core>
#include <QSharedPointer>
#include <random>

//=============================================================================================================
// DEFINE NAMESPACE SIMULATIONLIB
//=============================================================================================================

namespace SIMULATIONLIB
{

//=============================================================================================================
/**
 * Noise and artifact simulation class
 * 
 * Provides functionality to add various types of noise and artifacts to MEG/EEG data
 * for testing preprocessing algorithms and validation purposes.
 */
class SIMULATIONSHARED_EXPORT NoiseSimulation
{
public:
    typedef QSharedPointer<NoiseSimulation> SPtr;            /**< Shared pointer type for NoiseSimulation. */
    typedef QSharedPointer<const NoiseSimulation> ConstSPtr; /**< Const shared pointer type for NoiseSimulation. */

    /**
     * @brief ECG artifact parameters
     */
    struct ECGParams {
        double heart_rate;          /**< Heart rate in BPM (default: 72) */
        double amplitude;           /**< ECG amplitude (default: 5e-12) */
        double qrs_width;           /**< QRS complex width in seconds (default: 0.1) */
        double p_wave_amplitude;    /**< P wave amplitude ratio (default: 0.2) */
        double t_wave_amplitude;    /**< T wave amplitude ratio (default: 0.3) */
        std::vector<int> affected_channels; /**< Channels affected by ECG (empty = all) */
        bool add_variability;       /**< Add heart rate variability (default: true) */
        double variability_std;     /**< HRV standard deviation in BPM (default: 5.0) */
    };

    /**
     * @brief EOG artifact parameters
     */
    struct EOGParams {
        double blink_rate;          /**< Blink rate per minute (default: 15) */
        double amplitude;           /**< EOG amplitude (default: 10e-12) */
        double blink_duration;      /**< Blink duration in seconds (default: 0.2) */
        double saccade_rate;        /**< Saccade rate per minute (default: 180) */
        double saccade_amplitude;   /**< Saccade amplitude (default: 2e-12) */
        std::vector<int> frontal_channels; /**< Frontal channels for EOG (empty = auto-detect) */
        bool add_slow_drifts;       /**< Add slow eye movement drifts (default: true) */
    };

    /**
     * @brief cHPI (continuous head position indicator) parameters
     */
    struct CHPIParams {
        std::vector<double> frequencies; /**< HPI coil frequencies in Hz */
        std::vector<double> amplitudes;  /**< HPI coil amplitudes */
        std::vector<Eigen::Vector3d> positions; /**< HPI coil positions */
        bool add_movement;          /**< Add head movement (default: false) */
        double movement_amplitude;  /**< Movement amplitude in mm (default: 1.0) */
        double movement_frequency;  /**< Movement frequency in Hz (default: 0.1) */
    };

    /**
     * @brief General noise parameters
     */
    struct GeneralNoiseParams {
        double white_noise_level;   /**< White noise level (default: 1e-12) */
        double pink_noise_level;    /**< Pink noise level (default: 5e-13) */
        double line_noise_freq;     /**< Line noise frequency in Hz (default: 50.0) */
        double line_noise_amplitude; /**< Line noise amplitude (default: 1e-12) */
        std::vector<double> harmonic_freqs; /**< Harmonic frequencies */
        std::vector<double> harmonic_amps;  /**< Harmonic amplitudes */
        bool add_channel_noise;     /**< Add channel-specific noise (default: true) */
        int random_seed;            /**< Random seed (default: -1, random) */
    };

    //=========================================================================================================
    /**
     * Constructs a NoiseSimulation object.
     *
     * @param[in] info          MEG/EEG measurement info.
     * @param[in] random_seed   Random seed for reproducibility.
     */
    explicit NoiseSimulation(const FIFFLIB::FiffInfo::SPtr& info, int random_seed = -1);

    //=========================================================================================================
    /**
     * Destroys the NoiseSimulation object.
     */
    ~NoiseSimulation() = default;

    //=========================================================================================================
    /**
     * Add white noise to data.
     * 
     * @param[in,out] data      Data matrix to add noise to.
     * @param[in] noise_level   Noise level (variance).
     */
    void addNoise(Eigen::MatrixXd& data, double noise_level);

    //=========================================================================================================
    /**
     * Add ECG artifacts to data.
     * 
     * @param[in,out] data      Data matrix to add artifacts to.
     * @param[in] params        ECG parameters.
     */
    void addECG(Eigen::MatrixXd& data, const ECGParams& params = ECGParams());

    //=========================================================================================================
    /**
     * Add EOG artifacts to data.
     * 
     * @param[in,out] data      Data matrix to add artifacts to.
     * @param[in] params        EOG parameters.
     */
    void addEOG(Eigen::MatrixXd& data, const EOGParams& params = EOGParams());

    //=========================================================================================================
    /**
     * Add cHPI artifacts to data.
     * 
     * @param[in,out] data      Data matrix to add artifacts to.
     * @param[in] params        cHPI parameters.
     */
    void addCHPI(Eigen::MatrixXd& data, const CHPIParams& params = CHPIParams());

    //=========================================================================================================
    /**
     * Add comprehensive noise model to data.
     * 
     * @param[in,out] data      Data matrix to add noise to.
     * @param[in] params        General noise parameters.
     */
    void addGeneralNoise(Eigen::MatrixXd& data, const GeneralNoiseParams& params = GeneralNoiseParams());

    //=========================================================================================================
    /**
     * Add muscle artifacts (EMG).
     * 
     * @param[in,out] data      Data matrix to add artifacts to.
     * @param[in] amplitude     Muscle artifact amplitude.
     * @param[in] frequency_range Frequency range for muscle activity [low, high] Hz.
     * @param[in] burst_probability Probability of muscle bursts per second.
     */
    void addMuscleArtifacts(Eigen::MatrixXd& data, 
                           double amplitude = 20e-12,
                           const std::pair<double, double>& frequency_range = {20.0, 200.0},
                           double burst_probability = 0.1);

    //=========================================================================================================
    /**
     * Add movement artifacts.
     * 
     * @param[in,out] data      Data matrix to add artifacts to.
     * @param[in] amplitude     Movement amplitude.
     * @param[in] frequency     Movement frequency in Hz.
     */
    void addMovementArtifacts(Eigen::MatrixXd& data, 
                             double amplitude = 50e-12,
                             double frequency = 0.5);

    //=========================================================================================================
    /**
     * Add channel-specific bad channel simulation.
     * 
     * @param[in,out] data      Data matrix to modify.
     * @param[in] bad_channels  List of bad channel indices.
     * @param[in] artifact_type Type of artifact: "flat", "noisy", "drift".
     */
    void addBadChannels(Eigen::MatrixXd& data,
                       const std::vector<int>& bad_channels,
                       const std::string& artifact_type = "noisy");

    //=========================================================================================================
    /**
     * Get default parameters for different artifact types.
     */
    static ECGParams getDefaultECGParams();
    static EOGParams getDefaultEOGParams();
    static CHPIParams getDefaultCHPIParams();
    static GeneralNoiseParams getDefaultGeneralNoiseParams();

    //=========================================================================================================
    /**
     * Set random seed for reproducible noise generation.
     */
    void setRandomSeed(int seed);

private:
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
     * Generate band-limited noise.
     */
    Eigen::MatrixXd generateBandLimitedNoise(int n_channels, int n_times, 
                                           double low_freq, double high_freq, double variance);

    //=========================================================================================================
    /**
     * Generate sinusoidal signal.
     */
    Eigen::VectorXd generateSinusoid(int n_times, double frequency, double amplitude, double phase = 0.0);

    //=========================================================================================================
    /**
     * Generate Gaussian window.
     */
    Eigen::VectorXd generateGaussianWindow(int n_times, double center, double width);

    //=========================================================================================================
    /**
     * Apply spatial pattern to artifact.
     */
    void applySpatialPattern(Eigen::MatrixXd& data, const Eigen::VectorXd& artifact, 
                           const std::vector<int>& channels, const std::vector<double>& weights);

    //=========================================================================================================
    /**
     * Detect channel types for artifact simulation.
     */
    void detectChannelTypes();

    // Member variables
    FIFFLIB::FiffInfo::SPtr m_pInfo;                    // Measurement info
    
    // Random number generation
    std::mt19937 m_randomGenerator;                     // Random number generator
    std::normal_distribution<double> m_normalDist;      // Normal distribution
    std::uniform_real_distribution<double> m_uniformDist; // Uniform distribution
    
    // Channel information
    int m_nChannels;                                    // Number of channels
    double m_samplingFreq;                              // Sampling frequency
    std::vector<int> m_magChannels;                     // Magnetometer channels
    std::vector<int> m_gradChannels;                    // Gradiometer channels
    std::vector<int> m_eegChannels;                     // EEG channels
    std::vector<int> m_frontalChannels;                 // Frontal channels (for EOG)
};

//=============================================================================================================
/**
 * Convenience function to add noise to data.
 * 
 * @param[in,out] data      Data matrix.
 * @param[in] info          Measurement info.
 * @param[in] noise_level   Noise level.
 * @param[in] random_seed   Random seed.
 */
SIMULATIONSHARED_EXPORT void add_noise(Eigen::MatrixXd& data,
                                      const FIFFLIB::FiffInfo::SPtr& info,
                                      double noise_level,
                                      int random_seed = -1);

//=============================================================================================================
/**
 * Convenience function to add ECG artifacts.
 * 
 * @param[in,out] data      Data matrix.
 * @param[in] info          Measurement info.
 * @param[in] params        ECG parameters.
 */
SIMULATIONSHARED_EXPORT void add_ecg(Eigen::MatrixXd& data,
                                    const FIFFLIB::FiffInfo::SPtr& info,
                                    const NoiseSimulation::ECGParams& params = NoiseSimulation::ECGParams());

//=============================================================================================================
/**
 * Convenience function to add EOG artifacts.
 * 
 * @param[in,out] data      Data matrix.
 * @param[in] info          Measurement info.
 * @param[in] params        EOG parameters.
 */
SIMULATIONSHARED_EXPORT void add_eog(Eigen::MatrixXd& data,
                                    const FIFFLIB::FiffInfo::SPtr& info,
                                    const NoiseSimulation::EOGParams& params = NoiseSimulation::EOGParams());

//=============================================================================================================
/**
 * Convenience function to add cHPI artifacts.
 * 
 * @param[in,out] data      Data matrix.
 * @param[in] info          Measurement info.
 * @param[in] params        cHPI parameters.
 */
SIMULATIONSHARED_EXPORT void add_chpi(Eigen::MatrixXd& data,
                                     const FIFFLIB::FiffInfo::SPtr& info,
                                     const NoiseSimulation::CHPIParams& params = NoiseSimulation::CHPIParams());

} // namespace SIMULATIONLIB

#endif // NOISE_SIMULATION_H