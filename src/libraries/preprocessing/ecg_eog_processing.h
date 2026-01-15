//=============================================================================================================
/**
 * @file     ecg_eog_processing.h
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, Kiro AI Assistant. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    ECG and EOG processing algorithms for artifact detection and removal
 *
 */

#ifndef ECG_EOG_PROCESSING_H
#define ECG_EOG_PROCESSING_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "preprocessing_global.h"
#include <Eigen/Dense>
#include <vector>
#include <string>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QSharedPointer>

//=============================================================================================================
// FORWARD DECLARATIONS
//=============================================================================================================

//=============================================================================================================
// DEFINE NAMESPACE PREPROCESSINGLIB
//=============================================================================================================

namespace PREPROCESSINGLIB
{

//=============================================================================================================
// PREPROCESSINGLIB FORWARD DECLARATIONS
//=============================================================================================================

//=============================================================================================================
/**
 * Structure to represent an event (ECG or EOG peak)
 */
struct PREPROCESSINGSHARED_EXPORT Event
{
    int sample;             /**< Sample index of the event */
    double time;            /**< Time of the event in seconds */
    int event_id;           /**< Event ID (default: 999 for ECG, 998 for EOG) */
    
    Event(int sample = 0, double time = 0.0, int event_id = 999)
        : sample(sample), time(time), event_id(event_id) {}
};

//=============================================================================================================
/**
 * Structure to represent epochs data
 */
struct PREPROCESSINGSHARED_EXPORT EpochsData
{
    Eigen::MatrixXd data;           /**< Epochs data (n_epochs x n_channels x n_times) */
    std::vector<Event> events;      /**< Events used to create epochs */
    double tmin;                    /**< Start time of epochs relative to events */
    double tmax;                    /**< End time of epochs relative to events */
    double sfreq;                   /**< Sampling frequency */
    std::vector<std::string> ch_names; /**< Channel names */
    
    EpochsData() : tmin(0.0), tmax(0.0), sfreq(0.0) {}
};

//=============================================================================================================
/**
 * Structure to represent projection vectors
 */
struct PREPROCESSINGSHARED_EXPORT Projection
{
    Eigen::MatrixXd data;           /**< Projection vectors (n_components x n_channels) */
    std::vector<std::string> desc;  /**< Description of each projection component */
    std::string kind;               /**< Type of projection (ECG or EOG) */
    bool active;                    /**< Whether projection is active */
    
    Projection() : active(false) {}
};

//=============================================================================================================
/**
 * ECG and EOG processing algorithms for artifact detection and removal
 *
 * @brief The EcgEogProcessing class provides methods for detecting ECG and EOG artifacts
 * and computing projection vectors for artifact removal.
 */
class PREPROCESSINGSHARED_EXPORT EcgEogProcessing
{

public:
    typedef QSharedPointer<EcgEogProcessing> SPtr;            /**< Shared pointer type for EcgEogProcessing. */
    typedef QSharedPointer<const EcgEogProcessing> ConstSPtr; /**< Const shared pointer type for EcgEogProcessing. */

    //=========================================================================================================
    /**
     * Constructs an EcgEogProcessing object.
     */
    explicit EcgEogProcessing();

    //=========================================================================================================
    /**
     * Destructor
     */
    ~EcgEogProcessing();

    //=========================================================================================================
    /**
     * Find ECG events (R-peaks) in the data
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] ch_name       Name of ECG channel (default: "ECG")
     * @param[in] event_id      Event ID for ECG events (default: 999)
     * @param[in] l_freq        Low-pass filter frequency (default: 5.0 Hz)
     * @param[in] h_freq        High-pass filter frequency (default: 35.0 Hz)
     * @param[in] qrs_threshold Threshold for QRS detection (default: auto)
     * @param[in] tstart        Start time for event detection (default: 0.0)
     * @param[in] tstop         Stop time for event detection (default: end of data)
     *
     * @return Vector of ECG events
     */
    static std::vector<Event> find_ecg_events(const Eigen::MatrixXd& data,
                                             double sfreq,
                                             const std::string& ch_name = "ECG",
                                             int event_id = 999,
                                             double l_freq = 5.0,
                                             double h_freq = 35.0,
                                             double qrs_threshold = -1.0,
                                             double tstart = 0.0,
                                             double tstop = -1.0);

    //=========================================================================================================
    /**
     * Create ECG epochs from raw data
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] events        ECG events
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] ch_names      Channel names
     * @param[in] tmin          Start time of epochs relative to events (default: -0.2)
     * @param[in] tmax          End time of epochs relative to events (default: 0.4)
     * @param[in] baseline      Baseline correction period (default: None)
     * @param[in] reject        Rejection criteria (default: None)
     * @param[in] flat          Flat signal rejection criteria (default: None)
     *
     * @return ECG epochs data
     */
    static EpochsData create_ecg_epochs(const Eigen::MatrixXd& data,
                                       const std::vector<Event>& events,
                                       double sfreq,
                                       const std::vector<std::string>& ch_names,
                                       double tmin = -0.2,
                                       double tmax = 0.4,
                                       const std::pair<double, double>& baseline = std::make_pair(-1.0, -1.0),
                                       double reject = -1.0,
                                       double flat = -1.0);

    //=========================================================================================================
    /**
     * Compute ECG projection vectors for artifact removal
     *
     * @param[in] epochs_data   ECG epochs data
     * @param[in] n_grad        Number of gradient components (default: 2)
     * @param[in] n_mag         Number of magnetometer components (default: 2)
     * @param[in] n_eeg         Number of EEG components (default: 2)
     * @param[in] l_freq        Low-pass filter frequency (default: 1.0 Hz)
     * @param[in] h_freq        High-pass filter frequency (default: 35.0 Hz)
     * @param[in] average       Whether to average epochs before computing projections (default: true)
     *
     * @return ECG projection vectors
     */
    static std::vector<Projection> compute_proj_ecg(const EpochsData& epochs_data,
                                                   int n_grad = 2,
                                                   int n_mag = 2,
                                                   int n_eeg = 2,
                                                   double l_freq = 1.0,
                                                   double h_freq = 35.0,
                                                   bool average = true);

    //=========================================================================================================
    /**
     * Find EOG events (blinks and saccades) in the data
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] ch_name       Name of EOG channel (default: "EOG")
     * @param[in] event_id      Event ID for EOG events (default: 998)
     * @param[in] l_freq        Low-pass filter frequency (default: 1.0 Hz)
     * @param[in] h_freq        High-pass filter frequency (default: 10.0 Hz)
     * @param[in] thresh        Threshold for EOG detection (default: auto)
     * @param[in] tstart        Start time for event detection (default: 0.0)
     * @param[in] tstop         Stop time for event detection (default: end of data)
     *
     * @return Vector of EOG events
     */
    static std::vector<Event> find_eog_events(const Eigen::MatrixXd& data,
                                             double sfreq,
                                             const std::string& ch_name = "EOG",
                                             int event_id = 998,
                                             double l_freq = 1.0,
                                             double h_freq = 10.0,
                                             double thresh = -1.0,
                                             double tstart = 0.0,
                                             double tstop = -1.0);

    //=========================================================================================================
    /**
     * Create EOG epochs from raw data
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] events        EOG events
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] ch_names      Channel names
     * @param[in] tmin          Start time of epochs relative to events (default: -0.5)
     * @param[in] tmax          End time of epochs relative to events (default: 0.5)
     * @param[in] baseline      Baseline correction period (default: None)
     * @param[in] reject        Rejection criteria (default: None)
     * @param[in] flat          Flat signal rejection criteria (default: None)
     *
     * @return EOG epochs data
     */
    static EpochsData create_eog_epochs(const Eigen::MatrixXd& data,
                                       const std::vector<Event>& events,
                                       double sfreq,
                                       const std::vector<std::string>& ch_names,
                                       double tmin = -0.5,
                                       double tmax = 0.5,
                                       const std::pair<double, double>& baseline = std::make_pair(-1.0, -1.0),
                                       double reject = -1.0,
                                       double flat = -1.0);

    //=========================================================================================================
    /**
     * Compute EOG projection vectors for artifact removal
     *
     * @param[in] epochs_data   EOG epochs data
     * @param[in] n_grad        Number of gradient components (default: 2)
     * @param[in] n_mag         Number of magnetometer components (default: 2)
     * @param[in] n_eeg         Number of EEG components (default: 2)
     * @param[in] l_freq        Low-pass filter frequency (default: 1.0 Hz)
     * @param[in] h_freq        High-pass filter frequency (default: 10.0 Hz)
     * @param[in] average       Whether to average epochs before computing projections (default: true)
     *
     * @return EOG projection vectors
     */
    static std::vector<Projection> compute_proj_eog(const EpochsData& epochs_data,
                                                   int n_grad = 2,
                                                   int n_mag = 2,
                                                   int n_eeg = 2,
                                                   double l_freq = 1.0,
                                                   double h_freq = 10.0,
                                                   bool average = true);

private:
    //=========================================================================================================
    /**
     * Apply bandpass filter to the data
     *
     * @param[in] data          Input data matrix
     * @param[in] sfreq         Sampling frequency
     * @param[in] l_freq        Low-pass cutoff frequency
     * @param[in] h_freq        High-pass cutoff frequency
     *
     * @return Filtered data matrix
     */
    static Eigen::MatrixXd apply_bandpass_filter(const Eigen::MatrixXd& data,
                                                double sfreq,
                                                double l_freq,
                                                double h_freq);

    //=========================================================================================================
    /**
     * Find peaks in a signal
     *
     * @param[in] signal        Input signal
     * @param[in] threshold     Peak detection threshold
     * @param[in] min_distance  Minimum distance between peaks in samples
     *
     * @return Vector of peak indices
     */
    static std::vector<int> find_peaks(const Eigen::VectorXd& signal,
                                      double threshold,
                                      int min_distance = 1);

    //=========================================================================================================
    /**
     * Compute SVD-based projection vectors
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] n_components  Number of components to extract
     *
     * @return Projection matrix (n_components x n_channels)
     */
    static Eigen::MatrixXd compute_svd_projection(const Eigen::MatrixXd& data,
                                                 int n_components);

    //=========================================================================================================
    /**
     * Apply baseline correction to epochs
     *
     * @param[in,out] epochs_data  Epochs data to correct
     * @param[in] baseline         Baseline period (start, end) in seconds
     */
    static void apply_baseline_correction(EpochsData& epochs_data,
                                        const std::pair<double, double>& baseline);

    //=========================================================================================================
    /**
     * Find channel index by name
     *
     * @param[in] ch_names      Channel names
     * @param[in] ch_name       Channel name to find
     *
     * @return Channel index (-1 if not found)
     */
    static int find_channel_index(const std::vector<std::string>& ch_names,
                                 const std::string& ch_name);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE PREPROCESSINGLIB

#endif // ECG_EOG_PROCESSING_H