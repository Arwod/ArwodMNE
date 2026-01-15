//=============================================================================================================
/**
 * @file     artifact_detection.h
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
 * @brief    Artifact detection algorithms for automatic identification of artifacts in EEG/MEG data
 *
 */

#ifndef ARTIFACT_DETECTION_H
#define ARTIFACT_DETECTION_H

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
 * Structure to represent an annotation (artifact detection result)
 */
struct PREPROCESSINGSHARED_EXPORT Annotation
{
    double onset;           /**< Start time of the annotation in seconds */
    double duration;        /**< Duration of the annotation in seconds */
    std::string description; /**< Description of the annotation */
    
    Annotation(double onset = 0.0, double duration = 0.0, const std::string& description = "")
        : onset(onset), duration(duration), description(description) {}
};

//=============================================================================================================
/**
 * Artifact detection algorithms for automatic identification of artifacts in EEG/MEG data
 *
 * @brief The ArtifactDetection class provides methods for detecting various types of artifacts
 * in electrophysiological data including amplitude-based, muscle, and movement artifacts.
 */
class PREPROCESSINGSHARED_EXPORT ArtifactDetection
{

public:
    typedef QSharedPointer<ArtifactDetection> SPtr;            /**< Shared pointer type for ArtifactDetection. */
    typedef QSharedPointer<const ArtifactDetection> ConstSPtr; /**< Const shared pointer type for ArtifactDetection. */

    //=========================================================================================================
    /**
     * Constructs an ArtifactDetection object.
     */
    explicit ArtifactDetection();

    //=========================================================================================================
    /**
     * Destructor
     */
    ~ArtifactDetection();

    //=========================================================================================================
    /**
     * Detect amplitude-based artifacts in the data
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] peak          Peak amplitude threshold (default: 100e-6 for EEG, 4000e-15 for MEG)
     * @param[in] flat          Flat signal threshold (default: 1e-6 for EEG, 1e-15 for MEG)
     * @param[in] bad_percent   Percentage of channels that must be bad to mark as artifact (default: 5.0)
     * @param[in] min_duration  Minimum duration of artifact in seconds (default: 0.002)
     * @param[in] reject_by_annotation  Whether to reject based on existing annotations (default: true)
     *
     * @return Vector of annotations representing detected amplitude artifacts
     */
    static std::vector<Annotation> annotate_amplitude(const Eigen::MatrixXd& data,
                                                     double sfreq,
                                                     double peak = 100e-6,
                                                     double flat = 1e-6,
                                                     double bad_percent = 5.0,
                                                     double min_duration = 0.002,
                                                     bool reject_by_annotation = true);

    //=========================================================================================================
    /**
     * Detect muscle artifacts using z-score based method
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] threshold     Z-score threshold for detection (default: 4.0)
     * @param[in] min_length_good  Minimum length of good data between artifacts in seconds (default: 0.2)
     * @param[in] filter_freq   High-pass filter frequency for muscle detection (default: 110.0 Hz)
     * @param[in] n_jobs        Number of parallel jobs (default: 1)
     *
     * @return Vector of annotations representing detected muscle artifacts
     */
    static std::vector<Annotation> annotate_muscle_zscore(const Eigen::MatrixXd& data,
                                                         double sfreq,
                                                         double threshold = 4.0,
                                                         double min_length_good = 0.2,
                                                         double filter_freq = 110.0,
                                                         int n_jobs = 1);

    //=========================================================================================================
    /**
     * Detect movement artifacts based on signal variance changes
     *
     * @param[in] data          Input data matrix (channels x samples)
     * @param[in] sfreq         Sampling frequency in Hz
     * @param[in] threshold     Threshold for movement detection (default: 5.0)
     * @param[in] min_duration  Minimum duration of movement artifact in seconds (default: 0.1)
     * @param[in] window_size   Window size for variance calculation in seconds (default: 1.0)
     *
     * @return Vector of annotations representing detected movement artifacts
     */
    static std::vector<Annotation> annotate_movement(const Eigen::MatrixXd& data,
                                                    double sfreq,
                                                    double threshold = 5.0,
                                                    double min_duration = 0.1,
                                                    double window_size = 1.0);

private:
    //=========================================================================================================
    /**
     * Apply high-pass filter to the data
     *
     * @param[in] data          Input data matrix
     * @param[in] sfreq         Sampling frequency
     * @param[in] cutoff        Cutoff frequency for high-pass filter
     *
     * @return Filtered data matrix
     */
    static Eigen::MatrixXd apply_highpass_filter(const Eigen::MatrixXd& data,
                                                double sfreq,
                                                double cutoff);

    //=========================================================================================================
    /**
     * Calculate z-scores for the data
     *
     * @param[in] data          Input data vector
     *
     * @return Z-scores vector
     */
    static Eigen::VectorXd calculate_zscore(const Eigen::VectorXd& data);

    //=========================================================================================================
    /**
     * Merge overlapping annotations
     *
     * @param[in] annotations   Input annotations
     * @param[in] min_gap       Minimum gap between annotations to keep them separate
     *
     * @return Merged annotations
     */
    static std::vector<Annotation> merge_annotations(const std::vector<Annotation>& annotations,
                                                    double min_gap = 0.0);

    //=========================================================================================================
    /**
     * Calculate moving window variance
     *
     * @param[in] data          Input data vector
     * @param[in] window_size   Window size in samples
     *
     * @return Variance vector
     */
    static Eigen::VectorXd calculate_moving_variance(const Eigen::VectorXd& data,
                                                   int window_size);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE PREPROCESSINGLIB

#endif // ARTIFACT_DETECTION_H