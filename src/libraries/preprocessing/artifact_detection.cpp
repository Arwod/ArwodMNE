//=============================================================================================================
/**
 * @file     artifact_detection.cpp
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
 * @brief    Implementation of artifact detection algorithms
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "artifact_detection.h"
#include <algorithm>
#include <cmath>
#include <numeric>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace PREPROCESSINGLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

ArtifactDetection::ArtifactDetection()
{
}

//=============================================================================================================

ArtifactDetection::~ArtifactDetection()
{
}

//=============================================================================================================

std::vector<Annotation> ArtifactDetection::annotate_amplitude(const MatrixXd& data,
                                                             double sfreq,
                                                             double peak,
                                                             double flat,
                                                             double bad_percent,
                                                             double min_duration,
                                                             bool reject_by_annotation)
{
    Q_UNUSED(reject_by_annotation)
    
    std::vector<Annotation> annotations;
    
    if (data.rows() == 0 || data.cols() == 0) {
        qWarning() << "ArtifactDetection::annotate_amplitude - Empty data matrix";
        return annotations;
    }
    
    const int n_channels = data.rows();
    const int n_samples = data.cols();
    const int min_samples = static_cast<int>(min_duration * sfreq);
    const double bad_channel_threshold = (bad_percent / 100.0) * n_channels;
    
    // Track bad samples for each channel
    MatrixXi bad_samples = MatrixXi::Zero(n_channels, n_samples);
    
    // Detect amplitude violations for each channel
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int sample = 0; sample < n_samples; ++sample) {
            double value = std::abs(data(ch, sample));
            
            // Check for peak violations (too high amplitude)
            if (value > peak) {
                bad_samples(ch, sample) = 1;
            }
            
            // Check for flat violations (too low amplitude)
            if (value < flat) {
                bad_samples(ch, sample) = 1;
            }
        }
    }
    
    // Find time points where enough channels are bad
    VectorXi bad_time_points = VectorXi::Zero(n_samples);
    for (int sample = 0; sample < n_samples; ++sample) {
        int bad_count = bad_samples.col(sample).sum();
        if (bad_count >= bad_channel_threshold) {
            bad_time_points(sample) = 1;
        }
    }
    
    // Find continuous segments of bad data
    bool in_artifact = false;
    int artifact_start = 0;
    
    for (int sample = 0; sample < n_samples; ++sample) {
        if (bad_time_points(sample) == 1 && !in_artifact) {
            // Start of artifact
            in_artifact = true;
            artifact_start = sample;
        } else if (bad_time_points(sample) == 0 && in_artifact) {
            // End of artifact
            in_artifact = false;
            int artifact_length = sample - artifact_start;
            
            if (artifact_length >= min_samples) {
                double onset = artifact_start / sfreq;
                double duration = artifact_length / sfreq;
                annotations.emplace_back(onset, duration, "BAD_amplitude");
            }
        }
    }
    
    // Handle case where artifact extends to end of data
    if (in_artifact) {
        int artifact_length = n_samples - artifact_start;
        if (artifact_length >= min_samples) {
            double onset = artifact_start / sfreq;
            double duration = artifact_length / sfreq;
            annotations.emplace_back(onset, duration, "BAD_amplitude");
        }
    }
    
    return annotations;
}

//=============================================================================================================

std::vector<Annotation> ArtifactDetection::annotate_muscle_zscore(const MatrixXd& data,
                                                                 double sfreq,
                                                                 double threshold,
                                                                 double min_length_good,
                                                                 double filter_freq,
                                                                 int n_jobs)
{
    Q_UNUSED(n_jobs)
    
    std::vector<Annotation> annotations;
    
    if (data.rows() == 0 || data.cols() == 0) {
        qWarning() << "ArtifactDetection::annotate_muscle_zscore - Empty data matrix";
        return annotations;
    }
    
    // Apply high-pass filter to isolate muscle activity
    MatrixXd filtered_data = apply_highpass_filter(data, sfreq, filter_freq);
    
    const int n_channels = filtered_data.rows();
    const int n_samples = filtered_data.cols();
    const int min_good_samples = static_cast<int>(min_length_good * sfreq);
    
    // Calculate power (squared amplitude) for each channel
    MatrixXd power_data = filtered_data.array().square();
    
    // Calculate z-scores for each channel
    MatrixXd zscore_data(n_channels, n_samples);
    for (int ch = 0; ch < n_channels; ++ch) {
        VectorXd channel_power = power_data.row(ch);
        VectorXd channel_zscore = calculate_zscore(channel_power);
        zscore_data.row(ch) = channel_zscore;
    }
    
    // Find samples where any channel exceeds threshold
    VectorXi bad_samples = VectorXi::Zero(n_samples);
    for (int sample = 0; sample < n_samples; ++sample) {
        for (int ch = 0; ch < n_channels; ++ch) {
            if (std::abs(zscore_data(ch, sample)) > threshold) {
                bad_samples(sample) = 1;
                break;
            }
        }
    }
    
    // Find continuous segments and merge close ones
    bool in_artifact = false;
    int artifact_start = 0;
    std::vector<std::pair<int, int>> artifact_segments;
    
    for (int sample = 0; sample < n_samples; ++sample) {
        if (bad_samples(sample) == 1 && !in_artifact) {
            in_artifact = true;
            artifact_start = sample;
        } else if (bad_samples(sample) == 0 && in_artifact) {
            in_artifact = false;
            artifact_segments.emplace_back(artifact_start, sample - 1);
        }
    }
    
    // Handle case where artifact extends to end of data
    if (in_artifact) {
        artifact_segments.emplace_back(artifact_start, n_samples - 1);
    }
    
    // Merge segments that are too close together
    std::vector<std::pair<int, int>> merged_segments;
    if (!artifact_segments.empty()) {
        merged_segments.push_back(artifact_segments[0]);
        
        for (size_t i = 1; i < artifact_segments.size(); ++i) {
            int gap = artifact_segments[i].first - merged_segments.back().second;
            if (gap < min_good_samples) {
                // Merge with previous segment
                merged_segments.back().second = artifact_segments[i].second;
            } else {
                merged_segments.push_back(artifact_segments[i]);
            }
        }
    }
    
    // Convert to annotations
    for (const auto& segment : merged_segments) {
        double onset = segment.first / sfreq;
        double duration = (segment.second - segment.first + 1) / sfreq;
        annotations.emplace_back(onset, duration, "BAD_muscle");
    }
    
    return annotations;
}

//=============================================================================================================

std::vector<Annotation> ArtifactDetection::annotate_movement(const MatrixXd& data,
                                                            double sfreq,
                                                            double threshold,
                                                            double min_duration,
                                                            double window_size)
{
    std::vector<Annotation> annotations;
    
    if (data.rows() == 0 || data.cols() == 0) {
        qWarning() << "ArtifactDetection::annotate_movement - Empty data matrix";
        return annotations;
    }
    
    const int n_channels = data.rows();
    const int n_samples = data.cols();
    const int window_samples = static_cast<int>(window_size * sfreq);
    const int min_duration_samples = static_cast<int>(min_duration * sfreq);
    
    if (window_samples >= n_samples) {
        qWarning() << "ArtifactDetection::annotate_movement - Window size too large for data";
        return annotations;
    }
    
    // Calculate moving variance for each channel
    MatrixXd variance_data(n_channels, n_samples - window_samples + 1);
    for (int ch = 0; ch < n_channels; ++ch) {
        VectorXd channel_data = data.row(ch);
        VectorXd channel_variance = calculate_moving_variance(channel_data, window_samples);
        variance_data.row(ch) = channel_variance;
    }
    
    // Calculate z-scores of variance for each channel
    MatrixXd zscore_variance(n_channels, variance_data.cols());
    for (int ch = 0; ch < n_channels; ++ch) {
        VectorXd channel_variance = variance_data.row(ch);
        VectorXd channel_zscore = calculate_zscore(channel_variance);
        zscore_variance.row(ch) = channel_zscore;
    }
    
    // Find samples where any channel variance exceeds threshold
    VectorXi bad_samples = VectorXi::Zero(variance_data.cols());
    for (int sample = 0; sample < variance_data.cols(); ++sample) {
        for (int ch = 0; ch < n_channels; ++ch) {
            if (std::abs(zscore_variance(ch, sample)) > threshold) {
                bad_samples(sample) = 1;
                break;
            }
        }
    }
    
    // Find continuous segments of movement artifacts
    bool in_artifact = false;
    int artifact_start = 0;
    
    for (int sample = 0; sample < bad_samples.size(); ++sample) {
        if (bad_samples(sample) == 1 && !in_artifact) {
            in_artifact = true;
            artifact_start = sample;
        } else if (bad_samples(sample) == 0 && in_artifact) {
            in_artifact = false;
            int artifact_length = sample - artifact_start;
            
            if (artifact_length >= min_duration_samples) {
                double onset = artifact_start / sfreq;
                double duration = artifact_length / sfreq;
                annotations.emplace_back(onset, duration, "BAD_movement");
            }
        }
    }
    
    // Handle case where artifact extends to end of data
    if (in_artifact) {
        int artifact_length = bad_samples.size() - artifact_start;
        if (artifact_length >= min_duration_samples) {
            double onset = artifact_start / sfreq;
            double duration = artifact_length / sfreq;
            annotations.emplace_back(onset, duration, "BAD_movement");
        }
    }
    
    return annotations;
}

//=============================================================================================================

MatrixXd ArtifactDetection::apply_highpass_filter(const MatrixXd& data,
                                                 double sfreq,
                                                 double cutoff)
{
    // Simple high-pass filter implementation using first-order difference
    // This is a basic implementation - in practice, you might want to use
    // a more sophisticated filter design
    
    Q_UNUSED(sfreq)
    Q_UNUSED(cutoff)
    
    MatrixXd filtered_data = data;
    
    // Apply first-order high-pass filter (simple difference)
    for (int ch = 0; ch < data.rows(); ++ch) {
        for (int sample = 1; sample < data.cols(); ++sample) {
            filtered_data(ch, sample) = data(ch, sample) - data(ch, sample - 1);
        }
        filtered_data(ch, 0) = 0.0; // Set first sample to zero
    }
    
    return filtered_data;
}

//=============================================================================================================

VectorXd ArtifactDetection::calculate_zscore(const VectorXd& data)
{
    if (data.size() == 0) {
        return VectorXd::Zero(0);
    }
    
    double mean = data.mean();
    double std_dev = std::sqrt((data.array() - mean).square().mean());
    
    if (std_dev == 0.0) {
        return VectorXd::Zero(data.size());
    }
    
    return (data.array() - mean) / std_dev;
}

//=============================================================================================================

std::vector<Annotation> ArtifactDetection::merge_annotations(const std::vector<Annotation>& annotations,
                                                            double min_gap)
{
    if (annotations.empty()) {
        return annotations;
    }
    
    // Sort annotations by onset time
    std::vector<Annotation> sorted_annotations = annotations;
    std::sort(sorted_annotations.begin(), sorted_annotations.end(),
              [](const Annotation& a, const Annotation& b) {
                  return a.onset < b.onset;
              });
    
    std::vector<Annotation> merged;
    merged.push_back(sorted_annotations[0]);
    
    for (size_t i = 1; i < sorted_annotations.size(); ++i) {
        const Annotation& current = sorted_annotations[i];
        Annotation& last_merged = merged.back();
        
        double gap = current.onset - (last_merged.onset + last_merged.duration);
        
        if (gap <= min_gap) {
            // Merge with previous annotation
            double new_end = std::max(last_merged.onset + last_merged.duration,
                                    current.onset + current.duration);
            last_merged.duration = new_end - last_merged.onset;
            last_merged.description += "+" + current.description;
        } else {
            merged.push_back(current);
        }
    }
    
    return merged;
}

//=============================================================================================================

VectorXd ArtifactDetection::calculate_moving_variance(const VectorXd& data,
                                                     int window_size)
{
    if (data.size() < window_size || window_size <= 0) {
        return VectorXd::Zero(0);
    }
    
    int n_windows = data.size() - window_size + 1;
    VectorXd variance(n_windows);
    
    for (int i = 0; i < n_windows; ++i) {
        VectorXd window = data.segment(i, window_size);
        double mean = window.mean();
        double var = (window.array() - mean).square().mean();
        variance(i) = var;
    }
    
    return variance;
}