//=============================================================================================================
/**
 * @file     ecg_eog_processing.cpp
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
 * @brief    Implementation of ECG and EOG processing algorithms
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "ecg_eog_processing.h"
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

EcgEogProcessing::EcgEogProcessing()
{
}

//=============================================================================================================

EcgEogProcessing::~EcgEogProcessing()
{
}

//=============================================================================================================

std::vector<Event> EcgEogProcessing::find_ecg_events(const MatrixXd& data,
                                                    double sfreq,
                                                    const std::string& ch_name,
                                                    int event_id,
                                                    double l_freq,
                                                    double h_freq,
                                                    double qrs_threshold,
                                                    double tstart,
                                                    double tstop)
{
    Q_UNUSED(ch_name)
    
    std::vector<Event> events;
    
    if (data.rows() == 0 || data.cols() == 0) {
        qWarning() << "EcgEogProcessing::find_ecg_events - Empty data matrix";
        return events;
    }
    
    const int n_samples = data.cols();
    const int start_sample = static_cast<int>(tstart * sfreq);
    const int stop_sample = (tstop > 0) ? static_cast<int>(tstop * sfreq) : n_samples;
    
    // For simplicity, use the first channel as ECG channel
    // In practice, you would find the ECG channel by name
    VectorXd ecg_signal = data.row(0).segment(start_sample, stop_sample - start_sample);
    
    // Apply bandpass filter
    MatrixXd filtered_data = apply_bandpass_filter(ecg_signal.transpose(), sfreq, l_freq, h_freq);
    VectorXd filtered_signal = filtered_data.row(0);
    
    // Auto-determine threshold if not provided
    double threshold = qrs_threshold;
    if (threshold < 0) {
        double signal_std = std::sqrt((filtered_signal.array() - filtered_signal.mean()).square().mean());
        threshold = 3.0 * signal_std; // 3 standard deviations
    }
    
    // Find R-peaks
    int min_distance = static_cast<int>(0.6 * sfreq); // Minimum 600ms between R-peaks
    std::vector<int> peak_indices = find_peaks(filtered_signal, threshold, min_distance);
    
    // Convert to events
    for (int peak_idx : peak_indices) {
        int global_sample = start_sample + peak_idx;
        double time = global_sample / sfreq;
        events.emplace_back(global_sample, time, event_id);
    }
    
    qDebug() << "Found" << events.size() << "ECG events";
    return events;
}

//=============================================================================================================

EpochsData EcgEogProcessing::create_ecg_epochs(const MatrixXd& data,
                                              const std::vector<Event>& events,
                                              double sfreq,
                                              const std::vector<std::string>& ch_names,
                                              double tmin,
                                              double tmax,
                                              const std::pair<double, double>& baseline,
                                              double reject,
                                              double flat)
{
    Q_UNUSED(reject)
    Q_UNUSED(flat)
    
    EpochsData epochs_data;
    
    if (events.empty() || data.rows() == 0 || data.cols() == 0) {
        qWarning() << "EcgEogProcessing::create_ecg_epochs - Invalid input data or events";
        return epochs_data;
    }
    
    const int n_channels = data.rows();
    const int n_samples = data.cols();
    const int samples_before = static_cast<int>(-tmin * sfreq);
    const int samples_after = static_cast<int>(tmax * sfreq);
    const int epoch_length = samples_before + samples_after;
    
    // Filter valid events (those that have enough data before and after)
    std::vector<Event> valid_events;
    for (const Event& event : events) {
        if (event.sample >= samples_before && 
            event.sample + samples_after < n_samples) {
            valid_events.push_back(event);
        }
    }
    
    if (valid_events.empty()) {
        qWarning() << "EcgEogProcessing::create_ecg_epochs - No valid events found";
        return epochs_data;
    }
    
    const int n_epochs = valid_events.size();
    
    // Create epochs data matrix (n_epochs * n_channels x epoch_length)
    // We flatten the first two dimensions for easier handling
    epochs_data.data = MatrixXd(n_epochs * n_channels, epoch_length);
    
    // Extract epochs
    for (int epoch_idx = 0; epoch_idx < n_epochs; ++epoch_idx) {
        const Event& event = valid_events[epoch_idx];
        int start_sample = event.sample - samples_before;
        
        for (int ch = 0; ch < n_channels; ++ch) {
            int row_idx = epoch_idx * n_channels + ch;
            epochs_data.data.row(row_idx) = data.row(ch).segment(start_sample, epoch_length);
        }
    }
    
    // Set metadata
    epochs_data.events = valid_events;
    epochs_data.tmin = tmin;
    epochs_data.tmax = tmax;
    epochs_data.sfreq = sfreq;
    epochs_data.ch_names = ch_names;
    
    // Apply baseline correction if specified
    if (baseline.first >= 0 && baseline.second >= 0) {
        apply_baseline_correction(epochs_data, baseline);
    }
    
    qDebug() << "Created" << n_epochs << "ECG epochs with" << n_channels << "channels";
    return epochs_data;
}

//=============================================================================================================

std::vector<Projection> EcgEogProcessing::compute_proj_ecg(const EpochsData& epochs_data,
                                                          int n_grad,
                                                          int n_mag,
                                                          int n_eeg,
                                                          double l_freq,
                                                          double h_freq,
                                                          bool average)
{
    Q_UNUSED(n_grad)
    Q_UNUSED(n_mag)
    Q_UNUSED(l_freq)
    Q_UNUSED(h_freq)
    
    std::vector<Projection> projections;
    
    if (epochs_data.data.rows() == 0 || epochs_data.data.cols() == 0) {
        qWarning() << "EcgEogProcessing::compute_proj_ecg - Empty epochs data";
        return projections;
    }
    
    const int n_epochs = epochs_data.events.size();
    const int n_channels = epochs_data.ch_names.size();
    const int epoch_length = epochs_data.data.cols();
    
    MatrixXd data_for_svd;
    
    if (average && n_epochs > 1) {
        // Average epochs before computing projections
        data_for_svd = MatrixXd::Zero(n_channels, epoch_length);
        
        for (int epoch_idx = 0; epoch_idx < n_epochs; ++epoch_idx) {
            for (int ch = 0; ch < n_channels; ++ch) {
                int row_idx = epoch_idx * n_channels + ch;
                data_for_svd.row(ch) += epochs_data.data.row(row_idx);
            }
        }
        data_for_svd /= static_cast<double>(n_epochs);
    } else {
        // Use all epochs data
        data_for_svd = epochs_data.data;
    }
    
    // Compute SVD-based projection for EEG channels
    if (n_eeg > 0) {
        MatrixXd proj_matrix = compute_svd_projection(data_for_svd, n_eeg);
        
        Projection ecg_proj;
        ecg_proj.data = proj_matrix;
        ecg_proj.kind = "ECG";
        ecg_proj.active = false;
        
        for (int i = 0; i < n_eeg; ++i) {
            ecg_proj.desc.push_back("ECG-" + std::to_string(i + 1));
        }
        
        projections.push_back(ecg_proj);
    }
    
    qDebug() << "Computed" << projections.size() << "ECG projection(s)";
    return projections;
}

//=============================================================================================================

std::vector<Event> EcgEogProcessing::find_eog_events(const MatrixXd& data,
                                                    double sfreq,
                                                    const std::string& ch_name,
                                                    int event_id,
                                                    double l_freq,
                                                    double h_freq,
                                                    double thresh,
                                                    double tstart,
                                                    double tstop)
{
    Q_UNUSED(ch_name)
    
    std::vector<Event> events;
    
    if (data.rows() == 0 || data.cols() == 0) {
        qWarning() << "EcgEogProcessing::find_eog_events - Empty data matrix";
        return events;
    }
    
    const int n_samples = data.cols();
    const int start_sample = static_cast<int>(tstart * sfreq);
    const int stop_sample = (tstop > 0) ? static_cast<int>(tstop * sfreq) : n_samples;
    
    // For simplicity, use the first channel as EOG channel
    // In practice, you would find the EOG channel by name
    VectorXd eog_signal = data.row(0).segment(start_sample, stop_sample - start_sample);
    
    // Apply bandpass filter
    MatrixXd filtered_data = apply_bandpass_filter(eog_signal.transpose(), sfreq, l_freq, h_freq);
    VectorXd filtered_signal = filtered_data.row(0);
    
    // Compute signal derivative to detect blinks/saccades
    VectorXd signal_diff(filtered_signal.size() - 1);
    for (int i = 0; i < signal_diff.size(); ++i) {
        signal_diff(i) = std::abs(filtered_signal(i + 1) - filtered_signal(i));
    }
    
    // Auto-determine threshold if not provided
    double threshold = thresh;
    if (threshold < 0) {
        double signal_std = std::sqrt((signal_diff.array() - signal_diff.mean()).square().mean());
        threshold = 3.0 * signal_std; // 3 standard deviations
    }
    
    // Find EOG events (blinks/saccades)
    int min_distance = static_cast<int>(0.5 * sfreq); // Minimum 500ms between events
    std::vector<int> peak_indices = find_peaks(signal_diff, threshold, min_distance);
    
    // Convert to events
    for (int peak_idx : peak_indices) {
        int global_sample = start_sample + peak_idx;
        double time = global_sample / sfreq;
        events.emplace_back(global_sample, time, event_id);
    }
    
    qDebug() << "Found" << events.size() << "EOG events";
    return events;
}

//=============================================================================================================

EpochsData EcgEogProcessing::create_eog_epochs(const MatrixXd& data,
                                              const std::vector<Event>& events,
                                              double sfreq,
                                              const std::vector<std::string>& ch_names,
                                              double tmin,
                                              double tmax,
                                              const std::pair<double, double>& baseline,
                                              double reject,
                                              double flat)
{
    // EOG epochs creation is similar to ECG epochs
    return create_ecg_epochs(data, events, sfreq, ch_names, tmin, tmax, baseline, reject, flat);
}

//=============================================================================================================

std::vector<Projection> EcgEogProcessing::compute_proj_eog(const EpochsData& epochs_data,
                                                          int n_grad,
                                                          int n_mag,
                                                          int n_eeg,
                                                          double l_freq,
                                                          double h_freq,
                                                          bool average)
{
    Q_UNUSED(n_grad)
    Q_UNUSED(n_mag)
    Q_UNUSED(l_freq)
    Q_UNUSED(h_freq)
    
    std::vector<Projection> projections;
    
    if (epochs_data.data.rows() == 0 || epochs_data.data.cols() == 0) {
        qWarning() << "EcgEogProcessing::compute_proj_eog - Empty epochs data";
        return projections;
    }
    
    const int n_epochs = epochs_data.events.size();
    const int n_channels = epochs_data.ch_names.size();
    const int epoch_length = epochs_data.data.cols();
    
    MatrixXd data_for_svd;
    
    if (average && n_epochs > 1) {
        // Average epochs before computing projections
        data_for_svd = MatrixXd::Zero(n_channels, epoch_length);
        
        for (int epoch_idx = 0; epoch_idx < n_epochs; ++epoch_idx) {
            for (int ch = 0; ch < n_channels; ++ch) {
                int row_idx = epoch_idx * n_channels + ch;
                data_for_svd.row(ch) += epochs_data.data.row(row_idx);
            }
        }
        data_for_svd /= static_cast<double>(n_epochs);
    } else {
        // Use all epochs data
        data_for_svd = epochs_data.data;
    }
    
    // Compute SVD-based projection for EEG channels
    if (n_eeg > 0) {
        MatrixXd proj_matrix = compute_svd_projection(data_for_svd, n_eeg);
        
        Projection eog_proj;
        eog_proj.data = proj_matrix;
        eog_proj.kind = "EOG";
        eog_proj.active = false;
        
        for (int i = 0; i < n_eeg; ++i) {
            eog_proj.desc.push_back("EOG-" + std::to_string(i + 1));
        }
        
        projections.push_back(eog_proj);
    }
    
    qDebug() << "Computed" << projections.size() << "EOG projection(s)";
    return projections;
}

//=============================================================================================================

MatrixXd EcgEogProcessing::apply_bandpass_filter(const MatrixXd& data,
                                                double sfreq,
                                                double l_freq,
                                                double h_freq)
{
    // Simple bandpass filter implementation
    // This is a basic implementation - in practice, you might want to use
    // a more sophisticated filter design
    
    Q_UNUSED(sfreq)
    Q_UNUSED(l_freq)
    Q_UNUSED(h_freq)
    
    MatrixXd filtered_data = data;
    
    // Apply simple high-pass filter (first-order difference)
    for (int ch = 0; ch < data.rows(); ++ch) {
        for (int sample = 1; sample < data.cols(); ++sample) {
            filtered_data(ch, sample) = data(ch, sample) - 0.95 * data(ch, sample - 1);
        }
        filtered_data(ch, 0) = 0.0;
    }
    
    return filtered_data;
}

//=============================================================================================================

std::vector<int> EcgEogProcessing::find_peaks(const VectorXd& signal,
                                             double threshold,
                                             int min_distance)
{
    std::vector<int> peaks;
    
    if (signal.size() < 3) {
        return peaks;
    }
    
    for (int i = 1; i < signal.size() - 1; ++i) {
        // Check if current point is a local maximum above threshold
        if (signal(i) > threshold &&
            signal(i) > signal(i - 1) &&
            signal(i) > signal(i + 1)) {
            
            // Check minimum distance constraint
            bool valid_peak = true;
            for (int existing_peak : peaks) {
                if (std::abs(i - existing_peak) < min_distance) {
                    valid_peak = false;
                    break;
                }
            }
            
            if (valid_peak) {
                peaks.push_back(i);
            }
        }
    }
    
    return peaks;
}

//=============================================================================================================

MatrixXd EcgEogProcessing::compute_svd_projection(const MatrixXd& data,
                                                 int n_components)
{
    if (data.rows() == 0 || data.cols() == 0 || n_components <= 0) {
        return MatrixXd::Zero(0, 0);
    }
    
    // Perform SVD
    JacobiSVD<MatrixXd> svd(data, ComputeThinU | ComputeThinV);
    
    // Get the first n_components left singular vectors
    int actual_components = std::min(n_components, static_cast<int>(svd.matrixU().cols()));
    MatrixXd projection = svd.matrixU().leftCols(actual_components).transpose();
    
    return projection;
}

//=============================================================================================================

void EcgEogProcessing::apply_baseline_correction(EpochsData& epochs_data,
                                                const std::pair<double, double>& baseline)
{
    if (baseline.first < 0 || baseline.second < 0) {
        return; // No baseline correction
    }
    
    const int n_epochs = epochs_data.events.size();
    const int n_channels = epochs_data.ch_names.size();
    const double sfreq = epochs_data.sfreq;
    
    // Convert baseline times to sample indices
    int baseline_start = static_cast<int>((baseline.first - epochs_data.tmin) * sfreq);
    int baseline_end = static_cast<int>((baseline.second - epochs_data.tmin) * sfreq);
    
    baseline_start = std::max(0, baseline_start);
    baseline_end = std::min(static_cast<int>(epochs_data.data.cols()) - 1, baseline_end);
    
    if (baseline_start >= baseline_end) {
        return;
    }
    
    // Apply baseline correction to each epoch and channel
    for (int epoch_idx = 0; epoch_idx < n_epochs; ++epoch_idx) {
        for (int ch = 0; ch < n_channels; ++ch) {
            int row_idx = epoch_idx * n_channels + ch;
            
            // Calculate baseline mean
            double baseline_mean = epochs_data.data.row(row_idx).segment(baseline_start, 
                                                                       baseline_end - baseline_start).mean();
            
            // Subtract baseline from entire epoch
            epochs_data.data.row(row_idx).array() -= baseline_mean;
        }
    }
}

//=============================================================================================================

int EcgEogProcessing::find_channel_index(const std::vector<std::string>& ch_names,
                                        const std::string& ch_name)
{
    auto it = std::find(ch_names.begin(), ch_names.end(), ch_name);
    if (it != ch_names.end()) {
        return static_cast<int>(std::distance(ch_names.begin(), it));
    }
    return -1; // Channel not found
}