//=============================================================================================================
/**
 * @file     mne_core_types.cpp
 * @author   MNE-CPP Migration Team
 * @since    1.0.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP Migration Team. All rights reserved.
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
 * @brief     Implementation of core data types for MNE-CPP migration project
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "mne_core_types.h"
#include <algorithm>
#include <numeric>
#include <cmath>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace MNELIB;

//=============================================================================================================
// BaseData Implementation
//=============================================================================================================

BaseData::BaseData()
: sfreq(1000.0)
{
}

BaseData::BaseData(const BaseData& other)
: ch_names(other.ch_names)
, times(other.times)
, sfreq(other.sfreq)
{
}

BaseData& BaseData::operator=(const BaseData& other)
{
    if (this != &other) {
        ch_names = other.ch_names;
        times = other.times;
        sfreq = other.sfreq;
    }
    return *this;
}

//=============================================================================================================
// TimeSeriesData Implementation
//=============================================================================================================

TimeSeriesData::TimeSeriesData()
: BaseData()
, tmin(0.0)
, tmax(0.0)
{
}

TimeSeriesData::TimeSeriesData(const MatrixXd& data, 
                               double sfreq, 
                               const std::vector<std::string>& ch_names,
                               double tmin)
: BaseData()
, data(data)
, tmin(tmin)
{
    this->sfreq = sfreq;
    this->ch_names = ch_names;
    
    // Generate time vector
    int n_times = data.cols();
    times = VectorXd::LinSpaced(n_times, tmin, tmin + (n_times - 1) / sfreq);
    tmax = times(times.size() - 1);
}

BaseData::SPtr TimeSeriesData::copy() const
{
    auto copied = std::make_shared<TimeSeriesData>();
    copied->data = this->data;
    copied->sfreq = this->sfreq;
    copied->ch_names = this->ch_names;
    copied->times = this->times;
    copied->tmin = this->tmin;
    copied->tmax = this->tmax;
    return copied;
}

void TimeSeriesData::pick_channels(const std::vector<std::string>& ch_names_to_pick)
{
    std::vector<int> picks;
    for (const auto& name : ch_names_to_pick) {
        auto it = std::find(ch_names.begin(), ch_names.end(), name);
        if (it != ch_names.end()) {
            picks.push_back(std::distance(ch_names.begin(), it));
        }
    }
    
    if (picks.empty()) {
        return;
    }
    
    // Create new data matrix with selected channels
    MatrixXd new_data(picks.size(), data.cols());
    std::vector<std::string> new_ch_names;
    
    for (size_t i = 0; i < picks.size(); ++i) {
        new_data.row(i) = data.row(picks[i]);
        new_ch_names.push_back(ch_names[picks[i]]);
    }
    
    data = new_data;
    ch_names = new_ch_names;
}

bool TimeSeriesData::isEmpty() const
{
    return data.size() == 0 || ch_names.empty();
}

MatrixXd TimeSeriesData::get_data(double tmin_req, double tmax_req) const
{
    if (tmin_req < 0 && tmax_req < 0) {
        return data;
    }
    
    // Find time indices
    int start_idx = 0;
    int end_idx = times.size() - 1;
    
    if (tmin_req >= 0) {
        for (int i = 0; i < times.size(); ++i) {
            if (times(i) >= tmin_req) {
                start_idx = i;
                break;
            }
        }
    }
    
    if (tmax_req >= 0) {
        for (int i = times.size() - 1; i >= 0; --i) {
            if (times(i) <= tmax_req) {
                end_idx = i;
                break;
            }
        }
    }
    
    return data.block(0, start_idx, data.rows(), end_idx - start_idx + 1);
}

void TimeSeriesData::crop(double tmin_new, double tmax_new)
{
    // Find time indices
    int start_idx = 0;
    int end_idx = times.size() - 1;
    
    for (int i = 0; i < times.size(); ++i) {
        if (times(i) >= tmin_new) {
            start_idx = i;
            break;
        }
    }
    
    for (int i = times.size() - 1; i >= 0; --i) {
        if (times(i) <= tmax_new) {
            end_idx = i;
            break;
        }
    }
    
    // Crop data and times
    data = data.block(0, start_idx, data.rows(), end_idx - start_idx + 1);
    times = times.segment(start_idx, end_idx - start_idx + 1);
    tmin = times(0);
    tmax = times(times.size() - 1);
}
//=============================================================================================================
// FrequencyData Implementation
//=============================================================================================================

FrequencyData::FrequencyData()
: BaseData()
{
}

FrequencyData::FrequencyData(const std::vector<MatrixXcd>& data,
                             const VectorXd& freqs,
                             const VectorXd& times,
                             double sfreq,
                             const std::vector<std::string>& ch_names)
: BaseData()
, data(data)
, freqs(freqs)
{
    this->sfreq = sfreq;
    this->ch_names = ch_names;
    this->times = times;
}

BaseData::SPtr FrequencyData::copy() const
{
    auto copied = std::make_shared<FrequencyData>();
    copied->data = this->data;
    copied->freqs = this->freqs;
    copied->sfreq = this->sfreq;
    copied->ch_names = this->ch_names;
    copied->times = this->times;
    return copied;
}

void FrequencyData::pick_channels(const std::vector<std::string>& ch_names_to_pick)
{
    std::vector<int> picks;
    for (const auto& name : ch_names_to_pick) {
        auto it = std::find(ch_names.begin(), ch_names.end(), name);
        if (it != ch_names.end()) {
            picks.push_back(std::distance(ch_names.begin(), it));
        }
    }
    
    if (picks.empty()) {
        return;
    }
    
    // Create new data with selected channels
    std::vector<MatrixXcd> new_data;
    for (const auto& freq_data : data) {
        MatrixXcd new_freq_data(picks.size(), freq_data.cols());
        for (size_t i = 0; i < picks.size(); ++i) {
            new_freq_data.row(i) = freq_data.row(picks[i]);
        }
        new_data.push_back(new_freq_data);
    }
    
    std::vector<std::string> new_ch_names;
    for (int pick : picks) {
        new_ch_names.push_back(ch_names[pick]);
    }
    
    data = new_data;
    ch_names = new_ch_names;
}

bool FrequencyData::isEmpty() const
{
    return data.empty() || ch_names.empty();
}

std::vector<MatrixXd> FrequencyData::get_power() const
{
    std::vector<MatrixXd> power_data;
    for (const auto& freq_data : data) {
        power_data.push_back(freq_data.cwiseAbs2());
    }
    return power_data;
}

std::vector<MatrixXd> FrequencyData::get_phase() const
{
    std::vector<MatrixXd> phase_data;
    for (const auto& freq_data : data) {
        MatrixXd phase = MatrixXd::Zero(freq_data.rows(), freq_data.cols());
        for (int i = 0; i < freq_data.rows(); ++i) {
            for (int j = 0; j < freq_data.cols(); ++j) {
                phase(i, j) = std::arg(freq_data(i, j));
            }
        }
        phase_data.push_back(phase);
    }
    return phase_data;
}

//=============================================================================================================
// Info Implementation
//=============================================================================================================

Info::Info()
: sfreq(1000.0)
, highpass(0.0)
, lowpass(0.0)
{
}

Info::Info(const Info& other)
: sfreq(other.sfreq)
, ch_names(other.ch_names)
, ch_types(other.ch_types)
, bads(other.bads)
, highpass(other.highpass)
, lowpass(other.lowpass)
, custom_ref_applied(other.custom_ref_applied)
, description(other.description)
, experimenter(other.experimenter)
, dev_head_t(other.dev_head_t)
, dig(other.dig)
, projs(other.projs)
, comps(other.comps)
{
}

Info& Info::operator=(const Info& other)
{
    if (this != &other) {
        sfreq = other.sfreq;
        ch_names = other.ch_names;
        ch_types = other.ch_types;
        bads = other.bads;
        highpass = other.highpass;
        lowpass = other.lowpass;
        custom_ref_applied = other.custom_ref_applied;
        description = other.description;
        experimenter = other.experimenter;
        dev_head_t = other.dev_head_t;
        dig = other.dig;
        projs = other.projs;
        comps = other.comps;
    }
    return *this;
}

Info::SPtr Info::copy() const
{
    return std::make_shared<Info>(*this);
}

Info Info::pick_channels(const VectorXi& picks) const
{
    Info new_info = *this;
    
    std::vector<std::string> new_ch_names;
    std::vector<std::string> new_ch_types;
    
    for (int i = 0; i < picks.size(); ++i) {
        if (picks(i) >= 0 && picks(i) < static_cast<int>(ch_names.size())) {
            new_ch_names.push_back(ch_names[picks(i)]);
            if (picks(i) < static_cast<int>(ch_types.size())) {
                new_ch_types.push_back(ch_types[picks(i)]);
            }
        }
    }
    
    new_info.ch_names = new_ch_names;
    new_info.ch_types = new_ch_types;
    
    return new_info;
}

Info Info::pick_channels(const std::vector<std::string>& ch_names_to_pick) const
{
    VectorXi picks(ch_names_to_pick.size());
    int valid_picks = 0;
    
    for (size_t i = 0; i < ch_names_to_pick.size(); ++i) {
        auto it = std::find(ch_names.begin(), ch_names.end(), ch_names_to_pick[i]);
        if (it != ch_names.end()) {
            picks(valid_picks++) = std::distance(ch_names.begin(), it);
        }
    }
    
    picks.conservativeResize(valid_picks);
    return pick_channels(picks);
}

bool Info::operator==(const Info& other) const
{
    return sfreq == other.sfreq &&
           ch_names == other.ch_names &&
           ch_types == other.ch_types &&
           bads == other.bads &&
           highpass == other.highpass &&
           lowpass == other.lowpass &&
           description == other.description &&
           experimenter == other.experimenter;
}
//=============================================================================================================
// RawData Implementation
//=============================================================================================================

RawData::RawData()
: TimeSeriesData()
{
}

RawData::RawData(const MatrixXd& data, const Info& info, double tmin)
: TimeSeriesData(data, info.sfreq, info.ch_names, tmin)
, info(info)
{
}

BaseData::SPtr RawData::copy() const
{
    auto copied = std::make_shared<RawData>();
    copied->data = this->data;
    copied->sfreq = this->sfreq;
    copied->ch_names = this->ch_names;
    copied->times = this->times;
    copied->tmin = this->tmin;
    copied->tmax = this->tmax;
    copied->info = this->info;
    copied->annotations = this->annotations;
    return copied;
}

void RawData::filter(double l_freq, double h_freq)
{
    // Placeholder for filter implementation
    // This would typically involve FFT-based filtering
    // For now, just update the info
    if (l_freq > 0) {
        info.highpass = l_freq;
    }
    if (h_freq > 0 && h_freq < sfreq / 2) {
        info.lowpass = h_freq;
    }
}

void RawData::resample(double new_sfreq)
{
    if (new_sfreq == sfreq) {
        return;
    }
    
    // Placeholder for resampling implementation
    // This would typically involve interpolation
    double ratio = new_sfreq / sfreq;
    int new_n_times = static_cast<int>(data.cols() * ratio);
    
    // Simple linear interpolation (placeholder)
    MatrixXd new_data = MatrixXd::Zero(data.rows(), new_n_times);
    VectorXd new_times = VectorXd::LinSpaced(new_n_times, tmin, tmax);
    
    // Update sampling frequency and times
    sfreq = new_sfreq;
    info.sfreq = new_sfreq;
    times = new_times;
    data = new_data;
}

//=============================================================================================================
// EpochData Implementation
//=============================================================================================================

EpochData::EpochData()
: BaseData()
, tmin(-0.2)
{
}

EpochData::EpochData(const std::vector<MatrixXd>& epochs,
                     const Info& info,
                     const MatrixXi& events,
                     const std::map<std::string, int>& event_id,
                     double tmin)
: BaseData()
, epochs(epochs)
, info(info)
, events(events)
, event_id(event_id)
, tmin(tmin)
{
    this->sfreq = info.sfreq;
    this->ch_names = info.ch_names;
    
    if (!epochs.empty()) {
        int n_times = epochs[0].cols();
        times = VectorXd::LinSpaced(n_times, tmin, tmin + (n_times - 1) / sfreq);
        drop_log.resize(epochs.size(), false);
    }
}

BaseData::SPtr EpochData::copy() const
{
    auto copied = std::make_shared<EpochData>();
    copied->epochs = this->epochs;
    copied->info = this->info;
    copied->events = this->events;
    copied->event_id = this->event_id;
    copied->tmin = this->tmin;
    copied->sfreq = this->sfreq;
    copied->ch_names = this->ch_names;
    copied->times = this->times;
    copied->drop_log = this->drop_log;
    return copied;
}

void EpochData::pick_channels(const std::vector<std::string>& ch_names_to_pick)
{
    std::vector<int> picks;
    for (const auto& name : ch_names_to_pick) {
        auto it = std::find(ch_names.begin(), ch_names.end(), name);
        if (it != ch_names.end()) {
            picks.push_back(std::distance(ch_names.begin(), it));
        }
    }
    
    if (picks.empty()) {
        return;
    }
    
    // Update epochs data
    for (auto& epoch : epochs) {
        MatrixXd new_epoch(picks.size(), epoch.cols());
        for (size_t i = 0; i < picks.size(); ++i) {
            new_epoch.row(i) = epoch.row(picks[i]);
        }
        epoch = new_epoch;
    }
    
    // Update channel names
    std::vector<std::string> new_ch_names;
    for (int pick : picks) {
        new_ch_names.push_back(ch_names[pick]);
    }
    ch_names = new_ch_names;
    
    // Update info
    info = info.pick_channels(ch_names_to_pick);
}

bool EpochData::isEmpty() const
{
    return epochs.empty() || ch_names.empty();
}

std::shared_ptr<EvokedData> EpochData::average() const
{
    if (epochs.empty()) {
        return nullptr;
    }
    
    // Calculate average across epochs
    MatrixXd avg_data = MatrixXd::Zero(epochs[0].rows(), epochs[0].cols());
    int valid_epochs = 0;
    
    for (size_t i = 0; i < epochs.size(); ++i) {
        if (!drop_log[i]) {
            avg_data += epochs[i];
            valid_epochs++;
        }
    }
    
    if (valid_epochs > 0) {
        avg_data /= valid_epochs;
    }
    
    return std::make_shared<EvokedData>(avg_data, info, tmin, "Average", valid_epochs);
}

void EpochData::drop_bad()
{
    // Remove epochs marked as bad
    std::vector<MatrixXd> good_epochs;
    MatrixXi good_events(0, events.cols());
    
    for (size_t i = 0; i < epochs.size(); ++i) {
        if (!drop_log[i]) {
            good_epochs.push_back(epochs[i]);
            // Resize and copy event row
            good_events.conservativeResize(good_events.rows() + 1, events.cols());
            good_events.row(good_events.rows() - 1) = events.row(i);
        }
    }
    
    epochs = good_epochs;
    events = good_events;
    drop_log.clear();
    drop_log.resize(epochs.size(), false);
}

std::vector<MatrixXd> EpochData::get_data(const VectorXi& picks) const
{
    if (picks.size() == 0) {
        return epochs;
    }
    
    std::vector<MatrixXd> selected_epochs;
    for (int i = 0; i < picks.size(); ++i) {
        if (picks(i) >= 0 && picks(i) < static_cast<int>(epochs.size())) {
            selected_epochs.push_back(epochs[picks(i)]);
        }
    }
    
    return selected_epochs;
}

//=============================================================================================================
// EvokedData Implementation
//=============================================================================================================

EvokedData::EvokedData()
: TimeSeriesData()
, nave(1)
{
}

EvokedData::EvokedData(const MatrixXd& data,
                       const Info& info,
                       double tmin,
                       const std::string& comment,
                       int nave)
: TimeSeriesData(data, info.sfreq, info.ch_names, tmin)
, info(info)
, comment(comment)
, nave(nave)
{
    nave_per_channel = VectorXd::Constant(data.rows(), nave);
}

BaseData::SPtr EvokedData::copy() const
{
    auto copied = std::make_shared<EvokedData>();
    copied->data = this->data;
    copied->sfreq = this->sfreq;
    copied->ch_names = this->ch_names;
    copied->times = this->times;
    copied->tmin = this->tmin;
    copied->tmax = this->tmax;
    copied->info = this->info;
    copied->comment = this->comment;
    copied->nave = this->nave;
    copied->nave_per_channel = this->nave_per_channel;
    return copied;
}

void EvokedData::apply_baseline(const std::pair<double, double>& baseline)
{
    double baseline_start = baseline.first;
    double baseline_end = baseline.second;
    
    // Find baseline indices
    int start_idx = 0;
    int end_idx = times.size() - 1;
    
    for (int i = 0; i < times.size(); ++i) {
        if (times(i) >= baseline_start) {
            start_idx = i;
            break;
        }
    }
    
    for (int i = times.size() - 1; i >= 0; --i) {
        if (times(i) <= baseline_end) {
            end_idx = i;
            break;
        }
    }
    
    // Calculate baseline mean for each channel
    MatrixXd baseline_data = data.block(0, start_idx, data.rows(), end_idx - start_idx + 1);
    VectorXd baseline_mean = baseline_data.rowwise().mean();
    
    // Subtract baseline from all time points
    for (int i = 0; i < data.cols(); ++i) {
        data.col(i) -= baseline_mean;
    }
}