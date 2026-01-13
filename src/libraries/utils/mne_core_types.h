//=============================================================================================================
/**
 * @file     mne_core_types.h
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
 * @brief     Core data types for MNE-CPP migration project
 */

#ifndef MNE_CORE_TYPES_H
#define MNE_CORE_TYPES_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "utils_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <any>

//=============================================================================================================
// DEFINE NAMESPACE MNELIB
//=============================================================================================================

namespace MNELIB
{

//=============================================================================================================
// TYPE ALIASES
//=============================================================================================================

// Basic matrix and vector types
using MatrixXd = Eigen::MatrixXd;
using MatrixXi = Eigen::MatrixXi;
using VectorXd = Eigen::VectorXd;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXcd = Eigen::VectorXcd;
using VectorXi = Eigen::VectorXi;
using RowVectorXd = Eigen::RowVectorXd;
using RowVectorXf = Eigen::RowVectorXf;

//=============================================================================================================
// FORWARD DECLARATIONS
//=============================================================================================================

class BaseData;
class TimeSeriesData;
class FrequencyData;
class Info;
class RawData;
class EpochData;
class EvokedData;
class SourceEstimate;

//=============================================================================================================
/**
 * Base class for all MNE data types
 *
 * @brief Base data class providing common functionality
 */
class UTILSSHARED_EXPORT BaseData
{
public:
    typedef std::shared_ptr<BaseData> SPtr;
    typedef std::shared_ptr<const BaseData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    BaseData();

    //=========================================================================================================
    /**
     * Virtual destructor
     */
    virtual ~BaseData() = default;

    //=========================================================================================================
    /**
     * Copy constructor
     */
    BaseData(const BaseData& other);

    //=========================================================================================================
    /**
     * Assignment operator
     */
    BaseData& operator=(const BaseData& other);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    virtual BaseData::SPtr copy() const = 0;

    //=========================================================================================================
    /**
     * Pick channels by name
     *
     * @param[in] ch_names  List of channel names to pick
     */
    virtual void pick_channels(const std::vector<std::string>& ch_names) = 0;

    //=========================================================================================================
    /**
     * Check if data is empty
     *
     * @return true if empty, false otherwise
     */
    virtual bool isEmpty() const = 0;

public:
    std::vector<std::string> ch_names;      /**< Channel names */
    VectorXd times;                         /**< Time vector */
    double sfreq;                           /**< Sampling frequency */
};

//=============================================================================================================
/**
 * Time series data structure
 *
 * @brief Container for time-domain data
 */
class UTILSSHARED_EXPORT TimeSeriesData : public BaseData
{
public:
    typedef std::shared_ptr<TimeSeriesData> SPtr;
    typedef std::shared_ptr<const TimeSeriesData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    TimeSeriesData();

    //=========================================================================================================
    /**
     * Constructor with data
     *
     * @param[in] data      Data matrix (n_channels x n_times)
     * @param[in] sfreq     Sampling frequency
     * @param[in] ch_names  Channel names
     * @param[in] tmin      Start time
     */
    TimeSeriesData(const MatrixXd& data, 
                   double sfreq, 
                   const std::vector<std::string>& ch_names,
                   double tmin = 0.0);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    BaseData::SPtr copy() const override;

    //=========================================================================================================
    /**
     * Pick channels by name
     *
     * @param[in] ch_names  List of channel names to pick
     */
    void pick_channels(const std::vector<std::string>& ch_names) override;

    //=========================================================================================================
    /**
     * Check if data is empty
     *
     * @return true if empty, false otherwise
     */
    bool isEmpty() const override;

    //=========================================================================================================
    /**
     * Get data for specific time range
     *
     * @param[in] tmin  Start time
     * @param[in] tmax  End time
     * @return Data matrix for the specified time range
     */
    MatrixXd get_data(double tmin = -1, double tmax = -1) const;

    //=========================================================================================================
    /**
     * Crop data to specific time range
     *
     * @param[in] tmin  Start time
     * @param[in] tmax  End time
     */
    void crop(double tmin, double tmax);

public:
    MatrixXd data;                          /**< Data matrix (n_channels x n_times) */
    double tmin;                            /**< Start time */
    double tmax;                            /**< End time */
};

//=============================================================================================================
/**
 * Frequency domain data structure
 *
 * @brief Container for frequency-domain data
 */
class UTILSSHARED_EXPORT FrequencyData : public BaseData
{
public:
    typedef std::shared_ptr<FrequencyData> SPtr;
    typedef std::shared_ptr<const FrequencyData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    FrequencyData();

    //=========================================================================================================
    /**
     * Constructor with data
     *
     * @param[in] data      Complex data matrix (n_channels x n_freqs x n_times)
     * @param[in] freqs     Frequency vector
     * @param[in] times     Time vector
     * @param[in] sfreq     Sampling frequency
     * @param[in] ch_names  Channel names
     */
    FrequencyData(const std::vector<MatrixXcd>& data,
                  const VectorXd& freqs,
                  const VectorXd& times,
                  double sfreq,
                  const std::vector<std::string>& ch_names);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    BaseData::SPtr copy() const override;

    //=========================================================================================================
    /**
     * Pick channels by name
     *
     * @param[in] ch_names  List of channel names to pick
     */
    void pick_channels(const std::vector<std::string>& ch_names) override;

    //=========================================================================================================
    /**
     * Check if data is empty
     *
     * @return true if empty, false otherwise
     */
    bool isEmpty() const override;

    //=========================================================================================================
    /**
     * Get power spectrum
     *
     * @return Power spectrum matrix (n_channels x n_freqs x n_times)
     */
    std::vector<MatrixXd> get_power() const;

    //=========================================================================================================
    /**
     * Get phase spectrum
     *
     * @return Phase spectrum matrix (n_channels x n_freqs x n_times)
     */
    std::vector<MatrixXd> get_phase() const;

public:
    std::vector<MatrixXcd> data;            /**< Complex data (n_freqs matrices of n_channels x n_times) */
    VectorXd freqs;                         /**< Frequency vector */
};

//=============================================================================================================
/**
 * Information structure compatible with MNE-Python
 *
 * @brief Measurement information structure
 */
class UTILSSHARED_EXPORT Info
{
public:
    typedef std::shared_ptr<Info> SPtr;
    typedef std::shared_ptr<const Info> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    Info();

    //=========================================================================================================
    /**
     * Copy constructor
     */
    Info(const Info& other);

    //=========================================================================================================
    /**
     * Assignment operator
     */
    Info& operator=(const Info& other);

    //=========================================================================================================
    /**
     * Create a deep copy
     *
     * @return Shared pointer to copied info
     */
    Info::SPtr copy() const;

    //=========================================================================================================
    /**
     * Pick channels by indices
     *
     * @param[in] picks  Channel indices to pick
     * @return New Info object with selected channels
     */
    Info pick_channels(const VectorXi& picks) const;

    //=========================================================================================================
    /**
     * Pick channels by names
     *
     * @param[in] ch_names  Channel names to pick
     * @return New Info object with selected channels
     */
    Info pick_channels(const std::vector<std::string>& ch_names) const;

    //=========================================================================================================
    /**
     * Check equality with another Info object
     *
     * @param[in] other  Other Info object
     * @return true if equal, false otherwise
     */
    bool operator==(const Info& other) const;

public:
    double sfreq;                                           /**< Sampling frequency */
    std::vector<std::string> ch_names;                     /**< Channel names */
    std::vector<std::string> ch_types;                     /**< Channel types */
    std::vector<std::string> bads;                         /**< Bad channel names */
    double highpass;                                        /**< Highpass filter frequency */
    double lowpass;                                         /**< Lowpass filter frequency */
    std::map<std::string, std::any> custom_ref_applied;   /**< Custom reference information */
    std::string description;                                /**< Description */
    std::string experimenter;                               /**< Experimenter name */
    
    // Coordinate transformation and digitization info
    // (simplified for now, can be expanded later)
    std::map<std::string, std::any> dev_head_t;           /**< Device to head transformation */
    std::vector<std::map<std::string, std::any>> dig;     /**< Digitization points */
    
    // Projection and compensation info
    std::vector<std::map<std::string, std::any>> projs;   /**< SSP projectors */
    std::vector<std::map<std::string, std::any>> comps;   /**< CTF compensators */
};

//=============================================================================================================
/**
 * Raw data structure
 *
 * @brief Container for continuous raw data
 */
class UTILSSHARED_EXPORT RawData : public TimeSeriesData
{
public:
    typedef std::shared_ptr<RawData> SPtr;
    typedef std::shared_ptr<const RawData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    RawData();

    //=========================================================================================================
    /**
     * Constructor with data and info
     *
     * @param[in] data  Data matrix (n_channels x n_times)
     * @param[in] info  Measurement information
     * @param[in] tmin  Start time
     */
    RawData(const MatrixXd& data, const Info& info, double tmin = 0.0);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    BaseData::SPtr copy() const override;

    //=========================================================================================================
    /**
     * Apply filter to the data
     *
     * @param[in] l_freq  Low frequency cutoff
     * @param[in] h_freq  High frequency cutoff
     */
    void filter(double l_freq, double h_freq);

    //=========================================================================================================
    /**
     * Resample the data
     *
     * @param[in] new_sfreq  New sampling frequency
     */
    void resample(double new_sfreq);

public:
    Info info;                                              /**< Measurement information */
    std::vector<std::map<std::string, std::any>> annotations; /**< Data annotations */
};

//=============================================================================================================
/**
 * Epoch data structure
 *
 * @brief Container for epoched data
 */
class UTILSSHARED_EXPORT EpochData : public BaseData
{
public:
    typedef std::shared_ptr<EpochData> SPtr;
    typedef std::shared_ptr<const EpochData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    EpochData();

    //=========================================================================================================
    /**
     * Constructor with data and info
     *
     * @param[in] epochs    Vector of epoch data matrices
     * @param[in] info      Measurement information
     * @param[in] events    Event information
     * @param[in] event_id  Event ID mapping
     * @param[in] tmin      Start time relative to event
     */
    EpochData(const std::vector<MatrixXd>& epochs,
              const Info& info,
              const MatrixXi& events,
              const std::map<std::string, int>& event_id,
              double tmin = -0.2);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    BaseData::SPtr copy() const override;

    //=========================================================================================================
    /**
     * Pick channels by name
     *
     * @param[in] ch_names  List of channel names to pick
     */
    void pick_channels(const std::vector<std::string>& ch_names) override;

    //=========================================================================================================
    /**
     * Check if data is empty
     *
     * @return true if empty, false otherwise
     */
    bool isEmpty() const override;

    //=========================================================================================================
    /**
     * Average epochs to create evoked response
     *
     * @return Evoked data
     */
    std::shared_ptr<class EvokedData> average() const;

    //=========================================================================================================
    /**
     * Drop bad epochs
     */
    void drop_bad();

    //=========================================================================================================
    /**
     * Get data for specific epochs
     *
     * @param[in] picks  Epoch indices to pick
     * @return Selected epoch data
     */
    std::vector<MatrixXd> get_data(const VectorXi& picks = VectorXi()) const;

public:
    std::vector<MatrixXd> epochs;                           /**< Epoch data matrices */
    Info info;                                              /**< Measurement information */
    MatrixXi events;                                        /**< Event information */
    std::map<std::string, int> event_id;                   /**< Event ID mapping */
    double tmin;                                            /**< Start time relative to event */
    std::vector<bool> drop_log;                            /**< Drop log for bad epochs */
};

//=============================================================================================================
/**
 * Evoked data structure
 *
 * @brief Container for averaged evoked responses
 */
class UTILSSHARED_EXPORT EvokedData : public TimeSeriesData
{
public:
    typedef std::shared_ptr<EvokedData> SPtr;
    typedef std::shared_ptr<const EvokedData> ConstSPtr;

    //=========================================================================================================
    /**
     * Default constructor
     */
    EvokedData();

    //=========================================================================================================
    /**
     * Constructor with data and info
     *
     * @param[in] data     Averaged data matrix (n_channels x n_times)
     * @param[in] info     Measurement information
     * @param[in] tmin     Start time
     * @param[in] comment  Comment describing the evoked response
     * @param[in] nave     Number of averaged epochs
     */
    EvokedData(const MatrixXd& data,
               const Info& info,
               double tmin,
               const std::string& comment = "",
               int nave = 1);

    //=========================================================================================================
    /**
     * Create a deep copy of the data
     *
     * @return Shared pointer to copied data
     */
    BaseData::SPtr copy() const override;

    //=========================================================================================================
    /**
     * Apply baseline correction
     *
     * @param[in] baseline  Baseline time range [tmin, tmax]
     */
    void apply_baseline(const std::pair<double, double>& baseline);

public:
    Info info;                                              /**< Measurement information */
    std::string comment;                                    /**< Comment */
    int nave;                                               /**< Number of averaged epochs */
    VectorXd nave_per_channel;                             /**< Number of epochs per channel */
};

} // namespace MNELIB

#endif // MNE_CORE_TYPES_H