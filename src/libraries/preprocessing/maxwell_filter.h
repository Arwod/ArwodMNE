//=============================================================================================================
/**
 * @file     maxwell_filter.h
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
 * @brief    Maxwell filtering algorithms for MEG signal space separation (SSS)
 *
 */

#ifndef MAXWELL_FILTER_H
#define MAXWELL_FILTER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "preprocessing_global.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <complex>

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
 * Structure to represent sensor positions and orientations
 */
struct PREPROCESSINGSHARED_EXPORT SensorInfo
{
    Eigen::Vector3d position;      /**< Sensor position in meters */
    Eigen::Vector3d orientation;   /**< Sensor orientation (unit vector) */
    std::string ch_name;           /**< Channel name */
    std::string ch_type;           /**< Channel type (mag, grad1, grad2) */
    
    SensorInfo() : position(Eigen::Vector3d::Zero()), orientation(Eigen::Vector3d::Zero()) {}
    SensorInfo(const Eigen::Vector3d& pos, const Eigen::Vector3d& ori, 
               const std::string& name, const std::string& type)
        : position(pos), orientation(ori), ch_name(name), ch_type(type) {}
};

//=============================================================================================================
/**
 * Structure to represent Maxwell basis functions
 */
struct PREPROCESSINGSHARED_EXPORT MaxwellBasis
{
    Eigen::MatrixXcd internal_basis;    /**< Internal basis functions (n_channels x n_internal) */
    Eigen::MatrixXcd external_basis;    /**< External basis functions (n_channels x n_external) */
    std::vector<int> internal_orders;   /**< Orders of internal basis functions */
    std::vector<int> external_orders;   /**< Orders of external basis functions */
    Eigen::Vector3d origin;             /**< Origin of the coordinate system */
    double int_order;                   /**< Maximum internal order */
    double ext_order;                   /**< Maximum external order */
    
    MaxwellBasis() : origin(Eigen::Vector3d::Zero()), int_order(8), ext_order(3) {}
};

//=============================================================================================================
/**
 * Structure to represent Maxwell filtering results
 */
struct PREPROCESSINGSHARED_EXPORT MaxwellFilterResult
{
    Eigen::MatrixXd filtered_data;      /**< Filtered data (n_channels x n_samples) */
    Eigen::VectorXd internal_moments;   /**< Internal multipole moments */
    Eigen::VectorXd external_moments;   /**< External multipole moments */
    std::vector<int> bad_channels;      /**< Indices of bad channels */
    double goodness_of_fit;             /**< Goodness of fit measure */
    
    MaxwellFilterResult() : goodness_of_fit(0.0) {}
};

//=============================================================================================================
/**
 * Maxwell filtering algorithms for MEG signal space separation (SSS)
 *
 * @brief The MaxwellFilter class provides methods for Maxwell filtering of MEG data
 * using spherical harmonic basis functions for signal space separation.
 */
class PREPROCESSINGSHARED_EXPORT MaxwellFilter
{

public:
    typedef QSharedPointer<MaxwellFilter> SPtr;            /**< Shared pointer type for MaxwellFilter. */
    typedef QSharedPointer<const MaxwellFilter> ConstSPtr; /**< Const shared pointer type for MaxwellFilter. */

    //=========================================================================================================
    /**
     * Constructs a MaxwellFilter object.
     */
    explicit MaxwellFilter();

    //=========================================================================================================
    /**
     * Destructor
     */
    ~MaxwellFilter();

    //=========================================================================================================
    /**
     * Apply Maxwell filtering to MEG data
     *
     * @param[in] data              Input MEG data matrix (channels x samples)
     * @param[in] sensor_info       Sensor position and orientation information
     * @param[in] origin            Origin of the coordinate system (default: head center)
     * @param[in] int_order         Internal expansion order (default: 8)
     * @param[in] ext_order         External expansion order (default: 3)
     * @param[in] coord_frame       Coordinate frame ('head' or 'device', default: 'head')
     * @param[in] regularize        Regularization parameter (default: 0.1)
     * @param[in] ignore_ref        Ignore reference channels (default: true)
     * @param[in] bad_condition     Bad condition threshold (default: 'warning')
     * @param[in] head_pos          Head position transformation matrix (default: identity)
     * @param[in] calibration       Calibration matrix (default: identity)
     * @param[in] cross_talk        Cross-talk matrix (default: identity)
     *
     * @return Maxwell filtering results
     */
    static MaxwellFilterResult maxwell_filter(const Eigen::MatrixXd& data,
                                             const std::vector<SensorInfo>& sensor_info,
                                             const Eigen::Vector3d& origin = Eigen::Vector3d::Zero(),
                                             int int_order = 8,
                                             int ext_order = 3,
                                             const std::string& coord_frame = "head",
                                             double regularize = 0.1,
                                             bool ignore_ref = true,
                                             const std::string& bad_condition = "warning",
                                             const Eigen::Matrix4d& head_pos = Eigen::Matrix4d::Identity(),
                                             const Eigen::MatrixXd& calibration = Eigen::MatrixXd(),
                                             const Eigen::MatrixXd& cross_talk = Eigen::MatrixXd());

    //=========================================================================================================
    /**
     * Compute Maxwell basis functions (spherical harmonics)
     *
     * @param[in] sensor_info       Sensor position and orientation information
     * @param[in] origin            Origin of the coordinate system
     * @param[in] int_order         Internal expansion order
     * @param[in] ext_order         External expansion order
     * @param[in] coord_frame       Coordinate frame ('head' or 'device')
     * @param[in] regularize        Regularization parameter
     *
     * @return Maxwell basis functions
     */
    static MaxwellBasis compute_maxwell_basis(const std::vector<SensorInfo>& sensor_info,
                                             const Eigen::Vector3d& origin = Eigen::Vector3d::Zero(),
                                             int int_order = 8,
                                             int ext_order = 3,
                                             const std::string& coord_frame = "head",
                                             double regularize = 0.1);

    //=========================================================================================================
    /**
     * Find bad channels using Maxwell filtering
     *
     * @param[in] data              Input MEG data matrix (channels x samples)
     * @param[in] sensor_info       Sensor position and orientation information
     * @param[in] origin            Origin of the coordinate system
     * @param[in] int_order         Internal expansion order (default: 8)
     * @param[in] ext_order         External expansion order (default: 3)
     * @param[in] coord_frame       Coordinate frame ('head' or 'device', default: 'head')
     * @param[in] regularize        Regularization parameter (default: 0.1)
     * @param[in] limit             Limit for bad channel detection (default: 7.0)
     * @param[in] duration          Duration for analysis in seconds (default: 5.0)
     * @param[in] min_count         Minimum count for bad channel detection (default: 5)
     *
     * @return Vector of bad channel indices
     */
    static std::vector<int> find_bad_channels_maxwell(const Eigen::MatrixXd& data,
                                                     const std::vector<SensorInfo>& sensor_info,
                                                     const Eigen::Vector3d& origin = Eigen::Vector3d::Zero(),
                                                     int int_order = 8,
                                                     int ext_order = 3,
                                                     const std::string& coord_frame = "head",
                                                     double regularize = 0.1,
                                                     double limit = 7.0,
                                                     double duration = 5.0,
                                                     int min_count = 5);

private:
    //=========================================================================================================
    /**
     * Compute spherical harmonic basis functions
     *
     * @param[in] positions         Sensor positions
     * @param[in] orientations      Sensor orientations
     * @param[in] origin            Origin of coordinate system
     * @param[in] max_order         Maximum harmonic order
     * @param[in] internal          Whether to compute internal (true) or external (false) basis
     *
     * @return Spherical harmonic basis matrix
     */
    static Eigen::MatrixXcd compute_spherical_harmonics(const std::vector<Eigen::Vector3d>& positions,
                                                       const std::vector<Eigen::Vector3d>& orientations,
                                                       const Eigen::Vector3d& origin,
                                                       int max_order,
                                                       bool internal = true);

    //=========================================================================================================
    /**
     * Compute associated Legendre polynomials
     *
     * @param[in] l                 Degree
     * @param[in] m                 Order
     * @param[in] x                 Argument (cos(theta))
     *
     * @return Associated Legendre polynomial value
     */
    static double associated_legendre(int l, int m, double x);

    //=========================================================================================================
    /**
     * Compute spherical harmonic function
     *
     * @param[in] l                 Degree
     * @param[in] m                 Order
     * @param[in] theta             Polar angle
     * @param[in] phi               Azimuthal angle
     *
     * @return Complex spherical harmonic value
     */
    static std::complex<double> spherical_harmonic(int l, int m, double theta, double phi);

    //=========================================================================================================
    /**
     * Convert Cartesian coordinates to spherical coordinates
     *
     * @param[in] position          Cartesian position vector
     * @param[out] r                Radial distance
     * @param[out] theta            Polar angle
     * @param[out] phi              Azimuthal angle
     */
    static void cartesian_to_spherical(const Eigen::Vector3d& position,
                                      double& r, double& theta, double& phi);

    //=========================================================================================================
    /**
     * Apply regularization to the basis matrix
     *
     * @param[in,out] basis         Basis matrix to regularize
     * @param[in] regularize        Regularization parameter
     */
    static void apply_regularization(Eigen::MatrixXcd& basis, double regularize);

    //=========================================================================================================
    /**
     * Compute goodness of fit for Maxwell filtering
     *
     * @param[in] original_data     Original data
     * @param[in] filtered_data     Filtered data
     *
     * @return Goodness of fit value (0-1)
     */
    static double compute_goodness_of_fit(const Eigen::MatrixXd& original_data,
                                        const Eigen::MatrixXd& filtered_data);

    //=========================================================================================================
    /**
     * Detect bad channels based on reconstruction error
     *
     * @param[in] reconstruction_errors  Reconstruction errors for each channel
     * @param[in] limit                  Threshold for bad channel detection
     *
     * @return Vector of bad channel indices
     */
    static std::vector<int> detect_bad_channels(const Eigen::VectorXd& reconstruction_errors,
                                               double limit);

    //=========================================================================================================
    /**
     * Compute factorial
     *
     * @param[in] n                 Input number
     *
     * @return Factorial of n
     */
    static double factorial(int n);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE PREPROCESSINGLIB

#endif // MAXWELL_FILTER_H