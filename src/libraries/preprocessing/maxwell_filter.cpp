//=============================================================================================================
/**
 * @file     maxwell_filter.cpp
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
 * @brief    Implementation of Maxwell filtering algorithms
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "maxwell_filter.h"
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
using namespace std;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

MaxwellFilter::MaxwellFilter()
{
}

//=============================================================================================================

MaxwellFilter::~MaxwellFilter()
{
}

//=============================================================================================================

MaxwellFilterResult MaxwellFilter::maxwell_filter(const MatrixXd& data,
                                                 const std::vector<SensorInfo>& sensor_info,
                                                 const Vector3d& origin,
                                                 int int_order,
                                                 int ext_order,
                                                 const std::string& coord_frame,
                                                 double regularize,
                                                 bool ignore_ref,
                                                 const std::string& bad_condition,
                                                 const Matrix4d& head_pos,
                                                 const MatrixXd& calibration,
                                                 const MatrixXd& cross_talk)
{
    Q_UNUSED(coord_frame)
    Q_UNUSED(ignore_ref)
    Q_UNUSED(bad_condition)
    Q_UNUSED(head_pos)
    Q_UNUSED(calibration)
    Q_UNUSED(cross_talk)
    
    MaxwellFilterResult result;
    
    if (data.rows() == 0 || data.cols() == 0 || sensor_info.empty()) {
        qWarning() << "MaxwellFilter::maxwell_filter - Invalid input data";
        return result;
    }
    
    const int n_channels = data.rows();
    const int n_samples = data.cols();
    
    if (static_cast<int>(sensor_info.size()) != n_channels) {
        qWarning() << "MaxwellFilter::maxwell_filter - Sensor info size mismatch";
        return result;
    }
    
    // Compute Maxwell basis functions
    MaxwellBasis basis = compute_maxwell_basis(sensor_info, origin, int_order, ext_order, "head", regularize);
    
    if (basis.internal_basis.rows() == 0 || basis.external_basis.rows() == 0) {
        qWarning() << "MaxwellFilter::maxwell_filter - Failed to compute basis functions";
        return result;
    }
    
    // Combine internal and external basis
    const int n_internal = basis.internal_basis.cols();
    const int n_external = basis.external_basis.cols();
    const int n_total = n_internal + n_external;
    
    MatrixXcd combined_basis(n_channels, n_total);
    combined_basis.leftCols(n_internal) = basis.internal_basis;
    combined_basis.rightCols(n_external) = basis.external_basis;
    
    // Convert data to complex for computation
    MatrixXcd data_complex = data.cast<complex<double>>();
    
    // Solve for multipole moments using least squares
    // moments = (basis^H * basis + regularization)^-1 * basis^H * data
    MatrixXcd basis_H = combined_basis.adjoint();
    MatrixXcd gram_matrix = basis_H * combined_basis;
    
    // Add regularization
    MatrixXcd regularization = regularize * MatrixXcd::Identity(n_total, n_total);
    gram_matrix += regularization;
    
    // Solve for moments
    MatrixXcd moments_complex = gram_matrix.ldlt().solve(basis_H * data_complex);
    
    // Reconstruct filtered data
    MatrixXcd filtered_complex = combined_basis * moments_complex;
    result.filtered_data = filtered_complex.real();
    
    // Extract internal and external moments
    result.internal_moments = moments_complex.topRows(n_internal).real().rowwise().norm();
    result.external_moments = moments_complex.bottomRows(n_external).real().rowwise().norm();
    
    // Compute goodness of fit
    result.goodness_of_fit = compute_goodness_of_fit(data, result.filtered_data);
    
    // Find bad channels based on reconstruction error
    VectorXd reconstruction_errors(n_channels);
    for (int ch = 0; ch < n_channels; ++ch) {
        double error = (data.row(ch) - result.filtered_data.row(ch)).norm();
        double original_norm = data.row(ch).norm();
        reconstruction_errors(ch) = (original_norm > 0) ? error / original_norm : 0.0;
    }
    
    result.bad_channels = detect_bad_channels(reconstruction_errors, 0.2); // 20% error threshold
    
    qDebug() << "Maxwell filtering completed:"
             << "Internal moments:" << result.internal_moments.size()
             << "External moments:" << result.external_moments.size()
             << "Bad channels:" << result.bad_channels.size()
             << "Goodness of fit:" << result.goodness_of_fit;
    
    return result;
}

//=============================================================================================================

MaxwellBasis MaxwellFilter::compute_maxwell_basis(const std::vector<SensorInfo>& sensor_info,
                                                 const Vector3d& origin,
                                                 int int_order,
                                                 int ext_order,
                                                 const std::string& coord_frame,
                                                 double regularize)
{
    Q_UNUSED(coord_frame)
    
    MaxwellBasis basis;
    
    if (sensor_info.empty()) {
        qWarning() << "MaxwellFilter::compute_maxwell_basis - Empty sensor info";
        return basis;
    }
    
    const int n_channels = sensor_info.size();
    
    // Extract positions and orientations
    std::vector<Vector3d> positions, orientations;
    for (const SensorInfo& info : sensor_info) {
        positions.push_back(info.position);
        orientations.push_back(info.orientation);
    }
    
    // Compute internal basis (sources inside the sphere)
    basis.internal_basis = compute_spherical_harmonics(positions, orientations, origin, int_order, true);
    
    // Compute external basis (sources outside the sphere)
    basis.external_basis = compute_spherical_harmonics(positions, orientations, origin, ext_order, false);
    
    // Apply regularization
    apply_regularization(basis.internal_basis, regularize);
    apply_regularization(basis.external_basis, regularize);
    
    // Set metadata
    basis.origin = origin;
    basis.int_order = int_order;
    basis.ext_order = ext_order;
    
    // Generate order information
    for (int l = 1; l <= int_order; ++l) {
        for (int m = -l; m <= l; ++m) {
            basis.internal_orders.push_back(l);
        }
    }
    
    for (int l = 1; l <= ext_order; ++l) {
        for (int m = -l; m <= l; ++m) {
            basis.external_orders.push_back(l);
        }
    }
    
    qDebug() << "Maxwell basis computed:"
             << "Internal basis:" << basis.internal_basis.rows() << "x" << basis.internal_basis.cols()
             << "External basis:" << basis.external_basis.rows() << "x" << basis.external_basis.cols();
    
    return basis;
}

//=============================================================================================================

std::vector<int> MaxwellFilter::find_bad_channels_maxwell(const MatrixXd& data,
                                                        const std::vector<SensorInfo>& sensor_info,
                                                        const Vector3d& origin,
                                                        int int_order,
                                                        int ext_order,
                                                        const std::string& coord_frame,
                                                        double regularize,
                                                        double limit,
                                                        double duration,
                                                        int min_count)
{
    Q_UNUSED(duration)
    Q_UNUSED(min_count)
    
    std::vector<int> bad_channels;
    
    if (data.rows() == 0 || data.cols() == 0 || sensor_info.empty()) {
        qWarning() << "MaxwellFilter::find_bad_channels_maxwell - Invalid input data";
        return bad_channels;
    }
    
    // Apply Maxwell filtering
    MaxwellFilterResult result = maxwell_filter(data, sensor_info, origin, int_order, ext_order,
                                               coord_frame, regularize);
    
    if (result.filtered_data.rows() == 0) {
        qWarning() << "MaxwellFilter::find_bad_channels_maxwell - Maxwell filtering failed";
        return bad_channels;
    }
    
    const int n_channels = data.rows();
    
    // Compute reconstruction errors for each channel
    VectorXd reconstruction_errors(n_channels);
    for (int ch = 0; ch < n_channels; ++ch) {
        double error = (data.row(ch) - result.filtered_data.row(ch)).norm();
        double original_norm = data.row(ch).norm();
        reconstruction_errors(ch) = (original_norm > 0) ? error / original_norm : 0.0;
    }
    
    // Find channels with high reconstruction error
    bad_channels = detect_bad_channels(reconstruction_errors, limit);
    
    qDebug() << "Found" << bad_channels.size() << "bad channels using Maxwell filtering";
    
    return bad_channels;
}

//=============================================================================================================

MatrixXcd MaxwellFilter::compute_spherical_harmonics(const std::vector<Vector3d>& positions,
                                                    const std::vector<Vector3d>& orientations,
                                                    const Vector3d& origin,
                                                    int max_order,
                                                    bool internal)
{
    const int n_channels = positions.size();
    
    // Calculate total number of basis functions
    int n_basis = 0;
    for (int l = 1; l <= max_order; ++l) {
        n_basis += 2 * l + 1; // -l <= m <= l
    }
    
    MatrixXcd basis = MatrixXcd::Zero(n_channels, n_basis);
    
    int basis_idx = 0;
    for (int l = 1; l <= max_order; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int ch = 0; ch < n_channels; ++ch) {
                Vector3d rel_pos = positions[ch] - origin;
                Vector3d orientation = orientations[ch];
                
                double r, theta, phi;
                cartesian_to_spherical(rel_pos, r, theta, phi);
                
                if (r < 1e-10) {
                    continue; // Skip sensors at origin
                }
                
                // Compute spherical harmonic
                complex<double> Y_lm = spherical_harmonic(l, m, theta, phi);
                
                // Compute radial dependence
                complex<double> radial_term;
                if (internal) {
                    // Internal basis: r^l
                    radial_term = pow(r, l);
                } else {
                    // External basis: r^(-l-1)
                    radial_term = pow(r, -l - 1);
                }
                
                // Compute gradient for magnetometer/gradiometer response
                // This is a simplified version - full implementation would include
                // proper vector spherical harmonics
                Vector3d grad_direction = rel_pos.normalized();
                double dot_product = orientation.dot(grad_direction);
                
                basis(ch, basis_idx) = Y_lm * radial_term * dot_product;
            }
            basis_idx++;
        }
    }
    
    return basis;
}

//=============================================================================================================

double MaxwellFilter::associated_legendre(int l, int m, double x)
{
    if (m < 0) {
        m = -m;
        double sign = (m % 2 == 0) ? 1.0 : -1.0;
        double factor = factorial(l - m) / factorial(l + m);
        return sign * factor * associated_legendre(l, m, x);
    }
    
    if (m > l) {
        return 0.0;
    }
    
    // Use recurrence relation for associated Legendre polynomials
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    
    if (l == m) {
        return pmm;
    }
    
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) {
        return pmmp1;
    }
    
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    
    return pll;
}

//=============================================================================================================

complex<double> MaxwellFilter::spherical_harmonic(int l, int m, double theta, double phi)
{
    double cos_theta = cos(theta);
    double P_lm = associated_legendre(l, abs(m), cos_theta);
    
    // Normalization factor
    double norm = sqrt((2 * l + 1) * factorial(l - abs(m)) / (4.0 * M_PI * factorial(l + abs(m))));
    
    complex<double> exp_imphi(cos(m * phi), sin(m * phi));
    
    if (m < 0) {
        double sign = (abs(m) % 2 == 0) ? 1.0 : -1.0;
        exp_imphi = sign * conj(exp_imphi);
    }
    
    return norm * P_lm * exp_imphi;
}

//=============================================================================================================

void MaxwellFilter::cartesian_to_spherical(const Vector3d& position,
                                          double& r, double& theta, double& phi)
{
    r = position.norm();
    
    if (r < 1e-10) {
        theta = 0.0;
        phi = 0.0;
        return;
    }
    
    theta = acos(position.z() / r);
    phi = atan2(position.y(), position.x());
}

//=============================================================================================================

void MaxwellFilter::apply_regularization(MatrixXcd& basis, double regularize)
{
    if (regularize <= 0.0) {
        return;
    }
    
    // Apply Tikhonov regularization by adding small values to diagonal
    // This is done implicitly in the gram matrix computation
    Q_UNUSED(basis)
}

//=============================================================================================================

double MaxwellFilter::compute_goodness_of_fit(const MatrixXd& original_data,
                                             const MatrixXd& filtered_data)
{
    if (original_data.rows() != filtered_data.rows() || 
        original_data.cols() != filtered_data.cols()) {
        return 0.0;
    }
    
    double original_norm = original_data.norm();
    double residual_norm = (original_data - filtered_data).norm();
    
    if (original_norm < 1e-10) {
        return 1.0;
    }
    
    return 1.0 - (residual_norm / original_norm);
}

//=============================================================================================================

std::vector<int> MaxwellFilter::detect_bad_channels(const VectorXd& reconstruction_errors,
                                                   double limit)
{
    std::vector<int> bad_channels;
    
    for (int ch = 0; ch < reconstruction_errors.size(); ++ch) {
        if (reconstruction_errors(ch) > limit) {
            bad_channels.push_back(ch);
        }
    }
    
    return bad_channels;
}

//=============================================================================================================

double MaxwellFilter::factorial(int n)
{
    if (n <= 1) {
        return 1.0;
    }
    
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    
    return result;
}