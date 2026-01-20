//=============================================================================================================
/**
 * @file     iirfilter.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Massachusetts General Hospital nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MASSACHUSETTS GENERAL HOSPITAL BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    IIR filter implementation.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "iirfilter.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QtMath>

//=============================================================================================================
// STD INCLUDES
//=============================================================================================================

#include <algorithm>
#include <cmath>
#include <complex>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace RTPROCESSINGLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

IIRFilter::IIRFilter()
: m_isDesigned(false)
{
}

//=============================================================================================================

IIRFilter::IIRFilter(const FilterDesign& design)
: m_isDesigned(false)
{
    designFilter(design);
}

//=============================================================================================================

bool IIRFilter::designFilter(const FilterDesign& design)
{
    if (!validateDesign(design)) {
        qWarning() << "[IIRFilter::designFilter] Invalid filter design parameters";
        return false;
    }
    
    m_design = design;
    
    bool success = false;
    
    switch (design.method) {
        case BUTTERWORTH:
            success = designButterworth(design);
            break;
        case CHEBYSHEV1:
            success = designChebyshev1(design);
            break;
        case CHEBYSHEV2:
            success = designChebyshev2(design);
            break;
        case ELLIPTIC:
            success = designElliptic(design);
            break;
        case BESSEL:
            success = designBessel(design);
            break;
        default:
            qWarning() << "[IIRFilter::designFilter] Unknown design method";
            return false;
    }
    
    if (success) {
        // Initialize history buffers
        int maxOrder = std::max(m_state.b.size(), m_state.a.size()) - 1;
        m_state.x_history = VectorXd::Zero(maxOrder + 1);
        m_state.y_history = VectorXd::Zero(maxOrder + 1);
        m_state.historyIndex = 0;
        m_isDesigned = true;
    }
    
    return success;
}

//=============================================================================================================

VectorXd IIRFilter::filter(const VectorXd& input)
{
    if (!m_isDesigned) {
        qWarning() << "[IIRFilter::filter] Filter not designed";
        return input;
    }
    
    VectorXd output(input.size());
    
    for (int i = 0; i < input.size(); ++i) {
        output[i] = filterSample(input[i]);
    }
    
    return output;
}

//=============================================================================================================

double IIRFilter::filterSample(double sample)
{
    if (!m_isDesigned) {
        qWarning() << "[IIRFilter::filterSample] Filter not designed";
        return sample;
    }
    
    // Update input history
    m_state.x_history[m_state.historyIndex] = sample;
    
    // Compute output using difference equation
    // y[n] = (1/a[0]) * (Σ b[k]*x[n-k] - Σ a[k]*y[n-k])
    double output = 0.0;
    
    // Numerator part (feedforward)
    for (int k = 0; k < m_state.b.size(); ++k) {
        int index = (m_state.historyIndex - k + m_state.x_history.size()) % m_state.x_history.size();
        output += m_state.b[k] * m_state.x_history[index];
    }
    
    // Denominator part (feedback) - skip a[0] term
    for (int k = 1; k < m_state.a.size(); ++k) {
        int index = (m_state.historyIndex - k + 1 + m_state.y_history.size()) % m_state.y_history.size();
        output -= m_state.a[k] * m_state.y_history[index];
    }
    
    // Normalize by a[0]
    output /= m_state.a[0];
    
    // Update output history
    m_state.y_history[m_state.historyIndex] = output;
    
    // Update history index
    m_state.historyIndex = (m_state.historyIndex + 1) % m_state.x_history.size();
    
    return output;
}

//=============================================================================================================

void IIRFilter::reset()
{
    if (m_isDesigned) {
        m_state.x_history.setZero();
        m_state.y_history.setZero();
        m_state.historyIndex = 0;
    }
}

//=============================================================================================================

void IIRFilter::frequencyResponse(const VectorXd& frequencies,
                                  VectorXd& magnitude,
                                  VectorXd& phase) const
{
    if (!m_isDesigned) {
        qWarning() << "[IIRFilter::frequencyResponse] Filter not designed";
        return;
    }
    
    int numFreqs = frequencies.size();
    magnitude.resize(numFreqs);
    phase.resize(numFreqs);
    
    for (int i = 0; i < numFreqs; ++i) {
        double omega = 2.0 * M_PI * frequencies[i] / m_design.samplingRate;
        
        // Compute H(e^jω) = B(e^jω) / A(e^jω)
        std::complex<double> numerator(0.0, 0.0);
        std::complex<double> denominator(0.0, 0.0);
        
        // Numerator: B(e^jω) = Σ b[k] * e^(-jωk)
        for (int k = 0; k < m_state.b.size(); ++k) {
            std::complex<double> exponential(std::cos(-omega * k), std::sin(-omega * k));
            numerator += m_state.b[k] * exponential;
        }
        
        // Denominator: A(e^jω) = Σ a[k] * e^(-jωk)
        for (int k = 0; k < m_state.a.size(); ++k) {
            std::complex<double> exponential(std::cos(-omega * k), std::sin(-omega * k));
            denominator += m_state.a[k] * exponential;
        }
        
        std::complex<double> response = numerator / denominator;
        
        magnitude[i] = std::abs(response);
        phase[i] = std::arg(response);
    }
}

//=============================================================================================================

VectorXd IIRFilter::groupDelay(const VectorXd& frequencies) const
{
    if (!m_isDesigned) {
        qWarning() << "[IIRFilter::groupDelay] Filter not designed";
        return VectorXd();
    }
    
    int numFreqs = frequencies.size();
    VectorXd delay(numFreqs);
    
    // Compute group delay as -d(phase)/dω
    VectorXd magnitude, phase;
    frequencyResponse(frequencies, magnitude, phase);
    
    for (int i = 0; i < numFreqs; ++i) {
        if (i == 0) {
            delay[i] = -(phase[1] - phase[0]) / (2.0 * M_PI * (frequencies[1] - frequencies[0]) / m_design.samplingRate);
        } else if (i == numFreqs - 1) {
            delay[i] = -(phase[i] - phase[i-1]) / (2.0 * M_PI * (frequencies[i] - frequencies[i-1]) / m_design.samplingRate);
        } else {
            delay[i] = -(phase[i+1] - phase[i-1]) / (2.0 * M_PI * (frequencies[i+1] - frequencies[i-1]) / m_design.samplingRate);
        }
    }
    
    return delay;
}

//=============================================================================================================

void IIRFilter::getCoefficients(VectorXd& b, VectorXd& a) const
{
    b = m_state.b;
    a = m_state.a;
}

//=============================================================================================================

void IIRFilter::setCoefficients(const VectorXd& b, const VectorXd& a)
{
    m_state.b = b;
    m_state.a = a;
    
    // Initialize history buffers
    int maxOrder = std::max(b.size(), a.size()) - 1;
    m_state.x_history = VectorXd::Zero(maxOrder + 1);
    m_state.y_history = VectorXd::Zero(maxOrder + 1);
    m_state.historyIndex = 0;
    
    m_isDesigned = true;
}

//=============================================================================================================

IIRFilter::FilterDesign IIRFilter::getDesign() const
{
    return m_design;
}

//=============================================================================================================

bool IIRFilter::validateDesign(const FilterDesign& design)
{
    // Check sampling rate
    if (design.samplingRate <= 0) {
        qWarning() << "[IIRFilter::validateDesign] Invalid sampling rate";
        return false;
    }
    
    // Check filter order
    if (design.order < 1) {
        qWarning() << "[IIRFilter::validateDesign] Invalid filter order";
        return false;
    }
    
    // Check cutoff frequencies
    double nyquist = design.samplingRate / 2.0;
    for (int i = 0; i < design.cutoffFreqs.size(); ++i) {
        if (design.cutoffFreqs[i] <= 0 || design.cutoffFreqs[i] >= nyquist) {
            qWarning() << "[IIRFilter::validateDesign] Invalid cutoff frequency";
            return false;
        }
    }
    
    // Check number of cutoff frequencies for filter type
    switch (design.type) {
        case LOWPASS:
        case HIGHPASS:
            if (design.cutoffFreqs.size() != 1) {
                qWarning() << "[IIRFilter::validateDesign] Low/High-pass filters require exactly one cutoff frequency";
                return false;
            }
            break;
        case BANDPASS:
        case BANDSTOP:
            if (design.cutoffFreqs.size() != 2) {
                qWarning() << "[IIRFilter::validateDesign] Band-pass/stop filters require exactly two cutoff frequencies";
                return false;
            }
            if (design.cutoffFreqs[0] >= design.cutoffFreqs[1]) {
                qWarning() << "[IIRFilter::validateDesign] Lower cutoff must be less than upper cutoff";
                return false;
            }
            break;
    }
    
    return true;
}

//=============================================================================================================

int IIRFilter::estimateOrder(double passbandRipple, double stopbandAtten,
                            double passbandFreq, double stopbandFreq,
                            DesignMethod method)
{
    double wp = passbandFreq;
    double ws = stopbandFreq;
    double Rp = passbandRipple;
    double Rs = stopbandAtten;
    
    int order = 1;
    
    switch (method) {
        case BUTTERWORTH:
            {
                double k = wp / ws;
                double epsilon = std::sqrt(std::pow(10.0, Rp / 10.0) - 1.0);
                double A = std::pow(10.0, Rs / 20.0);
                order = static_cast<int>(std::ceil(std::log(A * A - 1.0) / (2.0 * std::log(1.0 / k)) - std::log(epsilon * epsilon) / (2.0 * std::log(1.0 / k))));
            }
            break;
        case CHEBYSHEV1:
            {
                double k = wp / ws;
                double epsilon = std::sqrt(std::pow(10.0, Rp / 10.0) - 1.0);
                double A = std::pow(10.0, Rs / 20.0);
                order = static_cast<int>(std::ceil(std::acosh(A / epsilon) / std::acosh(1.0 / k)));
            }
            break;
        case CHEBYSHEV2:
            {
                double k = wp / ws;
                double epsilon = std::sqrt(std::pow(10.0, Rp / 10.0) - 1.0);
                double A = std::pow(10.0, Rs / 20.0);
                order = static_cast<int>(std::ceil(std::acosh(A / epsilon) / std::acosh(k)));
            }
            break;
        default:
            // Use Butterworth estimation as default
            {
                double k = wp / ws;
                double epsilon = std::sqrt(std::pow(10.0, Rp / 10.0) - 1.0);
                double A = std::pow(10.0, Rs / 20.0);
                order = static_cast<int>(std::ceil(std::log(A * A - 1.0) / (2.0 * std::log(1.0 / k)) - std::log(epsilon * epsilon) / (2.0 * std::log(1.0 / k))));
            }
            break;
    }
    
    return std::max(order, 1);
}

//=============================================================================================================

bool IIRFilter::designButterworth(const FilterDesign& design)
{
    // Simplified Butterworth design - create basic low-pass prototype
    VectorXd normalizedCutoffs = normalizeCutoffFreqs(design.cutoffFreqs, design.samplingRate);
    
    // For simplicity, create a basic 2nd order Butterworth filter
    // This is a simplified implementation - a full implementation would use
    // analog prototype poles and bilinear transformation
    
    double wc = normalizedCutoffs[0] * M_PI; // Convert to radians
    double k = std::tan(wc / 2.0);
    double k2 = k * k;
    double sqrt2 = std::sqrt(2.0);
    double norm = 1.0 + sqrt2 * k + k2;
    
    // Second-order Butterworth coefficients
    m_state.b.resize(3);
    m_state.a.resize(3);
    
    switch (design.type) {
        case LOWPASS:
            m_state.b[0] = k2 / norm;
            m_state.b[1] = 2.0 * k2 / norm;
            m_state.b[2] = k2 / norm;
            m_state.a[0] = 1.0;
            m_state.a[1] = (2.0 * (k2 - 1.0)) / norm;
            m_state.a[2] = (1.0 - sqrt2 * k + k2) / norm;
            break;
            
        case HIGHPASS:
            m_state.b[0] = 1.0 / norm;
            m_state.b[1] = -2.0 / norm;
            m_state.b[2] = 1.0 / norm;
            m_state.a[0] = 1.0;
            m_state.a[1] = (2.0 * (k2 - 1.0)) / norm;
            m_state.a[2] = (1.0 - sqrt2 * k + k2) / norm;
            break;
            
        case BANDPASS:
            // Simplified bandpass - would need proper transformation
            if (design.cutoffFreqs.size() >= 2) {
                double wc1 = normalizeCutoffFreqs(design.cutoffFreqs.segment(0, 1), design.samplingRate)[0] * M_PI;
                double wc2 = normalizeCutoffFreqs(design.cutoffFreqs.segment(1, 1), design.samplingRate)[0] * M_PI;
                double bw = wc2 - wc1;
                double wc_center = (wc1 + wc2) / 2.0;
                
                // Basic bandpass approximation
                m_state.b.resize(3);
                m_state.a.resize(3);
                m_state.b[0] = bw / 2.0;
                m_state.b[1] = 0.0;
                m_state.b[2] = -bw / 2.0;
                m_state.a[0] = 1.0;
                m_state.a[1] = -2.0 * std::cos(wc_center);
                m_state.a[2] = 1.0 - bw;
            }
            break;
            
        default:
            return false;
    }
    
    return true;
}

//=============================================================================================================

bool IIRFilter::designChebyshev1(const FilterDesign& design)
{
    // Simplified Chebyshev Type I implementation
    // In a full implementation, this would compute the analog prototype
    // and apply bilinear transformation
    
    qWarning() << "[IIRFilter::designChebyshev1] Simplified implementation - using Butterworth";
    return designButterworth(design);
}

//=============================================================================================================

bool IIRFilter::designChebyshev2(const FilterDesign& design)
{
    // Simplified Chebyshev Type II implementation
    qWarning() << "[IIRFilter::designChebyshev2] Simplified implementation - using Butterworth";
    return designButterworth(design);
}

//=============================================================================================================

bool IIRFilter::designElliptic(const FilterDesign& design)
{
    // Simplified Elliptic implementation
    qWarning() << "[IIRFilter::designElliptic] Simplified implementation - using Butterworth";
    return designButterworth(design);
}

//=============================================================================================================

bool IIRFilter::designBessel(const FilterDesign& design)
{
    // Simplified Bessel implementation
    qWarning() << "[IIRFilter::designBessel] Simplified implementation - using Butterworth";
    return designButterworth(design);
}

//=============================================================================================================

void IIRFilter::bilinearTransform(const VectorXd& s_num, const VectorXd& s_den,
                                  double fs, VectorXd& z_num, VectorXd& z_den)
{
    // Simplified bilinear transformation implementation
    // This is a placeholder for the full bilinear transformation algorithm
    z_num = s_num;
    z_den = s_den;
}

//=============================================================================================================

void IIRFilter::frequencyTransform(const FilterDesign& design,
                                   const VectorXd& prototype_num,
                                   const VectorXd& prototype_den,
                                   VectorXd& transformed_num,
                                   VectorXd& transformed_den)
{
    // Simplified frequency transformation implementation
    transformed_num = prototype_num;
    transformed_den = prototype_den;
}

//=============================================================================================================

VectorXd IIRFilter::normalizeCutoffFreqs(const VectorXd& cutoffFreqs, double samplingRate)
{
    double nyquist = samplingRate / 2.0;
    return cutoffFreqs / nyquist;
}