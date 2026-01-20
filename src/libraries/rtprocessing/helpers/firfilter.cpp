//=============================================================================================================
/**
 * @file     firfilter.cpp
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
 * @brief    Enhanced FIR filter implementation.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "firfilter.h"

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

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace RTPROCESSINGLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

FIRFilter::FIRFilter()
: m_isDesigned(false)
{
}

//=============================================================================================================

FIRFilter::FIRFilter(const FilterDesign& design)
: m_isDesigned(false)
{
    designFilter(design);
}

//=============================================================================================================

bool FIRFilter::designFilter(const FilterDesign& design)
{
    if (!validateDesign(design)) {
        qWarning() << "[FIRFilter::designFilter] Invalid filter design parameters";
        return false;
    }
    
    m_design = design;
    
    // Design ideal filter response
    VectorXd idealResponse = designIdealResponse(design);
    
    // Apply window function
    m_coefficients = applyWindow(idealResponse, design);
    
    m_isDesigned = true;
    return true;
}

//=============================================================================================================

VectorXd FIRFilter::filter(const VectorXd& input) const
{
    if (!m_isDesigned) {
        qWarning() << "[FIRFilter::filter] Filter not designed";
        return input;
    }
    
    int inputLength = input.size();
    int filterLength = m_coefficients.size();
    int outputLength = inputLength + filterLength - 1;
    
    VectorXd output = VectorXd::Zero(outputLength);
    
    // Convolution
    for (int n = 0; n < outputLength; ++n) {
        for (int k = 0; k < filterLength; ++k) {
            int inputIndex = n - k;
            if (inputIndex >= 0 && inputIndex < inputLength) {
                output[n] += m_coefficients[k] * input[inputIndex];
            }
        }
    }
    
    // Return same length as input (remove transient)
    int delay = filterLength / 2;
    return output.segment(delay, inputLength);
}

//=============================================================================================================

VectorXd FIRFilter::filtfilt(const VectorXd& input) const
{
    if (!m_isDesigned) {
        qWarning() << "[FIRFilter::filtfilt] Filter not designed";
        return input;
    }
    
    // Forward filtering
    VectorXd forward = filter(input);
    
    // Reverse the signal
    VectorXd reversed = forward.reverse();
    
    // Filter in reverse direction
    VectorXd backwardFiltered = filter(reversed);
    
    // Reverse again to get final result
    return backwardFiltered.reverse();
}

//=============================================================================================================

void FIRFilter::frequencyResponse(const VectorXd& frequencies,
                                  VectorXd& magnitude,
                                  VectorXd& phase) const
{
    if (!m_isDesigned) {
        qWarning() << "[FIRFilter::frequencyResponse] Filter not designed";
        return;
    }
    
    int numFreqs = frequencies.size();
    magnitude.resize(numFreqs);
    phase.resize(numFreqs);
    
    double nyquist = m_design.samplingRate / 2.0;
    
    for (int i = 0; i < numFreqs; ++i) {
        double omega = 2.0 * M_PI * frequencies[i] / m_design.samplingRate;
        
        std::complex<double> response(0.0, 0.0);
        
        // Compute frequency response H(e^jω) = Σ h[n] * e^(-jωn)
        for (int n = 0; n < m_coefficients.size(); ++n) {
            std::complex<double> exponential(std::cos(-omega * n), std::sin(-omega * n));
            response += m_coefficients[n] * exponential;
        }
        
        magnitude[i] = std::abs(response);
        phase[i] = std::arg(response);
    }
}

//=============================================================================================================

VectorXd FIRFilter::groupDelay(const VectorXd& frequencies) const
{
    if (!m_isDesigned) {
        qWarning() << "[FIRFilter::groupDelay] Filter not designed";
        return VectorXd();
    }
    
    int numFreqs = frequencies.size();
    VectorXd delay(numFreqs);
    
    // For linear phase FIR filters, group delay is constant
    if (m_design.linearPhase) {
        double constantDelay = (m_coefficients.size() - 1) / 2.0;
        delay.setConstant(constantDelay);
    } else {
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
    }
    
    return delay;
}

//=============================================================================================================

VectorXd FIRFilter::getCoefficients() const
{
    return m_coefficients;
}

//=============================================================================================================

void FIRFilter::setCoefficients(const VectorXd& coefficients)
{
    m_coefficients = coefficients;
    m_isDesigned = true;
}

//=============================================================================================================

FIRFilter::FilterDesign FIRFilter::getDesign() const
{
    return m_design;
}

//=============================================================================================================

bool FIRFilter::validateDesign(const FilterDesign& design)
{
    // Check sampling rate
    if (design.samplingRate <= 0) {
        qWarning() << "[FIRFilter::validateDesign] Invalid sampling rate";
        return false;
    }
    
    // Check filter order
    if (design.order < 1) {
        qWarning() << "[FIRFilter::validateDesign] Invalid filter order";
        return false;
    }
    
    // Check cutoff frequencies
    double nyquist = design.samplingRate / 2.0;
    for (int i = 0; i < design.cutoffFreqs.size(); ++i) {
        if (design.cutoffFreqs[i] <= 0 || design.cutoffFreqs[i] >= nyquist) {
            qWarning() << "[FIRFilter::validateDesign] Invalid cutoff frequency";
            return false;
        }
    }
    
    // Check number of cutoff frequencies for filter type
    switch (design.type) {
        case LOWPASS:
        case HIGHPASS:
            if (design.cutoffFreqs.size() != 1) {
                qWarning() << "[FIRFilter::validateDesign] Low/High-pass filters require exactly one cutoff frequency";
                return false;
            }
            break;
        case BANDPASS:
        case BANDSTOP:
            if (design.cutoffFreqs.size() != 2) {
                qWarning() << "[FIRFilter::validateDesign] Band-pass/stop filters require exactly two cutoff frequencies";
                return false;
            }
            if (design.cutoffFreqs[0] >= design.cutoffFreqs[1]) {
                qWarning() << "[FIRFilter::validateDesign] Lower cutoff must be less than upper cutoff";
                return false;
            }
            break;
        default:
            break;
    }
    
    return true;
}

//=============================================================================================================

VectorXd FIRFilter::createWindow(WindowType type, int length, double beta, double alpha, double sigma)
{
    VectorXd window(length);
    
    switch (type) {
        case RECTANGULAR:
            window.setOnes();
            break;
            
        case HAMMING:
            for (int n = 0; n < length; ++n) {
                window[n] = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / (length - 1));
            }
            break;
            
        case HANNING:
            for (int n = 0; n < length; ++n) {
                window[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (length - 1)));
            }
            break;
            
        case BLACKMAN:
            for (int n = 0; n < length; ++n) {
                double arg = 2.0 * M_PI * n / (length - 1);
                window[n] = 0.42 - 0.5 * std::cos(arg) + 0.08 * std::cos(2.0 * arg);
            }
            break;
            
        case BLACKMAN_HARRIS:
            for (int n = 0; n < length; ++n) {
                double arg = 2.0 * M_PI * n / (length - 1);
                window[n] = 0.35875 - 0.48829 * std::cos(arg) + 0.14128 * std::cos(2.0 * arg) - 0.01168 * std::cos(3.0 * arg);
            }
            break;
            
        case KAISER:
            {
                // Simplified Kaiser window using approximation
                // I0(x) ≈ 1 + (x/2)^2 + (x/2)^4/4 + (x/2)^6/36 + ... (for small x)
                auto besselI0 = [](double x) -> double {
                    double sum = 1.0;
                    double term = 1.0;
                    double x_half = x / 2.0;
                    double x_half_sq = x_half * x_half;
                    
                    for (int k = 1; k <= 20; ++k) {
                        term *= x_half_sq / (k * k);
                        sum += term;
                        if (term < 1e-10) break;
                    }
                    return sum;
                };
                
                double I0_beta = besselI0(beta);
                for (int n = 0; n < length; ++n) {
                    double arg = beta * std::sqrt(1.0 - std::pow(2.0 * n / (length - 1) - 1.0, 2));
                    window[n] = besselI0(arg) / I0_beta;
                }
            }
            break;
            
        case TUKEY:
            {
                int transitionSamples = static_cast<int>(alpha * (length - 1) / 2.0);
                for (int n = 0; n < length; ++n) {
                    if (n < transitionSamples) {
                        window[n] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * n / (alpha * (length - 1)) - 1.0)));
                    } else if (n >= length - transitionSamples) {
                        window[n] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * (n - length + 1) / (alpha * (length - 1)) + 1.0)));
                    } else {
                        window[n] = 1.0;
                    }
                }
            }
            break;
            
        case GAUSSIAN:
            {
                double center = (length - 1) / 2.0;
                for (int n = 0; n < length; ++n) {
                    double arg = (n - center) / (sigma * length / 2.0);
                    window[n] = std::exp(-0.5 * arg * arg);
                }
            }
            break;
            
        case BARTLETT:
            for (int n = 0; n < length; ++n) {
                window[n] = 1.0 - std::abs(2.0 * n / (length - 1) - 1.0);
            }
            break;
            
        case FLATTOP:
            for (int n = 0; n < length; ++n) {
                double arg = 2.0 * M_PI * n / (length - 1);
                window[n] = 0.21557895 - 0.41663158 * std::cos(arg) + 0.277263158 * std::cos(2.0 * arg) 
                           - 0.083578947 * std::cos(3.0 * arg) + 0.006947368 * std::cos(4.0 * arg);
            }
            break;
            
        default:
            window.setOnes();
            break;
    }
    
    return window;
}

//=============================================================================================================

int FIRFilter::estimateOrder(double passbandRipple, double stopbandAtten, 
                            double transitionWidth, WindowType window)
{
    // Kaiser's formula for estimating filter order
    double A = stopbandAtten;
    double deltaF = transitionWidth;
    
    int order;
    
    switch (window) {
        case HAMMING:
            order = static_cast<int>(std::ceil((A - 21.0) / (22.0 * deltaF)));
            break;
        case HANNING:
            order = static_cast<int>(std::ceil((A - 21.0) / (22.0 * deltaF)));
            break;
        case BLACKMAN:
            order = static_cast<int>(std::ceil((A - 13.0) / (27.0 * deltaF)));
            break;
        case KAISER:
            {
                double beta;
                if (A > 50.0) {
                    beta = 0.1102 * (A - 8.7);
                } else if (A >= 21.0) {
                    beta = 0.5842 * std::pow(A - 21.0, 0.4) + 0.07886 * (A - 21.0);
                } else {
                    beta = 0.0;
                }
                order = static_cast<int>(std::ceil((A - 8.0) / (2.285 * deltaF * M_PI)));
            }
            break;
        default:
            order = static_cast<int>(std::ceil((A - 21.0) / (22.0 * deltaF)));
            break;
    }
    
    // Ensure odd order for linear phase
    if (order % 2 == 0) {
        order++;
    }
    
    return std::max(order, 3);
}

//=============================================================================================================

VectorXd FIRFilter::designIdealResponse(const FilterDesign& design)
{
    int numTaps = design.order + 1;
    VectorXd idealResponse(numTaps);
    
    // Normalize cutoff frequencies
    VectorXd normalizedCutoffs = normalizeCutoffFreqs(design.cutoffFreqs, design.samplingRate);
    
    int center = numTaps / 2;
    
    switch (design.type) {
        case LOWPASS:
            {
                double wc = normalizedCutoffs[0] * M_PI;
                for (int n = 0; n < numTaps; ++n) {
                    if (n == center) {
                        idealResponse[n] = wc / M_PI;
                    } else {
                        double arg = wc * (n - center);
                        idealResponse[n] = std::sin(arg) / (M_PI * (n - center));
                    }
                }
            }
            break;
            
        case HIGHPASS:
            {
                double wc = normalizedCutoffs[0] * M_PI;
                for (int n = 0; n < numTaps; ++n) {
                    if (n == center) {
                        idealResponse[n] = 1.0 - wc / M_PI;
                    } else {
                        double arg = M_PI * (n - center);
                        double arg2 = wc * (n - center);
                        idealResponse[n] = (std::sin(arg) - std::sin(arg2)) / (M_PI * (n - center));
                    }
                }
            }
            break;
            
        case BANDPASS:
            {
                double wc1 = normalizedCutoffs[0] * M_PI;
                double wc2 = normalizedCutoffs[1] * M_PI;
                for (int n = 0; n < numTaps; ++n) {
                    if (n == center) {
                        idealResponse[n] = (wc2 - wc1) / M_PI;
                    } else {
                        double arg1 = wc2 * (n - center);
                        double arg2 = wc1 * (n - center);
                        idealResponse[n] = (std::sin(arg1) - std::sin(arg2)) / (M_PI * (n - center));
                    }
                }
            }
            break;
            
        case BANDSTOP:
            {
                double wc1 = normalizedCutoffs[0] * M_PI;
                double wc2 = normalizedCutoffs[1] * M_PI;
                for (int n = 0; n < numTaps; ++n) {
                    if (n == center) {
                        idealResponse[n] = 1.0 - (wc2 - wc1) / M_PI;
                    } else {
                        double arg = M_PI * (n - center);
                        double arg1 = wc1 * (n - center);
                        double arg2 = wc2 * (n - center);
                        idealResponse[n] = (std::sin(arg) - std::sin(arg2) + std::sin(arg1)) / (M_PI * (n - center));
                    }
                }
            }
            break;
            
        case HILBERT:
            for (int n = 0; n < numTaps; ++n) {
                if (n == center) {
                    idealResponse[n] = 0.0;
                } else {
                    int k = n - center;
                    if (k % 2 != 0) {
                        idealResponse[n] = 2.0 / (M_PI * k);
                    } else {
                        idealResponse[n] = 0.0;
                    }
                }
            }
            break;
            
        case DIFFERENTIATOR:
            for (int n = 0; n < numTaps; ++n) {
                if (n == center) {
                    idealResponse[n] = 0.0;
                } else {
                    int k = n - center;
                    idealResponse[n] = std::cos(M_PI * k) / k;
                }
            }
            break;
            
        default:
            idealResponse.setZero();
            break;
    }
    
    // Ensure linear phase by enforcing symmetry
    if (design.linearPhase) {
        for (int n = 0; n < numTaps / 2; ++n) {
            double avg = (idealResponse[n] + idealResponse[numTaps - 1 - n]) / 2.0;
            idealResponse[n] = avg;
            idealResponse[numTaps - 1 - n] = avg;
        }
    }
    
    return idealResponse;
}

//=============================================================================================================

VectorXd FIRFilter::applyWindow(const VectorXd& idealResponse, const FilterDesign& design)
{
    VectorXd window = createWindow(design.window, idealResponse.size(), 
                                  design.kaiserBeta, design.tukeyAlpha, design.gaussianSigma);
    
    VectorXd windowed = idealResponse.cwiseProduct(window);
    
    // Ensure linear phase by enforcing symmetry after windowing
    if (design.linearPhase) {
        int numTaps = windowed.size();
        for (int n = 0; n < numTaps / 2; ++n) {
            double avg = (windowed[n] + windowed[numTaps - 1 - n]) / 2.0;
            windowed[n] = avg;
            windowed[numTaps - 1 - n] = avg;
        }
    }
    
    return windowed;
}

//=============================================================================================================

VectorXd FIRFilter::normalizeCutoffFreqs(const VectorXd& cutoffFreqs, double samplingRate)
{
    double nyquist = samplingRate / 2.0;
    return cutoffFreqs / nyquist;
}