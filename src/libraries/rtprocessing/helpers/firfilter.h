//=============================================================================================================
/**
 * @file     firfilter.h
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
 * @brief    Enhanced FIR filter design and implementation with multiple window functions
 *
 */

#ifndef FIRFILTER_H
#define FIRFILTER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "../rtprocessing_global.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QSharedPointer>
#include <QString>
#include <QVector>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>

//=============================================================================================================
// DEFINE NAMESPACE RTPROCESSINGLIB
//=============================================================================================================

namespace RTPROCESSINGLIB {

//=============================================================================================================
/**
 * Enhanced FIR filter design class with multiple window functions and linear phase response.
 * Provides comprehensive FIR filter design capabilities including various window functions,
 * frequency response analysis, and optimized filtering operations.
 *
 * @brief Enhanced FIR filter design and implementation.
 */
class RTPROCESINGSHARED_EXPORT FIRFilter
{

public:
    typedef QSharedPointer<FIRFilter> SPtr;            /**< Shared pointer type for FIRFilter. */
    typedef QSharedPointer<const FIRFilter> ConstSPtr; /**< Const shared pointer type for FIRFilter. */

    //=========================================================================================================
    /**
     * Enumeration of available window functions for FIR filter design.
     */
    enum WindowType {
        RECTANGULAR = 0,    /**< Rectangular window (no windowing) */
        HAMMING,           /**< Hamming window */
        HANNING,           /**< Hanning (Hann) window */
        BLACKMAN,          /**< Blackman window */
        BLACKMAN_HARRIS,   /**< Blackman-Harris window */
        KAISER,            /**< Kaiser window (requires beta parameter) */
        TUKEY,             /**< Tukey (tapered cosine) window */
        GAUSSIAN,          /**< Gaussian window */
        BARTLETT,          /**< Bartlett (triangular) window */
        FLATTOP            /**< Flat-top window */
    };

    //=========================================================================================================
    /**
     * Enumeration of FIR filter types.
     */
    enum FilterType {
        LOWPASS = 0,       /**< Low-pass filter */
        HIGHPASS,          /**< High-pass filter */
        BANDPASS,          /**< Band-pass filter */
        BANDSTOP,          /**< Band-stop (notch) filter */
        HILBERT,           /**< Hilbert transformer */
        DIFFERENTIATOR     /**< Differentiator */
    };

    //=========================================================================================================
    /**
     * Structure to hold filter design parameters.
     */
    struct FilterDesign {
        FilterType type;                    /**< Filter type */
        WindowType window;                  /**< Window function type */
        int order;                         /**< Filter order (number of taps - 1) */
        double samplingRate;               /**< Sampling rate in Hz */
        Eigen::VectorXd cutoffFreqs;       /**< Cutoff frequencies in Hz */
        double kaiserBeta;                 /**< Kaiser window beta parameter */
        double tukeyAlpha;                 /**< Tukey window alpha parameter */
        double gaussianSigma;              /**< Gaussian window sigma parameter */
        bool linearPhase;                  /**< Ensure linear phase response */
        
        FilterDesign() : type(LOWPASS), window(HAMMING), order(64), samplingRate(1000.0),
                        kaiserBeta(5.0), tukeyAlpha(0.5), gaussianSigma(0.4), linearPhase(true) {
            cutoffFreqs.resize(1);
            cutoffFreqs[0] = 100.0;
        }
    };

    //=========================================================================================================
    /**
     * Constructs a FIRFilter object.
     */
    explicit FIRFilter();

    //=========================================================================================================
    /**
     * Constructs a FIRFilter object with specified design parameters.
     *
     * @param[in] design    Filter design parameters.
     */
    explicit FIRFilter(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs a FIR filter using the windowing method.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if design was successful, false otherwise.
     */
    bool designFilter(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Applies the filter to input data.
     *
     * @param[in] input     Input signal data.
     *
     * @return Filtered output signal.
     */
    Eigen::VectorXd filter(const Eigen::VectorXd& input) const;

    //=========================================================================================================
    /**
     * Applies the filter to input data with zero-phase filtering (forward-backward).
     *
     * @param[in] input     Input signal data.
     *
     * @return Zero-phase filtered output signal.
     */
    Eigen::VectorXd filtfilt(const Eigen::VectorXd& input) const;

    //=========================================================================================================
    /**
     * Computes the frequency response of the filter.
     *
     * @param[in] frequencies   Frequency points to evaluate (in Hz).
     * @param[out] magnitude    Magnitude response.
     * @param[out] phase        Phase response (in radians).
     */
    void frequencyResponse(const Eigen::VectorXd& frequencies,
                          Eigen::VectorXd& magnitude,
                          Eigen::VectorXd& phase) const;

    //=========================================================================================================
    /**
     * Computes the group delay of the filter.
     *
     * @param[in] frequencies   Frequency points to evaluate (in Hz).
     *
     * @return Group delay in samples.
     */
    Eigen::VectorXd groupDelay(const Eigen::VectorXd& frequencies) const;

    //=========================================================================================================
    /**
     * Gets the filter coefficients.
     *
     * @return Filter coefficients.
     */
    Eigen::VectorXd getCoefficients() const;

    //=========================================================================================================
    /**
     * Sets custom filter coefficients.
     *
     * @param[in] coefficients  Filter coefficients.
     */
    void setCoefficients(const Eigen::VectorXd& coefficients);

    //=========================================================================================================
    /**
     * Gets the current filter design parameters.
     *
     * @return Filter design parameters.
     */
    FilterDesign getDesign() const;

    //=========================================================================================================
    /**
     * Validates filter design parameters.
     *
     * @param[in] design    Filter design parameters to validate.
     *
     * @return true if parameters are valid, false otherwise.
     */
    static bool validateDesign(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Creates a window function.
     *
     * @param[in] type      Window type.
     * @param[in] length    Window length.
     * @param[in] beta      Kaiser window beta parameter (ignored for other windows).
     * @param[in] alpha     Tukey window alpha parameter (ignored for other windows).
     * @param[in] sigma     Gaussian window sigma parameter (ignored for other windows).
     *
     * @return Window coefficients.
     */
    static Eigen::VectorXd createWindow(WindowType type, int length, 
                                       double beta = 5.0, double alpha = 0.5, double sigma = 0.4);

    //=========================================================================================================
    /**
     * Estimates the required filter order for given specifications.
     *
     * @param[in] passbandRipple    Passband ripple in dB.
     * @param[in] stopbandAtten     Stopband attenuation in dB.
     * @param[in] transitionWidth   Transition width normalized to Nyquist frequency.
     * @param[in] window           Window type.
     *
     * @return Estimated filter order.
     */
    static int estimateOrder(double passbandRipple, double stopbandAtten, 
                           double transitionWidth, WindowType window = HAMMING);

private:
    //=========================================================================================================
    /**
     * Designs the ideal filter response.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return Ideal impulse response.
     */
    Eigen::VectorXd designIdealResponse(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Applies window function to the ideal response.
     *
     * @param[in] idealResponse     Ideal impulse response.
     * @param[in] design           Filter design parameters.
     *
     * @return Windowed filter coefficients.
     */
    Eigen::VectorXd applyWindow(const Eigen::VectorXd& idealResponse, const FilterDesign& design);

    //=========================================================================================================
    /**
     * Normalizes cutoff frequencies to the range [0, 1] where 1 is Nyquist frequency.
     *
     * @param[in] cutoffFreqs       Cutoff frequencies in Hz.
     * @param[in] samplingRate     Sampling rate in Hz.
     *
     * @return Normalized cutoff frequencies.
     */
    static Eigen::VectorXd normalizeCutoffFreqs(const Eigen::VectorXd& cutoffFreqs, double samplingRate);

    FilterDesign        m_design;           /**< Current filter design parameters */
    Eigen::VectorXd     m_coefficients;     /**< Filter coefficients */
    bool                m_isDesigned;       /**< Flag indicating if filter has been designed */
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // namespace RTPROCESSINGLIB

#endif // FIRFILTER_H