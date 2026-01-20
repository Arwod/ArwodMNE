//=============================================================================================================
/**
 * @file     iirfilter.h
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
 * @brief    IIR filter design and real-time filtering implementation
 *
 */

#ifndef IIRFILTER_H
#define IIRFILTER_H

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
 * IIR filter design and real-time filtering class.
 * Provides comprehensive IIR filter design capabilities including Butterworth, Chebyshev,
 * and Elliptic filters with real-time processing support.
 *
 * @brief IIR filter design and real-time filtering implementation.
 */
class RTPROCESINGSHARED_EXPORT IIRFilter
{

public:
    typedef QSharedPointer<IIRFilter> SPtr;            /**< Shared pointer type for IIRFilter. */
    typedef QSharedPointer<const IIRFilter> ConstSPtr; /**< Const shared pointer type for IIRFilter. */

    //=========================================================================================================
    /**
     * Enumeration of available IIR filter types.
     */
    enum FilterType {
        LOWPASS = 0,       /**< Low-pass filter */
        HIGHPASS,          /**< High-pass filter */
        BANDPASS,          /**< Band-pass filter */
        BANDSTOP           /**< Band-stop (notch) filter */
    };

    //=========================================================================================================
    /**
     * Enumeration of IIR filter design methods.
     */
    enum DesignMethod {
        BUTTERWORTH = 0,   /**< Butterworth filter (maximally flat passband) */
        CHEBYSHEV1,        /**< Chebyshev Type I filter (ripple in passband) */
        CHEBYSHEV2,        /**< Chebyshev Type II filter (ripple in stopband) */
        ELLIPTIC,          /**< Elliptic filter (ripple in both bands) */
        BESSEL             /**< Bessel filter (maximally flat group delay) */
    };

    //=========================================================================================================
    /**
     * Structure to hold filter design parameters.
     */
    struct FilterDesign {
        FilterType type;                    /**< Filter type */
        DesignMethod method;                /**< Design method */
        int order;                         /**< Filter order */
        double samplingRate;               /**< Sampling rate in Hz */
        Eigen::VectorXd cutoffFreqs;       /**< Cutoff frequencies in Hz */
        double passbandRipple;             /**< Passband ripple in dB (for Chebyshev I and Elliptic) */
        double stopbandAtten;              /**< Stopband attenuation in dB (for Chebyshev II and Elliptic) */
        
        FilterDesign() : type(LOWPASS), method(BUTTERWORTH), order(4), samplingRate(1000.0),
                        passbandRipple(1.0), stopbandAtten(40.0) {
            cutoffFreqs.resize(1);
            cutoffFreqs[0] = 100.0;
        }
    };

    //=========================================================================================================
    /**
     * Structure to hold filter coefficients and state.
     */
    struct FilterState {
        Eigen::VectorXd b;                 /**< Numerator coefficients */
        Eigen::VectorXd a;                 /**< Denominator coefficients */
        Eigen::VectorXd x_history;         /**< Input history buffer */
        Eigen::VectorXd y_history;         /**< Output history buffer */
        int historyIndex;                  /**< Current position in history buffers */
        
        FilterState() : historyIndex(0) {}
    };

    //=========================================================================================================
    /**
     * Constructs an IIRFilter object.
     */
    explicit IIRFilter();

    //=========================================================================================================
    /**
     * Constructs an IIRFilter object with specified design parameters.
     *
     * @param[in] design    Filter design parameters.
     */
    explicit IIRFilter(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs an IIR filter with the specified parameters.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if design was successful, false otherwise.
     */
    bool designFilter(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Applies the filter to input data (batch processing).
     *
     * @param[in] input     Input signal data.
     *
     * @return Filtered output signal.
     */
    Eigen::VectorXd filter(const Eigen::VectorXd& input);

    //=========================================================================================================
    /**
     * Processes a single sample (real-time filtering).
     *
     * @param[in] sample    Input sample.
     *
     * @return Filtered output sample.
     */
    double filterSample(double sample);

    //=========================================================================================================
    /**
     * Resets the filter state (clears history buffers).
     */
    void reset();

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
     * @param[out] b    Numerator coefficients.
     * @param[out] a    Denominator coefficients.
     */
    void getCoefficients(Eigen::VectorXd& b, Eigen::VectorXd& a) const;

    //=========================================================================================================
    /**
     * Sets custom filter coefficients.
     *
     * @param[in] b     Numerator coefficients.
     * @param[in] a     Denominator coefficients.
     */
    void setCoefficients(const Eigen::VectorXd& b, const Eigen::VectorXd& a);

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
     * Estimates the required filter order for given specifications.
     *
     * @param[in] passbandRipple    Passband ripple in dB.
     * @param[in] stopbandAtten     Stopband attenuation in dB.
     * @param[in] passbandFreq      Passband edge frequency (normalized).
     * @param[in] stopbandFreq      Stopband edge frequency (normalized).
     * @param[in] method           Design method.
     *
     * @return Estimated filter order.
     */
    static int estimateOrder(double passbandRipple, double stopbandAtten,
                           double passbandFreq, double stopbandFreq,
                           DesignMethod method = BUTTERWORTH);

private:
    //=========================================================================================================
    /**
     * Designs a Butterworth filter.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if successful, false otherwise.
     */
    bool designButterworth(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs a Chebyshev Type I filter.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if successful, false otherwise.
     */
    bool designChebyshev1(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs a Chebyshev Type II filter.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if successful, false otherwise.
     */
    bool designChebyshev2(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs an Elliptic filter.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if successful, false otherwise.
     */
    bool designElliptic(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Designs a Bessel filter.
     *
     * @param[in] design    Filter design parameters.
     *
     * @return true if successful, false otherwise.
     */
    bool designBessel(const FilterDesign& design);

    //=========================================================================================================
    /**
     * Applies bilinear transformation to convert analog filter to digital.
     *
     * @param[in] s_num     Analog numerator coefficients.
     * @param[in] s_den     Analog denominator coefficients.
     * @param[in] fs        Sampling frequency.
     * @param[out] z_num    Digital numerator coefficients.
     * @param[out] z_den    Digital denominator coefficients.
     */
    void bilinearTransform(const Eigen::VectorXd& s_num, const Eigen::VectorXd& s_den,
                          double fs, Eigen::VectorXd& z_num, Eigen::VectorXd& z_den);

    //=========================================================================================================
    /**
     * Applies frequency transformation for different filter types.
     *
     * @param[in] design        Filter design parameters.
     * @param[in] prototype_num Prototype numerator coefficients.
     * @param[in] prototype_den Prototype denominator coefficients.
     * @param[out] transformed_num Transformed numerator coefficients.
     * @param[out] transformed_den Transformed denominator coefficients.
     */
    void frequencyTransform(const FilterDesign& design,
                           const Eigen::VectorXd& prototype_num,
                           const Eigen::VectorXd& prototype_den,
                           Eigen::VectorXd& transformed_num,
                           Eigen::VectorXd& transformed_den);

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
    FilterState         m_state;            /**< Filter state and coefficients */
    bool                m_isDesigned;       /**< Flag indicating if filter has been designed */
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // namespace RTPROCESSINGLIB

#endif // IIRFILTER_H