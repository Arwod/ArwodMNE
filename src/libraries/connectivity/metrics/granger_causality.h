//=============================================================================================================
/**
 * @file     granger_causality.h
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
 * @brief    GrangerCausality class declaration
 *
 */

#ifndef GRANGER_CAUSALITY_H
#define GRANGER_CAUSALITY_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "../connectivity_global.h"
#include "abstractmetric.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QSharedPointer>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>

//=============================================================================================================
// DEFINE NAMESPACE CONNECTIVITYLIB
//=============================================================================================================

namespace CONNECTIVITYLIB {

//=============================================================================================================
/**
 * This class computes Granger causality between time series.
 * Granger causality measures the extent to which past values of one time series
 * help predict future values of another time series.
 *
 * @brief This class computes Granger causality.
 */
class CONNECTIVITYSHARED_EXPORT GrangerCausality : public AbstractMetric
{

public:
    typedef QSharedPointer<GrangerCausality> SPtr;            /**< Shared pointer type for GrangerCausality. */
    typedef QSharedPointer<const GrangerCausality> ConstSPtr; /**< Const shared pointer type for GrangerCausality. */

    //=========================================================================================================
    /**
     * Constructs a GrangerCausality object.
     */
    explicit GrangerCausality();

    //=========================================================================================================
    /**
     * Computes Granger causality between time series.
     *
     * @param[in] matData        The input data matrix (channels x samples).
     * @param[in] matIndices     The indices of the channels to compute connectivity for.
     * @param[in] iNfft          The FFT length (not used for Granger causality).
     * @param[in] dSFreq         The sampling frequency.
     * @param[in] matPsd         The power spectral density (not used for Granger causality).
     *
     * @return The connectivity matrix.
     */
    static Eigen::MatrixXd calculate(const Eigen::MatrixXd& matData,
                                     const Eigen::MatrixXi& matIndices,
                                     int iNfft,
                                     double dSFreq,
                                     const Eigen::MatrixXd& matPsd = Eigen::MatrixXd());

    //=========================================================================================================
    /**
     * Computes Granger causality with specified model order.
     *
     * @param[in] matData        The input data matrix (channels x samples).
     * @param[in] matIndices     The indices of the channels to compute connectivity for.
     * @param[in] modelOrder     The autoregressive model order.
     * @param[in] dSFreq         The sampling frequency.
     *
     * @return The connectivity matrix.
     */
    static Eigen::MatrixXd calculateWithOrder(const Eigen::MatrixXd& matData,
                                               const Eigen::MatrixXi& matIndices,
                                               int modelOrder,
                                               double dSFreq);

    //=========================================================================================================
    /**
     * Computes pairwise Granger causality between two time series.
     *
     * @param[in] x              First time series.
     * @param[in] y              Second time series.
     * @param[in] modelOrder     The autoregressive model order.
     *
     * @return Granger causality value from x to y.
     */
    static double computePairwiseGC(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y,
                                    int modelOrder);

    //=========================================================================================================
    /**
     * Fits an autoregressive model using least squares.
     *
     * @param[in] data           Input time series data.
     * @param[in] order          Model order.
     * @param[out] coefficients  AR coefficients.
     * @param[out] residualVar   Residual variance.
     *
     * @return true if successful, false otherwise.
     */
    static bool fitARModel(const Eigen::VectorXd& data,
                           int order,
                           Eigen::VectorXd& coefficients,
                           double& residualVar);

    //=========================================================================================================
    /**
     * Fits a vector autoregressive (VAR) model.
     *
     * @param[in] data           Input multivariate time series (variables x samples).
     * @param[in] order          Model order.
     * @param[out] coefficients  VAR coefficients as vector of matrices.
     * @param[out] residualCov   Residual covariance matrix.
     *
     * @return true if successful, false otherwise.
     */
    static bool fitVARModel(const Eigen::MatrixXd& data,
                            int order,
                            QList<Eigen::MatrixXd>& coefficients,
                            Eigen::MatrixXd& residualCov);

    //=========================================================================================================
    /**
     * Determines optimal model order using information criteria.
     *
     * @param[in] data           Input time series data.
     * @param[in] maxOrder       Maximum order to test.
     * @param[in] criterion      Information criterion ("AIC", "BIC", "HQC").
     *
     * @return Optimal model order.
     */
    static int selectModelOrder(const Eigen::VectorXd& data,
                                int maxOrder = 20,
                                const QString& criterion = "AIC");

    //=========================================================================================================
    /**
     * Computes spectral Granger causality in frequency domain.
     *
     * @param[in] matData        The input data matrix (channels x samples).
     * @param[in] matIndices     The indices of the channels to compute connectivity for.
     * @param[in] modelOrder     The autoregressive model order.
     * @param[in] dSFreq         The sampling frequency.
     * @param[in] freqs          Frequency points for computation.
     *
     * @return The spectral Granger causality as list of matrices (one per frequency).
     */
    static QList<Eigen::MatrixXd> computeSpectralGC(const Eigen::MatrixXd& matData,
                                                     const Eigen::MatrixXi& matIndices,
                                                     int modelOrder,
                                                     double dSFreq,
                                                     const Eigen::VectorXd& freqs);

private:
    //=========================================================================================================
    /**
     * Computes Akaike Information Criterion.
     *
     * @param[in] logLikelihood  Log-likelihood value.
     * @param[in] numParams      Number of parameters.
     *
     * @return AIC value.
     */
    static double computeAIC(double logLikelihood, int numParams);

    //=========================================================================================================
    /**
     * Computes Bayesian Information Criterion.
     *
     * @param[in] logLikelihood  Log-likelihood value.
     * @param[in] numParams      Number of parameters.
     * @param[in] numSamples     Number of samples.
     *
     * @return BIC value.
     */
    static double computeBIC(double logLikelihood, int numParams, int numSamples);

    //=========================================================================================================
    /**
     * Creates design matrix for autoregressive model.
     *
     * @param[in] data           Input time series.
     * @param[in] order          Model order.
     *
     * @return Design matrix.
     */
    static Eigen::MatrixXd createDesignMatrix(const Eigen::VectorXd& data, int order);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // namespace CONNECTIVITYLIB

#endif // GRANGER_CAUSALITY_H