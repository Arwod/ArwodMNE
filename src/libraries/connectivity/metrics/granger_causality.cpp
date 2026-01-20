//=============================================================================================================
/**
 * @file     granger_causality.cpp
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
 * @brief    GrangerCausality class definition.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "granger_causality.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QtMath>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Cholesky>
#include <unsupported/Eigen/FFT>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace CONNECTIVITYLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

GrangerCausality::GrangerCausality()
{
}

//=============================================================================================================

MatrixXd GrangerCausality::calculate(const MatrixXd& matData,
                                      const MatrixXi& matIndices,
                                      int iNfft,
                                      double dSFreq,
                                      const MatrixXd& matPsd)
{
    Q_UNUSED(iNfft)
    Q_UNUSED(matPsd)
    
    // Use default model order selection
    int modelOrder = 10; // Default order, could be optimized
    
    return calculateWithOrder(matData, matIndices, modelOrder, dSFreq);
}

//=============================================================================================================

MatrixXd GrangerCausality::calculateWithOrder(const MatrixXd& matData,
                                               const MatrixXi& matIndices,
                                               int modelOrder,
                                               double dSFreq)
{
    Q_UNUSED(dSFreq)
    
    int nChannels = matIndices.rows();
    MatrixXd connectivityMatrix = MatrixXd::Zero(nChannels, nChannels);
    
    // Compute pairwise Granger causality
    for (int i = 0; i < nChannels; ++i) {
        for (int j = 0; j < nChannels; ++j) {
            if (i != j) {
                int idx1 = matIndices(i, 0);
                int idx2 = matIndices(j, 0);
                
                if (idx1 < matData.rows() && idx2 < matData.rows()) {
                    VectorXd x = matData.row(idx1);
                    VectorXd y = matData.row(idx2);
                    
                    double gcValue = computePairwiseGC(x, y, modelOrder);
                    connectivityMatrix(i, j) = gcValue;
                }
            }
        }
    }
    
    return connectivityMatrix;
}

//=============================================================================================================

double GrangerCausality::computePairwiseGC(const VectorXd& x,
                                            const VectorXd& y,
                                            int modelOrder)
{
    if (x.size() != y.size() || x.size() <= modelOrder) {
        qWarning() << "Invalid input for Granger causality computation";
        return 0.0;
    }
    
    // Fit unrestricted VAR model (bivariate)
    MatrixXd bivariateData(2, x.size());
    bivariateData.row(0) = x;
    bivariateData.row(1) = y;
    
    // Create design matrix for VAR model
    int nSamples = x.size() - modelOrder;
    MatrixXd X(nSamples, 2 * modelOrder + 1); // +1 for intercept
    VectorXd Y = y.segment(modelOrder, nSamples);
    
    // Fill design matrix
    X.col(0).setOnes(); // Intercept
    
    for (int lag = 1; lag <= modelOrder; ++lag) {
        X.col(2 * lag - 1) = x.segment(modelOrder - lag, nSamples);
        X.col(2 * lag) = y.segment(modelOrder - lag, nSamples);
    }
    
    // Fit unrestricted model: y(t) = c + sum(a_i * x(t-i)) + sum(b_i * y(t-i)) + e(t)
    VectorXd coeffsUnrestricted = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
    VectorXd residualsUnrestricted = Y - X * coeffsUnrestricted;
    double rssUnrestricted = residualsUnrestricted.squaredNorm();
    
    // Fit restricted model: y(t) = c + sum(b_i * y(t-i)) + e(t)
    MatrixXd XRestricted(nSamples, modelOrder + 1);
    XRestricted.col(0).setOnes(); // Intercept
    
    for (int lag = 1; lag <= modelOrder; ++lag) {
        XRestricted.col(lag) = y.segment(modelOrder - lag, nSamples);
    }
    
    VectorXd coeffsRestricted = (XRestricted.transpose() * XRestricted).ldlt().solve(XRestricted.transpose() * Y);
    VectorXd residualsRestricted = Y - XRestricted * coeffsRestricted;
    double rssRestricted = residualsRestricted.squaredNorm();
    
    // Compute Granger causality
    if (rssUnrestricted > 0 && rssRestricted > rssUnrestricted) {
        return std::log(rssRestricted / rssUnrestricted);
    }
    
    return 0.0;
}
//=============================================================================================================

bool GrangerCausality::fitARModel(const VectorXd& data,
                                   int order,
                                   VectorXd& coefficients,
                                   double& residualVar)
{
    if (data.size() <= order) {
        return false;
    }
    
    int nSamples = data.size() - order;
    MatrixXd X = createDesignMatrix(data, order);
    VectorXd y = data.segment(order, nSamples);
    
    // Solve least squares: X * coefficients = y
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    
    // Compute residual variance
    VectorXd residuals = y - X * coefficients;
    residualVar = residuals.squaredNorm() / (nSamples - order - 1);
    
    return true;
}

//=============================================================================================================

int GrangerCausality::selectModelOrder(const VectorXd& data,
                                        int maxOrder,
                                        const QString& criterion)
{
    if (data.size() <= maxOrder) {
        return std::min(static_cast<int>(data.size() / 4), 10);
    }
    
    double bestCriterion = std::numeric_limits<double>::max();
    int bestOrder = 1;
    
    for (int order = 1; order <= maxOrder; ++order) {
        VectorXd coefficients;
        double residualVar;
        
        if (!fitARModel(data, order, coefficients, residualVar)) {
            continue;
        }
        
        int nSamples = data.size() - order;
        double logLikelihood = -0.5 * nSamples * (std::log(2.0 * M_PI * residualVar) + 1.0);
        
        double criterionValue;
        if (criterion == "AIC") {
            criterionValue = computeAIC(logLikelihood, order + 1); // +1 for intercept
        } else if (criterion == "BIC") {
            criterionValue = computeBIC(logLikelihood, order + 1, nSamples);
        } else {
            // Default to AIC
            criterionValue = computeAIC(logLikelihood, order + 1);
        }
        
        if (criterionValue < bestCriterion) {
            bestCriterion = criterionValue;
            bestOrder = order;
        }
    }
    
    return bestOrder;
}

//=============================================================================================================

double GrangerCausality::computeAIC(double logLikelihood, int numParams)
{
    return -2.0 * logLikelihood + 2.0 * numParams;
}

//=============================================================================================================

double GrangerCausality::computeBIC(double logLikelihood, int numParams, int numSamples)
{
    return -2.0 * logLikelihood + numParams * std::log(numSamples);
}

//=============================================================================================================

MatrixXd GrangerCausality::createDesignMatrix(const VectorXd& data, int order)
{
    int nSamples = data.size() - order;
    MatrixXd X(nSamples, order + 1); // +1 for intercept
    
    X.col(0).setOnes(); // Intercept
    
    for (int lag = 1; lag <= order; ++lag) {
        X.col(lag) = data.segment(order - lag, nSamples);
    }
    
    return X;
}