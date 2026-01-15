//=============================================================================================================
/**
 * @file     coherency.cpp
 * @author   Daniel Strohmeier <Daniel.Strohmeier@tu-ilmenau.de>;
 *           Lorenz Esch <lesch@mgh.harvard.edu>
 * @since    0.1.0
 * @date     April, 2018
 *
 * @section  LICENSE
 *
 * Copyright (C) 2018, Daniel Strohmeier, Lorenz Esch. All rights reserved.
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
 * @note Notes:
 * - Some of this code was adapted from mne-python (https://martinos.org/mne) with permission from Alexandre Gramfort.
 * - QtConcurrent can be used to speed up computation.
 *
 * @brief     Coherency class declaration.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "coherency.h"
#include "../network/networknode.h"
#include "../network/networkedge.h"
#include "../network/network.h"
#include "../connectivitysettings.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QtConcurrent>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <unsupported/Eigen/FFT>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace CONNECTIVITYLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE GLOBAL METHODS
//=============================================================================================================

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

Coherency::Coherency()
{
}

//=============================================================================================================

void Coherency::calculateAbs(Network &finalNetwork,
                             ConnectivitySettings &connectivitySettings)
{
    if (connectivitySettings.isEmpty())
    {
        qDebug() << "Coherency::calculateReal - Input data is empty";
        return;
    }

#ifdef EIGEN_FFTW_DEFAULT
    fftw_make_planner_thread_safe();
#endif

    int iSignalLength = connectivitySettings.at(0).matData.cols();
    int iNfft = connectivitySettings.getFFTSize();

    // Generate tapers
    std::pair<MatrixXd, VectorXd> tapers = AbstractSpectralMetric::generateTapers(
        iSignalLength, connectivitySettings.getWindowType());

    // Initialize vecPsdAvg and vecCsdAvg
    int iNRows = connectivitySettings.at(0).matData.rows();
    int iNFreqs = int(floor(iNfft / 2.0)) + 1;

    // Compute PSD/CSD for each trial
    QMutex mutex;

    std::function<void(ConnectivitySettings::IntermediateTrialData &)> computeLambda = [&](ConnectivitySettings::IntermediateTrialData &inputData)
    {
        compute(inputData,
                connectivitySettings.getIntermediateSumData().matPsdSum,
                connectivitySettings.getIntermediateSumData().vecPairCsdSum,
                mutex,
                iNRows,
                iNFreqs,
                iNfft,
                tapers);
    };

    QFuture<void> result = QtConcurrent::map(connectivitySettings.getTrialData(),
                                             computeLambda);
    result.waitForFinished();

    // Compute CSD/sqrt(PSD_X * PSD_Y)
    std::function<void(QPair<int, MatrixXcd> &)> computePSDCSDLambda = [&](QPair<int, MatrixXcd> &pairInput)
    {
        computePSDCSDAbs(mutex,
                         finalNetwork,
                         pairInput,
                         connectivitySettings.getIntermediateSumData().matPsdSum);
    };

    QFuture<void> resultCSDPSD = QtConcurrent::map(connectivitySettings.getIntermediateSumData().vecPairCsdSum,
                                                   computePSDCSDLambda);
    resultCSDPSD.waitForFinished();
}

//=============================================================================================================

void Coherency::calculateImag(Network &finalNetwork,
                              ConnectivitySettings &connectivitySettings)
{
    if (connectivitySettings.isEmpty())
    {
        qDebug() << "Coherency::calculateImag - Input data is empty";
        return;
    }

#ifdef EIGEN_FFTW_DEFAULT
    fftw_make_planner_thread_safe();
#endif

    int iSignalLength = connectivitySettings.at(0).matData.cols();
    int iNfft = connectivitySettings.getFFTSize();

    // Generate tapers
    std::pair<MatrixXd, VectorXd> tapers = AbstractSpectralMetric::generateTapers(
        iSignalLength, connectivitySettings.getWindowType());

    // Initialize vecPsdAvg and vecCsdAvg
    int iNRows = connectivitySettings.at(0).matData.rows();
    int iNFreqs = int(floor(iNfft / 2.0)) + 1;

    // Compute PSD/CSD for each trial
    QMutex mutex;

    std::function<void(ConnectivitySettings::IntermediateTrialData &)> computeLambda = [&](ConnectivitySettings::IntermediateTrialData &inputData)
    {
        compute(inputData,
                connectivitySettings.getIntermediateSumData().matPsdSum,
                connectivitySettings.getIntermediateSumData().vecPairCsdSum,
                mutex,
                iNRows,
                iNFreqs,
                iNfft,
                tapers);
    };

    QFuture<void> result = QtConcurrent::map(connectivitySettings.getTrialData(),
                                             computeLambda);
    result.waitForFinished();

    // Compute CSD/sqrt(PSD_X * PSD_Y)
    std::function<void(QPair<int, MatrixXcd> &)> computePSDCSDLambda = [&](QPair<int, MatrixXcd> &pairInput)
    {
        computePSDCSDImag(mutex,
                          finalNetwork,
                          pairInput,
                          connectivitySettings.getIntermediateSumData().matPsdSum);
    };

    QFuture<void> resultCSDPSD = QtConcurrent::map(connectivitySettings.getIntermediateSumData().vecPairCsdSum,
                                                   computePSDCSDLambda);
    resultCSDPSD.waitForFinished();
}

//=============================================================================================================

void Coherency::compute(ConnectivitySettings::IntermediateTrialData &inputData,
                        MatrixXd &matPsdSum,
                        QVector<QPair<int, MatrixXcd>> &vecPairCsdSum,
                        QMutex &mutex,
                        int iNRows,
                        int iNFreqs,
                        int iNfft,
                        const std::pair<Eigen::MatrixXd, Eigen::VectorXd> &tapers)
{
    if (inputData.vecPairCsd.size() == iNRows)
    {
        return;
    }

    // Use AbstractSpectralMetric to compute tapered spectra
    // Note: computeTaperedSpectra returns raw spectra (without weights)
    QVector<Eigen::MatrixXcd> vecTapSpectra = AbstractSpectralMetric::computeTaperedSpectra(
        inputData.matData, tapers.first, iNfft, iNFreqs, tapers.second);

    // Apply weights to spectra if necessary (to match old implementation)
    // Old implementation: matTapSpectrum.row(j) = vecTmpFreq * tapers.second(j);
    // So we multiply each taper k by weight[k]
    for (int i = 0; i < vecTapSpectra.size(); ++i)
    {
        for (int k = 0; k < vecTapSpectra[i].rows(); ++k)
        {
            vecTapSpectra[i].row(k) *= tapers.second(k);
        }
    }

    // Store if needed
    // The old code stored it in inputData.vecTapSpectra but mostly for caching.
    // We can store it if we want to be 100% compatible or if multiple passes need it.
    // For now, let's use the local var.

    // Compute PSD
    inputData.matPsd = MatrixXd(iNRows, m_iNumberBinAmount);
    double denomPSD = tapers.second.cwiseAbs2().sum() / 2.0;

    bool bNfftEven = (iNfft % 2 == 0);

    for (int i = 0; i < iNRows; ++i)
    {
        // Average over tapers
        inputData.matPsd.row(i) = vecTapSpectra[i].block(0, m_iNumberBinStart, vecTapSpectra[i].rows(), m_iNumberBinAmount).cwiseAbs2().colwise().sum() / denomPSD;

        // Divide first and last element by 2 due to half spectrum
        if (m_iNumberBinStart == 0)
        {
            inputData.matPsd.row(i)(0) /= 2.0;
        }

        if (bNfftEven && m_iNumberBinStart + m_iNumberBinAmount >= iNFreqs)
        {
            inputData.matPsd.row(i).tail(1) /= 2.0;
        }
    }

    mutex.lock();
    if (matPsdSum.rows() == 0 || matPsdSum.cols() == 0)
    {
        matPsdSum = inputData.matPsd;
    }
    else
    {
        matPsdSum += inputData.matPsd;
    }
    mutex.unlock();

    // Compute CSD
    inputData.vecPairCsd.clear();
    double denomCSD = denomPSD; // Same denom if weights are same (squared sum / 2)

    for (int i = 0; i < iNRows; ++i)
    {
        MatrixXcd matCsd = MatrixXcd::Zero(iNRows, m_iNumberBinAmount);

        for (int j = i; j < iNRows; ++j)
        {
            // Compute CSD (average over tapers)
            matCsd.row(j) = vecTapSpectra[i].block(0, m_iNumberBinStart, vecTapSpectra[i].rows(), m_iNumberBinAmount).cwiseProduct(vecTapSpectra[j].block(0, m_iNumberBinStart, vecTapSpectra[j].rows(), m_iNumberBinAmount).conjugate()).colwise().sum() / denomCSD;

            // Divide first and last element by 2 due to half spectrum
            if (m_iNumberBinStart == 0)
            {
                matCsd.row(j)(0) /= 2.0;
            }

            if (bNfftEven && m_iNumberBinStart + m_iNumberBinAmount >= iNFreqs)
            {
                matCsd.row(j).tail(1) /= 2.0;
            }
        }
        inputData.vecPairCsd.append(QPair<int, MatrixXcd>(i, matCsd));
    }

    mutex.lock();
    if (vecPairCsdSum.isEmpty())
    {
        vecPairCsdSum = inputData.vecPairCsd;
    }
    else
    {
        for (int j = 0; j < vecPairCsdSum.size(); ++j)
        {
            vecPairCsdSum[j].second += inputData.vecPairCsd.at(j).second;
        }
    }
    mutex.unlock();

    // Do not store data to save memory
    if (!m_bStorageModeIsActive)
    {
        inputData.vecPairCsd.clear();
        inputData.vecTapSpectra.clear(); // We didn't fill it, but if we did...
    }
}

//=============================================================================================================

void Coherency::computePSDCSDAbs(QMutex &mutex,
                                 Network &finalNetwork,
                                 const QPair<int, MatrixXcd> &pairInput,
                                 const MatrixXd &matPsdSum)
{
    MatrixXd matPSDtmp(matPsdSum.rows(), matPsdSum.cols());
    RowVectorXd rowPsdSum = matPsdSum.row(pairInput.first);

    for (int j = 0; j < matPSDtmp.rows(); ++j)
    {
        matPSDtmp.row(j) = rowPsdSum.cwiseProduct(matPsdSum.row(j));
    }

    // Average. Note that the number of trials cancel each other out.
    MatrixXcd matCohy = pairInput.second.cwiseQuotient(matPSDtmp.cwiseSqrt());

    QSharedPointer<NetworkEdge> pEdge;
    MatrixXd matWeight;
    int j;
    int i = pairInput.first;

    for (j = i; j < matCohy.rows(); ++j)
    {
        matWeight = matCohy.row(j).cwiseAbs().transpose();
        pEdge = QSharedPointer<NetworkEdge>(new NetworkEdge(i, j, matWeight));

        mutex.lock();
        finalNetwork.getNodeAt(i)->append(pEdge);
        finalNetwork.getNodeAt(j)->append(pEdge);
        finalNetwork.append(pEdge);
        mutex.unlock();
    }
}

//=============================================================================================================

void Coherency::computePSDCSDImag(QMutex &mutex,
                                  Network &finalNetwork,
                                  const QPair<int, MatrixXcd> &pairInput,
                                  const MatrixXd &matPsdSum)
{
    MatrixXd matPSDtmp(matPsdSum.rows(), matPsdSum.cols());
    RowVectorXd rowPsdSum = matPsdSum.row(pairInput.first);

    for (int j = 0; j < matPSDtmp.rows(); ++j)
    {
        matPSDtmp.row(j) = rowPsdSum.cwiseProduct(matPsdSum.row(j));
    }

    MatrixXcd matCohy = pairInput.second.cwiseQuotient(matPSDtmp.cwiseSqrt());

    QSharedPointer<NetworkEdge> pEdge;
    MatrixXd matWeight;
    int j;
    int i = pairInput.first;

    for (j = i; j < matCohy.rows(); ++j)
    {
        matWeight = matCohy.row(j).imag().transpose();
        pEdge = QSharedPointer<NetworkEdge>(new NetworkEdge(i, j, matWeight));

        mutex.lock();
        finalNetwork.getNodeAt(i)->append(pEdge);
        finalNetwork.getNodeAt(j)->append(pEdge);
        finalNetwork.append(pEdge);
        mutex.unlock();
    }
}
