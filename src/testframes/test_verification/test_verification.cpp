//=============================================================================================================
/**
 * @file     test_verification.cpp
 * @author   MNE-Porting-Bot
 * @since    0.1.0
 * @date     January, 2026
 *
 * @section  LICENSE
 *
 * Copyright (C) 2026. All rights reserved.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/ioutils.h>
#include <utils/mnemath.h>
#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QtTest>
#include <QFile>
#include <QCoreApplication>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace UTILSLIB;
using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestVerification
 *
 * @brief Verification tests for MNE porting
 *
 */
class TestVerification : public QObject
{
    Q_OBJECT

public:
    TestVerification();

private slots:
    void initTestCase();
    void verifySignalLoad();
    void verifyHilbert();
    void verifyConvolution();
    void verifyWindows();
    void cleanupTestCase();

private:
    double dEpsilon;
    MatrixXd mSignalRaw;
    MatrixXd mSignalHilbertAbs;
    MatrixXd mSignalConv;
    MatrixXd mWinHanning;
    MatrixXd mWinHamming;
    MatrixXd mWinBlackman;

    QString dataPath;
};

//=============================================================================================================

TestVerification::TestVerification()
    : dEpsilon(0.000001)
{
}

//=============================================================================================================

void TestVerification::initTestCase()
{
    // Assume data is in ../tests/verification_data relative to where we run or source root
    // For now, let's look relative to source dir if possible, or build dir
    // The build usually puts binaries in out/Release/apps/
    // We generated data in ArwodMNE/data/ via python script

    // Check possible locations
    QStringList possiblePaths;
    possiblePaths << QCoreApplication::applicationDirPath() + "/../../../../../data"; // If binary is deep in out/Release/apps/test_verification
    possiblePaths << QCoreApplication::applicationDirPath() + "/../../../data";
    possiblePaths << QCoreApplication::applicationDirPath() + "/../data";
    possiblePaths << "../../../data";
    possiblePaths << "../data";
    possiblePaths << "data";

    // In CI/Dev environment, data was generated in ArwodMNE/data (based on python script default)
    // The python script OUTPUT_DIR = "data" relative to where it ran.
    // We ran it in ArwodMNE/ directory. So ArwodMNE/data/signal_raw.txt exists.

    bool found = false;
    for (const QString &path : possiblePaths)
    {
        if (QFile::exists(path + "/signal_raw.txt"))
        {
            dataPath = path;
            found = true;
            break;
        }
    }

    if (!found)
    {
        // Fallback: Try absolute path based on known env
        dataPath = "/Users/eric/Public/work/code/mne-project/ArwodMNE/data";
    }

    qDebug() << "Data Path:" << dataPath;

    // Load Data
    QVERIFY(IOUtils::read_eigen_matrix(mSignalRaw, dataPath + "/signal_raw.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(mSignalHilbertAbs, dataPath + "/signal_hilbert_abs.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(mSignalConv, dataPath + "/signal_conv_ma50.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(mWinHanning, dataPath + "/window_hanning_512.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(mWinHamming, dataPath + "/window_hamming_512.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(mWinBlackman, dataPath + "/window_blackman_512.txt"));

    // Ensure data is vectors (N x 1)
    if (mSignalRaw.cols() > 1 && mSignalRaw.rows() == 1)
        mSignalRaw.transposeInPlace();
    if (mSignalHilbertAbs.cols() > 1 && mSignalHilbertAbs.rows() == 1)
        mSignalHilbertAbs.transposeInPlace();
    if (mSignalConv.cols() > 1 && mSignalConv.rows() == 1)
        mSignalConv.transposeInPlace();
    if (mWinHanning.cols() > 1 && mWinHanning.rows() == 1)
        mWinHanning.transposeInPlace();
    if (mWinHamming.cols() > 1 && mWinHamming.rows() == 1)
        mWinHamming.transposeInPlace();
    if (mWinBlackman.cols() > 1 && mWinBlackman.rows() == 1)
        mWinBlackman.transposeInPlace();

    qDebug() << "Loaded Signal Raw:" << mSignalRaw.rows() << "x" << mSignalRaw.cols();
}

//=============================================================================================================

void TestVerification::verifySignalLoad()
{
    QVERIFY(mSignalRaw.size() > 0);
    QVERIFY(mSignalHilbertAbs.size() > 0);
    QCOMPARE(mSignalRaw.size(), mSignalHilbertAbs.size());
}

//=============================================================================================================

void TestVerification::verifyHilbert()
{
    // Use MNEMath implementation
    MatrixXcd analytic_signal = MNEMath::hilbert(mSignalRaw);

    // Compare Amplitude Envelope (abs)
    double max_diff = 0.0;
    int N = mSignalRaw.rows();

    for (int i = 0; i < N; ++i)
    {
        double abs_val = std::abs(analytic_signal(i, 0));
        double ref_val = mSignalHilbertAbs(i, 0);
        double diff = std::abs(abs_val - ref_val);
        if (diff > max_diff)
            max_diff = diff;
    }

    qDebug() << "Max Difference Hilbert:" << max_diff;

    // Relaxed tolerance for float/double diffs and different FFT implementations
    QVERIFY(max_diff < 1e-5);
}

//=============================================================================================================

void TestVerification::verifyConvolution()
{
    // Simple Moving Average Kernel (same as python script)
    int win_size = 50;
    VectorXd kernel = VectorXd::Ones(win_size) / (double)win_size;
    VectorXd signal = mSignalRaw.col(0);

    // Convolve "same" mode
    VectorXd conv_res = MNEMath::convolve(signal, kernel, QString("same"));

    QCOMPARE(conv_res.size(), mSignalConv.size());

    double max_diff = 0.0;
    for (int i = 0; i < conv_res.size(); ++i)
    {
        double diff = std::abs(conv_res[i] - mSignalConv(i, 0));
        if (diff > max_diff)
            max_diff = diff;
    }

    qDebug() << "Max Difference Convolution:" << max_diff;
    QVERIFY(max_diff < 1e-5);
}

//=============================================================================================================

void TestVerification::verifyWindows()
{
    int n = 512;

    // Hanning
    VectorXd hanning = MNEMath::hanning(n);
    QCOMPARE(hanning.size(), mWinHanning.size());
    double max_diff_hanning = (hanning - mWinHanning.col(0)).cwiseAbs().maxCoeff();
    qDebug() << "Max Difference Hanning:" << max_diff_hanning;
    QVERIFY(max_diff_hanning < 1e-7);

    // Hamming
    VectorXd hamming = MNEMath::hamming(n);
    QCOMPARE(hamming.size(), mWinHamming.size());
    double max_diff_hamming = (hamming - mWinHamming.col(0)).cwiseAbs().maxCoeff();
    qDebug() << "Max Difference Hamming:" << max_diff_hamming;
    QVERIFY(max_diff_hamming < 1e-7);

    // Blackman
    VectorXd blackman = MNEMath::blackman(n);
    QCOMPARE(blackman.size(), mWinBlackman.size());
    double max_diff_blackman = (blackman - mWinBlackman.col(0)).cwiseAbs().maxCoeff();
    qDebug() << "Max Difference Blackman:" << max_diff_blackman;
    QVERIFY(max_diff_blackman < 1e-7);
}

//=============================================================================================================

void TestVerification::cleanupTestCase()
{
}

//=============================================================================================================
// MAIN
//=============================================================================================================

QTEST_GUILESS_MAIN(TestVerification)
#include "test_verification.moc"
