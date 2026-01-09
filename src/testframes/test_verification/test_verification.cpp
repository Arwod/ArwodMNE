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
#include <tfr/tfr_utils.h>
#include <tfr/tfr_compute.h>
#include <tfr/psd.h>
#include <preprocessing/pca.h>
#include <preprocessing/fastica.h>
#include <preprocessing/ica.h>
#include <stats/ttest.h>
#include <stats/correction.h>
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
    void verifyMorlet();
    void verifyTFR();
    void verifyPSD();
    void verifyICA();
    void verifyStats();
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

void TestVerification::verifyMorlet()
{
    double sfreq = 1000.0;
    VectorXd freqs(2);
    freqs << 10.0, 20.0;
    double n_cycles = 5.0;

    std::vector<VectorXcd> wavelets = TFRLIB::TFRUtils::morlet(sfreq, freqs, n_cycles);

    QCOMPARE(wavelets.size(), 2);

    // 10Hz
    MatrixXd w10_real, w10_imag;
    QVERIFY(IOUtils::read_eigen_matrix(w10_real, dataPath + "/tfr_morlet_w_10hz_real.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(w10_imag, dataPath + "/tfr_morlet_w_10hz_imag.txt"));

    if (w10_real.cols() > 1 && w10_real.rows() == 1)
        w10_real.transposeInPlace();
    if (w10_imag.cols() > 1 && w10_imag.rows() == 1)
        w10_imag.transposeInPlace();

    VectorXcd w10_ref(w10_real.rows());
    for (int i = 0; i < w10_real.rows(); ++i)
        w10_ref[i] = std::complex<double>(w10_real(i, 0), w10_imag(i, 0));

    QCOMPARE(wavelets[0].size(), w10_ref.size());
    double diff10 = (wavelets[0] - w10_ref).norm();
    qDebug() << "Diff Morlet 10Hz:" << diff10;
    QVERIFY(diff10 < 1e-5);

    // 20Hz
    MatrixXd w20_real, w20_imag;
    QVERIFY(IOUtils::read_eigen_matrix(w20_real, dataPath + "/tfr_morlet_w_20hz_real.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(w20_imag, dataPath + "/tfr_morlet_w_20hz_imag.txt"));

    if (w20_real.cols() > 1 && w20_real.rows() == 1)
        w20_real.transposeInPlace();
    if (w20_imag.cols() > 1 && w20_imag.rows() == 1)
        w20_imag.transposeInPlace();

    VectorXcd w20_ref(w20_real.rows());
    for (int i = 0; i < w20_real.rows(); ++i)
        w20_ref[i] = std::complex<double>(w20_real(i, 0), w20_imag(i, 0));

    QCOMPARE(wavelets[1].size(), w20_ref.size());
    double diff20 = (wavelets[1] - w20_ref).norm();
    qDebug() << "Diff Morlet 20Hz:" << diff20;
    QVERIFY(diff20 < 1e-5);
}

//=============================================================================================================

void TestVerification::verifyTFR()
{
    // Load input signal (1 channel)
    double sfreq = 1000.0;
    VectorXd freqs(2);
    freqs << 10.0, 20.0;
    double n_cycles = 5.0;

    // tfr_morlet expects MatrixXd (channels x times)
    // mSignalRaw is (times x 1), transpose it
    MatrixXd input_data = mSignalRaw.transpose();

    auto power = TFRLIB::TFRCompute::tfr_morlet(input_data, sfreq, freqs, n_cycles);

    QCOMPARE(power.size(), 1);    // 1 channel
    QCOMPARE(power[0].size(), 2); // 2 freqs

    // Verify 10Hz Power
    MatrixXd p10_ref;
    QVERIFY(IOUtils::read_eigen_matrix(p10_ref, dataPath + "/tfr_power_10hz.txt"));
    if (p10_ref.rows() == 1)
        p10_ref.transposeInPlace();

    QCOMPARE(power[0][0].size(), p10_ref.size());

    // Compare center to avoid edge effects (wavelet is wide)
    int n_samples = power[0][0].size();
    int crop = n_samples / 4; // Crop 25% from each side
    int len = n_samples - 2 * crop;

    double diff10 = (power[0][0].segment(crop, len) - p10_ref.col(0).segment(crop, len)).norm();
    qDebug() << "Diff TFR 10Hz (Center):" << diff10;
    // TFR differences can be slightly larger due to convolution edge effects or precision
    QVERIFY(diff10 < 1e-3);

    // Verify 20Hz Power
    MatrixXd p20_ref;
    QVERIFY(IOUtils::read_eigen_matrix(p20_ref, dataPath + "/tfr_power_20hz.txt"));
    if (p20_ref.rows() == 1)
        p20_ref.transposeInPlace();

    double diff20 = (power[0][1].segment(crop, len) - p20_ref.col(0).segment(crop, len)).norm();
    qDebug() << "Diff TFR 20Hz (Center):" << diff20;
    QVERIFY(diff20 < 1e-3);
}

//=============================================================================================================

void TestVerification::verifyPSD()
{
    // Load input (transposed to 1xN)
    MatrixXd input_data = mSignalRaw.transpose();
    double sfreq = 1000.0;
    int n_fft = 256;

    // Compute PSD
    auto res = TFRLIB::PSD::psd_welch(input_data, sfreq, n_fft, 0, n_fft, "hamming");

    MatrixXd psds = res.first;
    VectorXd freqs = res.second;

    // Verify Freqs
    MatrixXd freqs_ref_mat;
    QVERIFY(IOUtils::read_eigen_matrix(freqs_ref_mat, dataPath + "/psd_welch_freqs.txt"));
    // freqs_ref_mat likely Nx1 or 1xN
    if (freqs_ref_mat.cols() > 1)
        freqs_ref_mat.transposeInPlace();
    VectorXd freqs_ref = freqs_ref_mat.col(0);

    QCOMPARE(freqs.size(), freqs_ref.size());
    double diff_freqs = (freqs - freqs_ref).norm();
    qDebug() << "Diff PSD Freqs:" << diff_freqs;
    QVERIFY(diff_freqs < 1e-5);

    // Verify PSDs
    MatrixXd psds_ref_mat;
    QVERIFY(IOUtils::read_eigen_matrix(psds_ref_mat, dataPath + "/psd_welch_psds.txt"));
    if (psds_ref_mat.cols() > 1)
        psds_ref_mat.transposeInPlace();
    VectorXd psds_ref = psds_ref_mat.col(0);

    QCOMPARE(psds.cols(), psds_ref.size());

    // psds is (n_channels x n_freqs) -> (1 x n_freqs)
    VectorXd psds_calc = psds.row(0);

    double diff_psds = (psds_calc - psds_ref).norm();
    qDebug() << "Diff PSDs:" << diff_psds;
    
    QVERIFY(diff_psds < 5e-4);
}

//=============================================================================================================

void TestVerification::verifyICA()
{
    // Load mixed signal
    MatrixXd X_mne; // (n_channels, n_samples)
    QVERIFY(IOUtils::read_eigen_matrix(X_mne, dataPath + "/ica_mixed_signal.txt"));
    if (X_mne.rows() > X_mne.cols()) X_mne.transposeInPlace(); // Ensure 3x2000

    // Load sklearn results
    MatrixXd S_sklearn; // (n_samples, n_components) -> need to transpose to (n_components, n_samples)
    QVERIFY(IOUtils::read_eigen_matrix(S_sklearn, dataPath + "/ica_sklearn_sources.txt"));
    if (S_sklearn.rows() > S_sklearn.cols()) S_sklearn.transposeInPlace();
    
    // Run C++ ICA
    PREPROCESSINGLIB::ICA ica(3, "fastica", "logcosh", 0); // Random state 0
    ica.fit(X_mne);
    
    MatrixXd S_cpp = ica.get_sources(X_mne); // (3, 2000)
    
    QCOMPARE(S_cpp.rows(), 3);
    QCOMPARE(S_cpp.cols(), X_mne.cols());
    
    // Compare Sources
    // ICA sources can be permuted and sign-flipped.
    // Check correlation between each C++ source and Sklearn source.
    
    for (int i = 0; i < 3; ++i) {
        VectorXd s_cpp = S_cpp.row(i);
        // Find best match in sklearn sources
        double max_corr = 0.0;
        
        for (int j = 0; j < 3; ++j) {
            VectorXd s_sk = S_sklearn.row(j);
            
            // Pearson correlation
            double mean_cpp = s_cpp.mean();
            double mean_sk = s_sk.mean();
            
            VectorXd centered_cpp = s_cpp.array() - mean_cpp;
            VectorXd centered_sk = s_sk.array() - mean_sk;
            
            double corr = centered_cpp.dot(centered_sk) / (centered_cpp.norm() * centered_sk.norm());
            if (std::abs(corr) > max_corr) {
                max_corr = std::abs(corr);
            }
        }
        
        qDebug() << "ICA Source" << i << "Max Correlation:" << max_corr;
        QVERIFY(max_corr > 0.95); // Should be very high
    }
    
    // Verify Unmixing Matrix?
    // A_inv = W. If A is estimated correctly, W * A_true should be Permutation * Scale.
    // But let's rely on sources correlation.
}

//=============================================================================================================

void TestVerification::verifyStats()
{
    // 1. T-Test 1-Sample
    MatrixXd data_1samp;
    QVERIFY(IOUtils::read_eigen_matrix(data_1samp, dataPath + "/stats_ttest_1samp_data.txt"));
    
    auto res_1samp = STATSLIB::TTest::ttest_1samp(data_1samp);
    VectorXd t_calc = res_1samp.first;
    VectorXd p_calc = res_1samp.second;
    
    MatrixXd t_ref_mat, p_ref_mat;
    QVERIFY(IOUtils::read_eigen_matrix(t_ref_mat, dataPath + "/stats_ttest_1samp_t.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(p_ref_mat, dataPath + "/stats_ttest_1samp_p.txt"));
    
    // Check T-values
    double diff_t = (t_calc - t_ref_mat.row(0)).norm();
    qDebug() << "Diff T-Test 1Samp T:" << diff_t;
    QVERIFY(diff_t < 1e-4);
    
    // Check P-values (approx)
    double diff_p = (p_calc - p_ref_mat.row(0)).norm();
    qDebug() << "Diff T-Test 1Samp P:" << diff_p;
    QVERIFY(diff_p < 0.05); // Looser tolerance for approximation
    
    // 2. T-Test Ind
    MatrixXd data_a, data_b;
    QVERIFY(IOUtils::read_eigen_matrix(data_a, dataPath + "/stats_ttest_ind_data_a.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(data_b, dataPath + "/stats_ttest_ind_data_b.txt"));
    
    auto res_ind = STATSLIB::TTest::ttest_ind(data_a, data_b);
    t_calc = res_ind.first;
    p_calc = res_ind.second;
    
    QVERIFY(IOUtils::read_eigen_matrix(t_ref_mat, dataPath + "/stats_ttest_ind_t.txt"));
    QVERIFY(IOUtils::read_eigen_matrix(p_ref_mat, dataPath + "/stats_ttest_ind_p.txt"));
    
    diff_t = (t_calc - t_ref_mat.row(0)).norm();
    qDebug() << "Diff T-Test Ind T:" << diff_t;
    QVERIFY(diff_t < 1e-4);
    
    diff_p = (p_calc - p_ref_mat.row(0)).norm();
    qDebug() << "Diff T-Test Ind P:" << diff_p;
    QVERIFY(diff_p < 0.05);
    
    // 3. Correction
    MatrixXd p_vals_mat;
    QVERIFY(IOUtils::read_eigen_matrix(p_vals_mat, dataPath + "/stats_p_values.txt"));
    VectorXd p_vals = p_vals_mat.row(0); // Assuming saved as 1xN or Nx1, usually read as Matrix
    if (p_vals_mat.rows() > 1) p_vals = p_vals_mat.col(0);
    
    // Bonferroni
    VectorXd bonf_calc = STATSLIB::Correction::bonferroni(p_vals);
    MatrixXd bonf_ref_mat;
    QVERIFY(IOUtils::read_eigen_matrix(bonf_ref_mat, dataPath + "/stats_bonferroni.txt"));
    
    double diff_bonf = (bonf_calc - bonf_ref_mat.row(0).transpose()).norm(); 
    // Wait, read_eigen_matrix might read 1D array as 1xN or Nx1 depending on file format
    // numpy savetxt 1D saves as column usually? No, rows.
    // Let's check dimensions.
    if (bonf_ref_mat.rows() != bonf_calc.rows()) bonf_ref_mat.transposeInPlace();
    diff_bonf = (bonf_calc - bonf_ref_mat.col(0)).norm();
    
    qDebug() << "Diff Bonferroni:" << diff_bonf;
    QVERIFY(diff_bonf < 1e-5);
    
    // FDR
    VectorXd fdr_calc = STATSLIB::Correction::fdr(p_vals);
    MatrixXd fdr_ref_mat;
    QVERIFY(IOUtils::read_eigen_matrix(fdr_ref_mat, dataPath + "/stats_fdr.txt"));
    if (fdr_ref_mat.rows() != fdr_calc.rows()) fdr_ref_mat.transposeInPlace();
    
    double diff_fdr = (fdr_calc - fdr_ref_mat.col(0)).norm();
    qDebug() << "Diff FDR:" << diff_fdr;
    QVERIFY(diff_fdr < 1e-5);
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
