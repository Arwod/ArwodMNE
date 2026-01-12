#include <QtTest/QtTest>
#include <tfr/tfr_utils.h>
#include <tfr/psd.h>
#include <tfr/csd.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace TFRLIB;
using namespace Eigen;

class TestTFRMultitaper : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testDPSS();
    void testPSD();
    void testCSD();
};

void TestTFRMultitaper::initTestCase()
{
}

void TestTFRMultitaper::cleanupTestCase()
{
}

void TestTFRMultitaper::testDPSS()
{
    int N = 256;
    double NW = 4.0;
    int K = 4;
    
    std::pair<MatrixXd, VectorXd> dpss = TFRUtils::dpss_windows(N, NW, K);
    MatrixXd windows = dpss.first;
    
    QCOMPARE(windows.rows(), N);
    QCOMPARE(windows.cols(), K);
    
    // Check orthogonality: W^T * W = I
    MatrixXd gram = windows.transpose() * windows;
    MatrixXd I = MatrixXd::Identity(K, K);
    
    // Check if close to Identity
    double diff = (gram - I).norm();
    qDebug() << "DPSS Orthogonality Error:" << diff;
    QVERIFY(diff < 1e-10);
}

void TestTFRMultitaper::testPSD()
{
    // Generate Sine Wave: 10 Hz
    double sfreq = 100.0;
    int n_times = 1000;
    double freq = 10.0;
    
    MatrixXd data(1, n_times);
    for (int i = 0; i < n_times; ++i) {
        double t = (double)i / sfreq;
        data(0, i) = std::sin(2.0 * M_PI * freq * t);
    }
    
    // Compute PSD
    // bandwidth = 0 -> default 4/T = 4 / 10 = 0.4 Hz.
    auto res = PSD::psd_multitaper(data, sfreq);
    MatrixXd psds = res.first;
    VectorXd freqs = res.second;
    
    // Find peak
    int max_idx = 0;
    double max_val = 0.0;
    for (int i = 0; i < freqs.size(); ++i) {
        if (psds(0, i) > max_val) {
            max_val = psds(0, i);
            max_idx = i;
        }
    }
    
    double peak_freq = freqs[max_idx];
    qDebug() << "PSD Peak Freq:" << peak_freq;
    QVERIFY(std::abs(peak_freq - freq) < 0.5); 
}

void TestTFRMultitaper::testCSD()
{
    // Generate Correlated Sine Waves: 10 Hz
    double sfreq = 100.0;
    int n_times = 1000;
    double freq = 10.0;
    
    MatrixXd data(2, n_times);
    for (int i = 0; i < n_times; ++i) {
        double t = (double)i / sfreq;
        data(0, i) = std::sin(2.0 * M_PI * freq * t);
        data(1, i) = std::sin(2.0 * M_PI * freq * t + M_PI/2.0); // 90 deg phase shift
    }
    
    std::vector<MatrixXd> epochs;
    epochs.push_back(data);
    
    CSD csd = CSD::compute_multitaper(epochs, sfreq, 0, 0, 8.0, 12.0);
    
    // Find peak at 10 Hz
    int idx_10hz = -1;
    double min_diff = 100.0;
    
    for (int i = 0; i < csd.freqs.size(); ++i) {
        if (std::abs(csd.freqs[i] - 10.0) < min_diff) {
            min_diff = std::abs(csd.freqs[i] - 10.0);
            idx_10hz = i;
        }
    }
    
    QVERIFY(idx_10hz != -1);
    QVERIFY(min_diff < 0.5);
    
    MatrixXcd C = csd.data[idx_10hz];
    
    // C[0,0] and C[1,1] should be real and positive (auto-spectra)
    qDebug() << "C(0,0):" << C(0,0).real() << C(0,0).imag();
    QVERIFY(C(0,0).real() > 0.0);
    // Imag part might not be exactly 0 due to numerical error but should be small
    QVERIFY(std::abs(C(0,0).imag()) < 1e-2 * C(0,0).real());
    
    // C[0,1] should be imaginary (due to 90 deg phase shift)
    qDebug() << "C(0,1):" << C(0,1).real() << C(0,1).imag();
    
    // Ideally Real part is small compared to Imag part
    // C[0,1] magnitude should be significant
    double mag = std::abs(C(0,1));
    QVERIFY(mag > 0.1 * C(0,0).real()); // Significant coherence
    
    QVERIFY(std::abs(C(0,1).real()) < 0.5 * mag); // Mostly imaginary
}

QTEST_GUILESS_MAIN(TestTFRMultitaper)
#include "test_tfr_multitaper.moc"
