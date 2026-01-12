#include <QtTest/QtTest>
#include <inverse/beamformer/dics.h>
#include <tfr/csd.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace INVERSELIB;
using namespace TFRLIB;
using namespace Eigen;

class TestDICS : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testDICSCompute();
};

void TestDICS::initTestCase()
{
}

void TestDICS::cleanupTestCase()
{
}

void TestDICS::testDICSCompute()
{
    // 1. Simulation Setup
    int n_channels = 4;
    int n_sources = 5;
    int n_times = 1000;
    double sfreq = 100.0;
    
    // Leadfield: Identity for first 4 sources (each channel sees one source perfectly), 5th source seen by all.
    // To make it interesting, let's use a random mixing matrix but ensure separability.
    // For simplicity: Diagonal Leadfield (Ch i sees Src i).
    // Src 0 -> Ch 0
    // Src 1 -> Ch 1
    // ...
    // Src 4 -> All Chs (0.5)
    
    MatrixXd leadfield = MatrixXd::Zero(n_channels, n_sources); // 1 orientation
    for(int i=0; i<n_channels; ++i) leadfield(i, i) = 1.0;
    leadfield.col(n_channels).setConstant(0.5); // Source 4 is 'deep' source
    
    // 2. Generate Data
    // Source 0: 10 Hz Sine
    // Source 1: 20 Hz Sine
    // Others: Noise
    
    MatrixXd S(n_sources, n_times);
    S.setZero();
    
    for (int t = 0; t < n_times; ++t) {
        double time = t / sfreq;
        S(0, t) = std::sin(2.0 * M_PI * 10.0 * time); // 10 Hz
        S(1, t) = std::sin(2.0 * M_PI * 20.0 * time); // 20 Hz
        // Add some noise to all
        for(int i=0; i<n_sources; ++i) S(i, t) += 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    MatrixXd X = leadfield * S; // (Ch x Times)
    
    // 3. Compute CSD
    std::vector<MatrixXd> epochs;
    epochs.push_back(X);
    
    // Compute CSD for 8-22 Hz
    CSD csd = CSD::compute_multitaper(epochs, sfreq, 0, 0, 8.0, 22.0);
    
    // 4. Compute DICS Power
    // Use regularization 0.05
    MatrixXd power = DICS::compute_source_power(leadfield, csd, 0.05, 1, true);
    
    // 5. Verify
    // Find index for ~10 Hz and ~20 Hz
    int idx_10 = -1, idx_20 = -1;
    double min_d10 = 100, min_d20 = 100;
    
    for(int i=0; i<csd.freqs.size(); ++i) {
        if(std::abs(csd.freqs[i] - 10.0) < min_d10) { min_d10 = std::abs(csd.freqs[i] - 10.0); idx_10 = i; }
        if(std::abs(csd.freqs[i] - 20.0) < min_d20) { min_d20 = std::abs(csd.freqs[i] - 20.0); idx_20 = i; }
    }
    
    qDebug() << "Freq 10Hz idx:" << idx_10 << "Val:" << csd.freqs[idx_10];
    qDebug() << "Freq 20Hz idx:" << idx_20 << "Val:" << csd.freqs[idx_20];
    
    // Check Source 0 Power at 10 Hz
    // Should be max among all sources (except maybe Src 4 which leaks)
    // Actually Src 0 maps only to Ch 0.
    // DICS filter for Src 0 should pick Ch 0.
    
    VectorXd p_at_10 = power.col(idx_10);
    qDebug() << "Power at 10Hz:" << p_at_10[0] << p_at_10[1] << p_at_10[2] << p_at_10[3] << p_at_10[4];
    
    // Source 0 should have high power
    QVERIFY(p_at_10[0] > p_at_10[2]); // Src 0 > Noise Src 2
    QVERIFY(p_at_10[0] > p_at_10[3]);
    
    // Check Source 1 Power at 20 Hz
    VectorXd p_at_20 = power.col(idx_20);
    qDebug() << "Power at 20Hz:" << p_at_20[0] << p_at_20[1] << p_at_20[2] << p_at_20[3] << p_at_20[4];
    
    QVERIFY(p_at_20[1] > p_at_20[0]); // Src 1 > Src 0 (at 20Hz)
    QVERIFY(p_at_20[1] > p_at_20[2]);
}

QTEST_GUILESS_MAIN(TestDICS)
#include "test_dics.moc"
