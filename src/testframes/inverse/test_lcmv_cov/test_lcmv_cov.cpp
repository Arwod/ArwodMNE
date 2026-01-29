#include <QtTest/QtTest>
#include <inverse/beamformer/lcmv.h>
#include <inverse/beamformer/covariance.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace INVERSELIB;
using namespace Eigen;

class TestLCMVCov : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCovarianceCompute();
    void testLCMVMaxPower();
    void testLCMVUNG();
};

void TestLCMVCov::initTestCase()
{
}

void TestLCMVCov::cleanupTestCase()
{
}

void TestLCMVCov::testCovarianceCompute()
{
    // Generate data with known covariance
    // Ch 0 and Ch 1 correlated
    int n_channels = 3;
    int n_samples = 1000;
    MatrixXd data(n_channels, n_samples);
    
    // Seed random
    srand(42);
    
    for(int i=0; i<n_samples; ++i) {
        double common = (double)rand() / RAND_MAX;
        data(0, i) = common + 0.1 * ((double)rand() / RAND_MAX);
        data(1, i) = common + 0.1 * ((double)rand() / RAND_MAX);
        data(2, i) = (double)rand() / RAND_MAX; // Uncorrelated
    }
    
    Covariance cov = Covariance::compute_empirical(data);
    
    QCOMPARE(cov.data.rows(), n_channels);
    QCOMPARE(cov.data.cols(), n_channels);
    QCOMPARE(cov.nfree, n_samples - 1);
    
    // Check correlation
    double var0 = cov.data(0, 0);
    double var1 = cov.data(1, 1);
    double cov01 = cov.data(0, 1);
    double corr = cov01 / std::sqrt(var0 * var1);
    
    qDebug() << "Correlation 0-1:" << corr;
    QVERIFY(corr > 0.9); // Should be high
    
    double cov02 = cov.data(0, 2);
    double corr02 = cov02 / std::sqrt(var0 * cov.data(2, 2));
    qDebug() << "Correlation 0-2:" << corr02;
    QVERIFY(std::abs(corr02) < 0.2); // Should be low
}

void TestLCMVCov::testLCMVMaxPower()
{
    int n_channels = 3;
    int n_sources = 1;
    int n_ori = 3;
    
    // Leadfield: Source 0 (X, Y, Z)
    // Assume Source 0 is oriented along X axis (1, 0, 0)
    // So L_x is strong, L_y, L_z are weak or zero.
    // L matrix (Ch x 3)
    MatrixXd L(n_channels, n_ori);
    L << 1, 0, 0,
         0.5, 0, 0,
         0.2, 0, 0; 
         
    // Data Covariance: Identity (white noise) + Source Signal
    // Source signal along X.
    MatrixXd signal_cov(n_channels, n_channels);
    VectorXd source_proj = L.col(0);
    signal_cov = source_proj * source_proj.transpose() * 10.0; // High power
    
    MatrixXd noise_cov = MatrixXd::Identity(n_channels, n_channels);
    MatrixXd data_cov_mat = signal_cov + noise_cov;
    
    Covariance data_cov(data_cov_mat);
    Covariance noise_c(noise_cov);
    
    // Compute LCMV with max-power
    BeamformerWeights res = LCMV::make_lcmv(L, data_cov, noise_c, 0.0, "max-power", "none", 3);
    
    QCOMPARE(res.weights.rows(), 1); // 1 source scalar
    
    // Apply to data that is pure signal
    VectorXd data_vec = source_proj;
    MatrixXd data_mat(n_channels, 1);
    data_mat.col(0) = data_vec;
    
    MatrixXd est = LCMV::apply(res, data_mat);
    
    qDebug() << "Source Estimate:" << est(0,0);
    QVERIFY(std::abs(est(0,0)) > 0.1); // Should detect signal
}

void TestLCMVCov::testLCMVUNG()
{
    int n_channels = 3;
    int n_sources = 1;
    int n_ori = 3;
    
    MatrixXd L = MatrixXd::Identity(n_channels, n_ori); // Identity leadfield
    Covariance data_cov(MatrixXd::Identity(n_channels, n_channels));
    Covariance noise_cov(MatrixXd::Identity(n_channels, n_channels)); // Identity noise
    
    // With UNG, output noise power should be 1.0 (unit noise gain)
    // P_noise = W * N * W^T
    // If W is normalized, diag(P_noise) should be 1.
    
    BeamformerWeights res = LCMV::make_lcmv(L, data_cov, noise_cov, 0.0, "vector", "unit-noise-gain", 3);
    
    MatrixXd W = res.weights;
    MatrixXd P_noise = W * noise_cov.data * W.transpose();
    
    qDebug() << "Noise Power Diag:" << P_noise(0,0) << P_noise(1,1) << P_noise(2,2);
    
    QVERIFY(std::abs(P_noise(0,0) - 1.0) < 1e-5);
    QVERIFY(std::abs(P_noise(1,1) - 1.0) < 1e-5);
    QVERIFY(std::abs(P_noise(2,2) - 1.0) < 1e-5);
}

QTEST_GUILESS_MAIN(TestLCMVCov)
#include "test_lcmv_cov.moc"
