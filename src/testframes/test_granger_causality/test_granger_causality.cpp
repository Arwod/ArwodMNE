#include <QtTest/QtTest>
#include <connectivity/metrics/granger_causality.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace CONNECTIVITYLIB;
using namespace Eigen;

class TestGrangerCausality : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Core functionality tests
    void testUnidirectionalCausality();
    void testBidirectionalCausality();
    void testNoCausality();
    void testModelOrderSelection();
    void testARModelFitting();
    
    // Edge cases and robustness
    void testShortTimeSeries();
    void testIdenticalSignals();
    void testAsymmetry();
    
    // Accuracy tests
    void testKnownCausalRelationship();
    void testCausalityMagnitude();

private:
    // Helper functions
    VectorXd generateAR(int n_samples, const VectorXd& coeffs, double noise_std, std::mt19937& rng);
    std::pair<VectorXd, VectorXd> generateCoupledAR(int n_samples, double coupling, double noise_std, std::mt19937& rng);
};

void TestGrangerCausality::initTestCase()
{
    qDebug() << "Starting Granger Causality tests";
}

void TestGrangerCausality::cleanupTestCase()
{
    qDebug() << "Granger Causality tests completed";
}

VectorXd TestGrangerCausality::generateAR(int n_samples, const VectorXd& coeffs, double noise_std, std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, noise_std);
    int order = coeffs.size();
    VectorXd signal = VectorXd::Zero(n_samples);
    
    // Initialize with noise
    for(int i = 0; i < order; ++i) {
        signal(i) = dist(rng);
    }
    
    // Generate AR process
    for(int t = order; t < n_samples; ++t) {
        double value = 0.0;
        for(int lag = 0; lag < order; ++lag) {
            value += coeffs(lag) * signal(t - lag - 1);
        }
        signal(t) = value + dist(rng);
    }
    
    return signal;
}

std::pair<VectorXd, VectorXd> TestGrangerCausality::generateCoupledAR(int n_samples, double coupling, double noise_std, std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, noise_std);
    
    VectorXd x = VectorXd::Zero(n_samples);
    VectorXd y = VectorXd::Zero(n_samples);
    
    // Initialize with noise
    x(0) = dist(rng);
    y(0) = dist(rng);
    
    // Generate coupled AR processes
    // x(t) = 0.5*x(t-1) + noise
    // y(t) = coupling*x(t-1) + 0.3*y(t-1) + noise
    for(int t = 1; t < n_samples; ++t) {
        x(t) = 0.5 * x(t-1) + dist(rng);
        y(t) = coupling * x(t-1) + 0.3 * y(t-1) + dist(rng);
    }
    
    return {x, y};
}

void TestGrangerCausality::testUnidirectionalCausality()
{
    // Test that X -> Y causality is detected when X causes Y but not vice versa
    std::mt19937 rng(42);
    int n_samples = 500;
    double coupling = 0.6;
    double noise_std = 0.1;
    
    auto [x, y] = generateCoupledAR(n_samples, coupling, noise_std, rng);
    
    int modelOrder = 5;
    
    // Compute GC from X to Y (should be significant)
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    
    // Compute GC from Y to X (should be near zero)
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "Unidirectional test: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // X should cause Y
    QVERIFY2(gc_x_to_y > 0.01, "X->Y causality should be detected");
    
    // Y should not cause X (or much weaker)
    QVERIFY2(gc_y_to_x < gc_x_to_y * 0.5, "Y->X causality should be weaker than X->Y");
}

void TestGrangerCausality::testBidirectionalCausality()
{
    // Test bidirectional causality where both X -> Y and Y -> X
    std::mt19937 rng(123);
    int n_samples = 500;
    double noise_std = 0.1;
    std::normal_distribution<double> dist(0.0, noise_std);
    
    VectorXd x = VectorXd::Zero(n_samples);
    VectorXd y = VectorXd::Zero(n_samples);
    
    x(0) = dist(rng);
    y(0) = dist(rng);
    
    // Bidirectional coupling
    // x(t) = 0.4*x(t-1) + 0.3*y(t-1) + noise
    // y(t) = 0.3*x(t-1) + 0.4*y(t-1) + noise
    for(int t = 1; t < n_samples; ++t) {
        x(t) = 0.4 * x(t-1) + 0.3 * y(t-1) + dist(rng);
        y(t) = 0.3 * x(t-1) + 0.4 * y(t-1) + dist(rng);
    }
    
    int modelOrder = 5;
    
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "Bidirectional test: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // Both directions should show causality
    QVERIFY2(gc_x_to_y > 0.01, "X->Y causality should be detected");
    QVERIFY2(gc_y_to_x > 0.01, "Y->X causality should be detected");
    
    // Should be roughly similar magnitude (within factor of 3)
    double ratio = std::max(gc_x_to_y, gc_y_to_x) / std::min(gc_x_to_y, gc_y_to_x);
    QVERIFY2(ratio < 3.0, "Bidirectional causality should have similar magnitudes");
}

void TestGrangerCausality::testNoCausality()
{
    // Test that independent signals show no causality
    std::mt19937 rng(456);
    int n_samples = 500;
    
    // Generate two independent AR processes
    VectorXd coeffs1(2);
    coeffs1 << 0.5, -0.2;
    VectorXd x = generateAR(n_samples, coeffs1, 0.1, rng);
    
    VectorXd coeffs2(2);
    coeffs2 << 0.4, -0.3;
    VectorXd y = generateAR(n_samples, coeffs2, 0.1, rng);
    
    int modelOrder = 5;
    
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "No causality test: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // Both should be near zero (less than 0.05)
    QVERIFY2(gc_x_to_y < 0.05, "Independent signals should show minimal X->Y causality");
    QVERIFY2(gc_y_to_x < 0.05, "Independent signals should show minimal Y->X causality");
}

void TestGrangerCausality::testModelOrderSelection()
{
    // Test automatic model order selection
    std::mt19937 rng(789);
    int n_samples = 500;
    
    // Generate AR(3) process
    VectorXd coeffs(3);
    coeffs << 0.5, -0.3, 0.2;
    VectorXd signal = generateAR(n_samples, coeffs, 0.1, rng);
    
    // Test AIC criterion
    int order_aic = GrangerCausality::selectModelOrder(signal, 10, "AIC");
    qDebug() << "Selected order (AIC):" << order_aic;
    QVERIFY2(order_aic >= 2 && order_aic <= 6, "AIC should select reasonable order near true order 3");
    
    // Test BIC criterion
    int order_bic = GrangerCausality::selectModelOrder(signal, 10, "BIC");
    qDebug() << "Selected order (BIC):" << order_bic;
    QVERIFY2(order_bic >= 2 && order_bic <= 6, "BIC should select reasonable order near true order 3");
}

void TestGrangerCausality::testARModelFitting()
{
    // Test AR model fitting accuracy
    std::mt19937 rng(101112);
    int n_samples = 1000;
    
    // Generate AR(2) process with known coefficients
    VectorXd true_coeffs(2);
    true_coeffs << 0.6, -0.3;
    VectorXd signal = generateAR(n_samples, true_coeffs, 0.1, rng);
    
    // Fit AR model
    VectorXd fitted_coeffs;
    double residual_var;
    bool success = GrangerCausality::fitARModel(signal, 2, fitted_coeffs, residual_var);
    
    QVERIFY2(success, "AR model fitting should succeed");
    QVERIFY2(fitted_coeffs.size() == 3, "Should return 3 coefficients (intercept + 2 AR coeffs)");
    
    qDebug() << "True coeffs: [" << true_coeffs(0) << ", " << true_coeffs(1) << "]";
    qDebug() << "Fitted coeffs: [" << fitted_coeffs(0) << ", " << fitted_coeffs(1) << ", " << fitted_coeffs(2) << "]";
    qDebug() << "Residual variance:" << residual_var;
    
    // Check coefficient accuracy (within 20% relative error)
    // Skip intercept (index 0), check AR coefficients (indices 1 and 2)
    for(int i = 0; i < 2; ++i) {
        double rel_error = std::abs(fitted_coeffs(i + 1) - true_coeffs(i)) / std::abs(true_coeffs(i));
        QVERIFY2(rel_error < 0.2, QString("AR coefficient %1 should be accurate").arg(i).toStdString().c_str());
    }
    
    // Residual variance should be reasonable
    QVERIFY2(residual_var > 0.005 && residual_var < 0.05, "Residual variance should be reasonable");
}

void TestGrangerCausality::testShortTimeSeries()
{
    // Test behavior with short time series
    std::mt19937 rng(131415);
    int n_samples = 50;  // Short series
    
    VectorXd coeffs(2);
    coeffs << 0.5, -0.2;
    VectorXd x = generateAR(n_samples, coeffs, 0.1, rng);
    VectorXd y = generateAR(n_samples, coeffs, 0.1, rng);
    
    int modelOrder = 5;
    
    // Should handle short series gracefully
    double gc = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    
    qDebug() << "Short series GC:" << gc;
    
    // Should return a valid value (not NaN or inf)
    QVERIFY2(!std::isnan(gc), "GC should not be NaN for short series");
    QVERIFY2(!std::isinf(gc), "GC should not be infinite for short series");
    QVERIFY2(gc >= 0.0, "GC should be non-negative");
}

void TestGrangerCausality::testIdenticalSignals()
{
    // Test with identical signals
    std::mt19937 rng(161718);
    int n_samples = 500;
    
    VectorXd coeffs(2);
    coeffs << 0.5, -0.2;
    VectorXd x = generateAR(n_samples, coeffs, 0.1, rng);
    VectorXd y = x;  // Identical signal
    
    int modelOrder = 5;
    
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "Identical signals: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // For identical signals, GC should be very small (near machine precision)
    // because the restricted model (y predicting itself) is as good as the unrestricted model
    QVERIFY2(gc_x_to_y < 0.01, "Identical signals should show minimal X->Y causality");
    QVERIFY2(gc_y_to_x < 0.01, "Identical signals should show minimal Y->X causality");
    
    // Should be symmetric
    double diff = std::abs(gc_x_to_y - gc_y_to_x);
    QVERIFY2(diff < 1e-10, "Causality should be symmetric for identical signals");
}

void TestGrangerCausality::testAsymmetry()
{
    // Test that GC is asymmetric (GC(X->Y) != GC(Y->X) in general)
    std::mt19937 rng(192021);
    int n_samples = 500;
    double coupling = 0.7;
    double noise_std = 0.1;
    
    auto [x, y] = generateCoupledAR(n_samples, coupling, noise_std, rng);
    
    int modelOrder = 5;
    
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "Asymmetry test: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // For unidirectional coupling, should be asymmetric
    QVERIFY2(gc_x_to_y != gc_y_to_x, "GC should be asymmetric");
    QVERIFY2(gc_x_to_y > gc_y_to_x, "X->Y should be stronger than Y->X for unidirectional coupling");
}

void TestGrangerCausality::testKnownCausalRelationship()
{
    // Test with a known causal relationship: y(t) = 0.8*x(t-1) + noise
    std::mt19937 rng(222324);
    int n_samples = 500;
    double noise_std = 0.1;
    std::normal_distribution<double> dist(0.0, noise_std);
    
    VectorXd x = VectorXd::Zero(n_samples);
    VectorXd y = VectorXd::Zero(n_samples);
    
    // Generate x as AR(1) process
    x(0) = dist(rng);
    for(int t = 1; t < n_samples; ++t) {
        x(t) = 0.5 * x(t-1) + dist(rng);
    }
    
    // y depends on past x with strong coupling
    y(0) = dist(rng);
    for(int t = 1; t < n_samples; ++t) {
        y(t) = 0.8 * x(t-1) + dist(rng);
    }
    
    int modelOrder = 5;
    
    double gc_x_to_y = GrangerCausality::computePairwiseGC(x, y, modelOrder);
    double gc_y_to_x = GrangerCausality::computePairwiseGC(y, x, modelOrder);
    
    qDebug() << "Known relationship: GC(X->Y)=" << gc_x_to_y << ", GC(Y->X)=" << gc_y_to_x;
    
    // Strong X->Y causality should be detected
    QVERIFY2(gc_x_to_y > 0.5, "Strong X->Y causality should be detected");
    
    // Y->X should be much weaker
    QVERIFY2(gc_y_to_x < 0.1, "Y->X causality should be weak");
    
    // Ratio should be large
    double ratio = gc_x_to_y / (gc_y_to_x + 1e-10);
    QVERIFY2(ratio > 5.0, "X->Y should be much stronger than Y->X");
}

void TestGrangerCausality::testCausalityMagnitude()
{
    // Test that causality magnitude increases with coupling strength
    std::mt19937 rng(252627);
    int n_samples = 500;
    double noise_std = 0.1;
    
    // Test with weak coupling
    auto [x1, y1] = generateCoupledAR(n_samples, 0.2, noise_std, rng);
    double gc_weak = GrangerCausality::computePairwiseGC(x1, y1, 5);
    
    // Test with strong coupling
    rng.seed(252627);  // Reset seed for fair comparison
    auto [x2, y2] = generateCoupledAR(n_samples, 0.8, noise_std, rng);
    double gc_strong = GrangerCausality::computePairwiseGC(x2, y2, 5);
    
    qDebug() << "Weak coupling GC:" << gc_weak;
    qDebug() << "Strong coupling GC:" << gc_strong;
    
    // Stronger coupling should result in higher GC value
    QVERIFY2(gc_strong > gc_weak, "Stronger coupling should result in higher GC");
    QVERIFY2(gc_strong > 1.5 * gc_weak, "GC should increase substantially with coupling strength");
}

QTEST_MAIN(TestGrangerCausality)
#include "test_granger_causality.moc"
