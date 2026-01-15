//=============================================================================================================
/**
 * @file     test_parametric_tests.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Unit tests for parametric statistical tests
 *           Validates: Requirements 5.1
 *
 * Tests the accuracy and consistency of parametric statistical tests
 * including t-tests, ANOVA, effect sizes, and bootstrap confidence intervals.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <stats/statistics.h>
#include <utils/generics/applicationlogger.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QTest>
#include <QCoreApplication>
#include <QDebug>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Dense>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace Eigen;
using namespace STATSLIB;

//=============================================================================================================
/**
 * DECLARE CLASS TestParametricTests
 *
 * @brief The TestParametricTests class provides unit tests for parametric statistical tests
 *
 */
class TestParametricTests: public QObject
{
    Q_OBJECT

public:
    TestParametricTests();

private slots:
    void initTestCase();
    void testOneSampleTTest();
    void testIndependentTTest();
    void testWelchTTest();
    void testPairedTTest();
    void testOneWayANOVA();
    void testCohensD();
    void testEtaSquared();
    void testBootstrapCI();
    void cleanupTestCase();

private:
    // Helper methods
    MatrixXd createNormalData(int n_samples, int n_features, double mean, double std);
    MatrixXd createGroupData(int n_samples, int n_features, double mean_diff);
    bool checkTValueSign(double t_value, double expected_sign_indicator);
    bool checkEffectSizeRange(double effect_size, double min_val, double max_val);
    
    // Test parameters
    double m_tolerance;
    int m_n_samples;
    int m_n_features;
};

//=============================================================================================================

TestParametricTests::TestParametricTests()
: m_tolerance(0.1)
, m_n_samples(100)
, m_n_features(10)
{
}

//=============================================================================================================

void TestParametricTests::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Parametric Tests Unit Tests";
    qDebug() << "Testing statistical test accuracy and consistency";
    qDebug() << "Tolerance:" << m_tolerance;
    qDebug() << "Sample size:" << m_n_samples;
    qDebug() << "Number of features:" << m_n_features;
    
    // Set random seed for reproducibility
    srand(42);
}

//=============================================================================================================

void TestParametricTests::testOneSampleTTest()
{
    qDebug() << "Testing one-sample t-test...";
    
    // Create data with known mean
    double true_mean = 5.0;
    MatrixXd data = createNormalData(m_n_samples, m_n_features, true_mean, 1.0);
    
    // Test against population mean of 0
    RowVectorXd t_values = Statistics::ttest_1samp_no_p(data, 0.0);
    
    qDebug() << "Computed t-values for" << m_n_features << "features";
    qDebug() << "Mean t-value:" << t_values.mean();
    qDebug() << "Min t-value:" << t_values.minCoeff();
    qDebug() << "Max t-value:" << t_values.maxCoeff();
    
    // Test: Should have correct dimensions
    QCOMPARE(t_values.cols(), m_n_features);
    
    // Test: T-values should be positive (data mean > 0)
    QVERIFY(t_values.mean() > 0.0);
    
    // Test: T-values should be large (significant difference from 0)
    QVERIFY(t_values.minCoeff() > 10.0);  // With n=100, mean=5, std=1, t should be ~50
    
    // Test against the true mean (should give t-values near 0)
    RowVectorXd t_values_true = Statistics::ttest_1samp_no_p(data, true_mean);
    qDebug() << "T-values against true mean - mean:" << t_values_true.mean() 
             << "max abs:" << t_values_true.cwiseAbs().maxCoeff();
    
    // T-values should be small when testing against true mean
    QVERIFY(t_values_true.cwiseAbs().maxCoeff() < 3.0);
    
    qDebug() << "One-sample t-test passed";
}

//=============================================================================================================

void TestParametricTests::testIndependentTTest()
{
    qDebug() << "Testing independent samples t-test...";
    
    // Create two groups with different means
    double mean1 = 0.0;
    double mean2 = 1.0;
    MatrixXd group1 = createNormalData(m_n_samples, m_n_features, mean1, 1.0);
    MatrixXd group2 = createNormalData(m_n_samples, m_n_features, mean2, 1.0);
    
    // Compute t-test
    RowVectorXd t_values = Statistics::ttest_ind_no_p(group1, group2, true);
    
    qDebug() << "Computed t-values for" << m_n_features << "features";
    qDebug() << "Mean t-value:" << t_values.mean();
    
    // Test: Should have correct dimensions
    QCOMPARE(t_values.cols(), m_n_features);
    
    // Test: T-values should be negative (group1 mean < group2 mean)
    QVERIFY(t_values.mean() < 0.0);
    
    // Test: T-values should be significant
    QVERIFY(t_values.cwiseAbs().minCoeff() > 3.0);
    
    // Test with equal groups (should give t-values near 0)
    MatrixXd group3 = createNormalData(m_n_samples, m_n_features, mean1, 1.0);
    RowVectorXd t_values_equal = Statistics::ttest_ind_no_p(group1, group3, true);
    
    qDebug() << "T-values for equal groups - max abs:" << t_values_equal.cwiseAbs().maxCoeff();
    QVERIFY(t_values_equal.cwiseAbs().maxCoeff() < 3.0);
    
    qDebug() << "Independent samples t-test passed";
}

//=============================================================================================================

void TestParametricTests::testWelchTTest()
{
    qDebug() << "Testing Welch's t-test...";
    
    // Create two groups with different means and variances
    MatrixXd group1 = createNormalData(m_n_samples, m_n_features, 0.0, 1.0);
    MatrixXd group2 = createNormalData(m_n_samples, m_n_features, 1.0, 2.0);
    
    // Compute Welch's t-test
    RowVectorXd t_values = Statistics::welch_ttest(group1, group2);
    
    qDebug() << "Computed Welch's t-values for" << m_n_features << "features";
    qDebug() << "Mean t-value:" << t_values.mean();
    
    // Test: Should have correct dimensions
    QCOMPARE(t_values.cols(), m_n_features);
    
    // Test: T-values should be negative (group1 mean < group2 mean)
    QVERIFY(t_values.mean() < 0.0);
    
    // Test: Should produce finite values
    bool all_finite = true;
    for(int i = 0; i < t_values.cols(); ++i) {
        if(!std::isfinite(t_values(i))) {
            all_finite = false;
            break;
        }
    }
    QVERIFY(all_finite);
    
    qDebug() << "Welch's t-test passed";
}

//=============================================================================================================

void TestParametricTests::testPairedTTest()
{
    qDebug() << "Testing paired samples t-test...";
    
    // Create paired data (same subjects, two conditions)
    MatrixXd data1 = createNormalData(m_n_samples, m_n_features, 0.0, 1.0);
    MatrixXd data2 = data1.array() + 0.5;  // Add constant difference
    
    // Compute paired t-test
    RowVectorXd t_values = Statistics::paired_ttest(data1, data2);
    
    qDebug() << "Computed paired t-values for" << m_n_features << "features";
    qDebug() << "Mean t-value:" << t_values.mean();
    
    // Test: Should have correct dimensions
    QCOMPARE(t_values.cols(), m_n_features);
    
    // Test: T-values should be negative (data1 < data2)
    QVERIFY(t_values.mean() < 0.0);
    
    // Test: T-values should be significant
    QVERIFY(t_values.cwiseAbs().minCoeff() > 3.0);
    
    // Test with identical data (should give t-values of 0)
    RowVectorXd t_values_same = Statistics::paired_ttest(data1, data1);
    qDebug() << "T-values for identical data - max abs:" << t_values_same.cwiseAbs().maxCoeff();
    QVERIFY(t_values_same.cwiseAbs().maxCoeff() < 1e-10);
    
    qDebug() << "Paired samples t-test passed";
}

//=============================================================================================================

void TestParametricTests::testOneWayANOVA()
{
    qDebug() << "Testing one-way ANOVA...";
    
    // Create three groups with different means
    MatrixXd group1 = createNormalData(m_n_samples, m_n_features, 0.0, 1.0);
    MatrixXd group2 = createNormalData(m_n_samples, m_n_features, 1.0, 1.0);
    MatrixXd group3 = createNormalData(m_n_samples, m_n_features, 2.0, 1.0);
    
    QList<MatrixXd> groups;
    groups.append(group1);
    groups.append(group2);
    groups.append(group3);
    
    // Compute F-statistic
    RowVectorXd f_values = Statistics::f_oneway(groups);
    
    qDebug() << "Computed F-values for" << m_n_features << "features";
    qDebug() << "Mean F-value:" << f_values.mean();
    qDebug() << "Min F-value:" << f_values.minCoeff();
    
    // Test: Should have correct dimensions
    QCOMPARE(f_values.cols(), m_n_features);
    
    // Test: F-values should be positive
    QVERIFY(f_values.minCoeff() > 0.0);
    
    // Test: F-values should be large (significant group differences)
    QVERIFY(f_values.mean() > 10.0);
    
    // Test with equal groups (should give small F-values)
    QList<MatrixXd> equal_groups;
    equal_groups.append(group1);
    equal_groups.append(createNormalData(m_n_samples, m_n_features, 0.0, 1.0));
    equal_groups.append(createNormalData(m_n_samples, m_n_features, 0.0, 1.0));
    
    RowVectorXd f_values_equal = Statistics::f_oneway(equal_groups);
    qDebug() << "F-values for equal groups - mean:" << f_values_equal.mean();
    QVERIFY(f_values_equal.mean() < 5.0);
    
    qDebug() << "One-way ANOVA passed";
}

//=============================================================================================================

void TestParametricTests::testCohensD()
{
    qDebug() << "Testing Cohen's d effect size...";
    
    // Create two groups with known difference
    MatrixXd group1 = createNormalData(m_n_samples, m_n_features, 0.0, 1.0);
    MatrixXd group2 = createNormalData(m_n_samples, m_n_features, 1.0, 1.0);
    
    // Compute Cohen's d
    RowVectorXd cohens_d = Statistics::cohens_d(group1, group2, true);
    
    qDebug() << "Computed Cohen's d for" << m_n_features << "features";
    qDebug() << "Mean Cohen's d:" << cohens_d.mean();
    qDebug() << "Expected Cohen's d: ~1.0 (1 SD difference)";
    
    // Test: Should have correct dimensions
    QCOMPARE(cohens_d.cols(), m_n_features);
    
    // Test: Cohen's d should be negative (group1 < group2)
    QVERIFY(cohens_d.mean() < 0.0);
    
    // Test: Absolute value should be close to 1.0 (1 SD difference)
    double mean_abs_d = cohens_d.cwiseAbs().mean();
    qDebug() << "Mean absolute Cohen's d:" << mean_abs_d;
    QVERIFY(mean_abs_d > 0.7 && mean_abs_d < 1.3);
    
    // Test with identical groups (should give d near 0)
    RowVectorXd cohens_d_zero = Statistics::cohens_d(group1, group1, true);
    qDebug() << "Cohen's d for identical groups - max abs:" << cohens_d_zero.cwiseAbs().maxCoeff();
    QVERIFY(cohens_d_zero.cwiseAbs().maxCoeff() < 0.1);
    
    qDebug() << "Cohen's d effect size passed";
}

//=============================================================================================================

void TestParametricTests::testEtaSquared()
{
    qDebug() << "Testing eta-squared effect size...";
    
    // Create three groups with different means
    MatrixXd group1 = createNormalData(m_n_samples, m_n_features, 0.0, 1.0);
    MatrixXd group2 = createNormalData(m_n_samples, m_n_features, 1.0, 1.0);
    MatrixXd group3 = createNormalData(m_n_samples, m_n_features, 2.0, 1.0);
    
    QList<MatrixXd> groups;
    groups.append(group1);
    groups.append(group2);
    groups.append(group3);
    
    // Compute eta-squared
    RowVectorXd eta_sq = Statistics::eta_squared(groups);
    
    qDebug() << "Computed eta-squared for" << m_n_features << "features";
    qDebug() << "Mean eta-squared:" << eta_sq.mean();
    
    // Test: Should have correct dimensions
    QCOMPARE(eta_sq.cols(), m_n_features);
    
    // Test: Eta-squared should be between 0 and 1
    QVERIFY(eta_sq.minCoeff() >= 0.0);
    QVERIFY(eta_sq.maxCoeff() <= 1.0);
    
    // Test: Should be large for groups with large differences
    QVERIFY(eta_sq.mean() > 0.3);  // Large effect
    
    qDebug() << "Eta-squared effect size passed";
}

//=============================================================================================================

void TestParametricTests::testBootstrapCI()
{
    qDebug() << "Testing bootstrap confidence intervals...";
    
    // Create data with known mean
    double true_mean = 5.0;
    MatrixXd data = createNormalData(m_n_samples, m_n_features, true_mean, 1.0);
    
    // Compute bootstrap CI for mean
    auto ci = Statistics::bootstrap_ci_mean(data, 100, 0.95, 
                                           Statistics::BootstrapMethod::Percentile, 42);
    
    RowVectorXd lower = ci.first;
    RowVectorXd upper = ci.second;
    
    qDebug() << "Computed bootstrap CI for" << m_n_features << "features";
    qDebug() << "Mean lower bound:" << lower.mean();
    qDebug() << "Mean upper bound:" << upper.mean();
    qDebug() << "True mean:" << true_mean;
    
    // Test: Should have correct dimensions
    QCOMPARE(lower.cols(), m_n_features);
    QCOMPARE(upper.cols(), m_n_features);
    
    // Test: Lower bound should be less than upper bound
    for(int i = 0; i < m_n_features; ++i) {
        QVERIFY(lower(i) < upper(i));
    }
    
    // Test: True mean should be within CI for most features
    int within_ci = 0;
    for(int i = 0; i < m_n_features; ++i) {
        if(lower(i) <= true_mean && true_mean <= upper(i)) {
            within_ci++;
        }
    }
    
    double coverage = static_cast<double>(within_ci) / m_n_features;
    qDebug() << "Coverage:" << (coverage * 100) << "% (expected ~95%)";
    QVERIFY(coverage > 0.80);  // Allow some variation
    
    qDebug() << "Bootstrap confidence intervals passed";
}

//=============================================================================================================

void TestParametricTests::cleanupTestCase()
{
    qDebug() << "Parametric Tests Unit Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestParametricTests::createNormalData(int n_samples, int n_features, double mean, double std)
{
    MatrixXd data(n_samples, n_features);
    
    for(int i = 0; i < n_samples; ++i) {
        for(int j = 0; j < n_features; ++j) {
            // Box-Muller transform for normal distribution
            double u1 = static_cast<double>(rand()) / RAND_MAX;
            double u2 = static_cast<double>(rand()) / RAND_MAX;
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            data(i, j) = mean + std * z;
        }
    }
    
    return data;
}

//=============================================================================================================

MatrixXd TestParametricTests::createGroupData(int n_samples, int n_features, double mean_diff)
{
    return createNormalData(n_samples, n_features, mean_diff, 1.0);
}

//=============================================================================================================

bool TestParametricTests::checkTValueSign(double t_value, double expected_sign_indicator)
{
    if(expected_sign_indicator > 0) {
        return t_value > 0;
    } else if(expected_sign_indicator < 0) {
        return t_value < 0;
    } else {
        return std::abs(t_value) < 1.0;  // Near zero
    }
}

//=============================================================================================================

bool TestParametricTests::checkEffectSizeRange(double effect_size, double min_val, double max_val)
{
    return effect_size >= min_val && effect_size <= max_val;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestParametricTests)
#include "test_parametric_tests.moc"
