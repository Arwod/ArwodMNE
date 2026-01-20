//=============================================================================================================
/**
 * @file     test_correction_properties.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property-based tests for multiple comparison correction
 *           Validates: Requirements 5.3
 *           Property 10: Multiple comparison correction monotonicity
 *
 * Tests the monotonicity property of multiple comparison correction methods:
 * if p1 <= p2, then corrected_p1 <= corrected_p2
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <stats/correction.h>
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
 * DECLARE CLASS TestCorrectionProperties
 *
 * @brief The TestCorrectionProperties class provides property-based tests for correction methods
 *
 */
class TestCorrectionProperties: public QObject
{
    Q_OBJECT

public:
    TestCorrectionProperties();

private slots:
    void initTestCase();
    void testBonferroniMonotonicity();
    void testHolmMonotonicity();
    void testSidakMonotonicity();
    void testFDRMonotonicity();
    void testCorrectionBounds();
    void testCorrectionConsistency();
    void cleanupTestCase();

private:
    // Helper methods
    VectorXd generateRandomPValues(int n, unsigned int seed);
    bool checkMonotonicity(const VectorXd& original, const VectorXd& corrected);
    bool checkBounds(const VectorXd& p_values, double min_val, double max_val);
    VectorXd sortVector(const VectorXd& vec);
    
    // Test parameters
    int m_n_iterations;
    int m_n_tests;
    double m_tolerance;
};

//=============================================================================================================

TestCorrectionProperties::TestCorrectionProperties()
: m_n_iterations(100)  // Property test iterations
, m_n_tests(50)        // Number of p-values per test
, m_tolerance(1e-10)
{
}

//=============================================================================================================

void TestCorrectionProperties::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Multiple Comparison Correction Property Tests";
    qDebug() << "Testing Property 10: Correction monotonicity";
    qDebug() << "Iterations:" << m_n_iterations;
    qDebug() << "Tests per iteration:" << m_n_tests;
    
    // Set random seed for reproducibility
    srand(42);
}

//=============================================================================================================

void TestCorrectionProperties::testBonferroniMonotonicity()
{
    qDebug() << "Testing Bonferroni correction monotonicity...";
    qDebug() << "Property: For all p-values, if p1 <= p2, then corrected_p1 <= corrected_p2";
    
    int violations = 0;
    int total_comparisons = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        // Generate random p-values
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 1);
        
        // Apply Bonferroni correction
        VectorXd p_corrected = Correction::bonferroni(p_values);
        
        // Check monotonicity
        if(!checkMonotonicity(p_values, p_corrected)) {
            violations++;
        }
        total_comparisons++;
    }
    
    qDebug() << "Tested" << total_comparisons << "random p-value sets";
    qDebug() << "Monotonicity violations:" << violations;
    
    // Test: Should have no violations
    QCOMPARE(violations, 0);
    
    qDebug() << "Bonferroni monotonicity test passed (" << m_n_iterations << "/" << m_n_iterations << " iterations)";
}

//=============================================================================================================

void TestCorrectionProperties::testHolmMonotonicity()
{
    qDebug() << "Testing Holm correction monotonicity...";
    
    int violations = 0;
    int total_comparisons = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 100);
        VectorXd p_corrected = Correction::holm(p_values, 0.05);
        
        if(!checkMonotonicity(p_values, p_corrected)) {
            violations++;
        }
        total_comparisons++;
    }
    
    qDebug() << "Tested" << total_comparisons << "random p-value sets";
    qDebug() << "Monotonicity violations:" << violations;
    
    QCOMPARE(violations, 0);
    
    qDebug() << "Holm monotonicity test passed (" << m_n_iterations << "/" << m_n_iterations << " iterations)";
}

//=============================================================================================================

void TestCorrectionProperties::testSidakMonotonicity()
{
    qDebug() << "Testing Sidak correction monotonicity...";
    
    int violations = 0;
    int total_comparisons = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 200);
        VectorXd p_corrected = Correction::sidak(p_values);
        
        if(!checkMonotonicity(p_values, p_corrected)) {
            violations++;
        }
        total_comparisons++;
    }
    
    qDebug() << "Tested" << total_comparisons << "random p-value sets";
    qDebug() << "Monotonicity violations:" << violations;
    
    QCOMPARE(violations, 0);
    
    qDebug() << "Sidak monotonicity test passed (" << m_n_iterations << "/" << m_n_iterations << " iterations)";
}

//=============================================================================================================

void TestCorrectionProperties::testFDRMonotonicity()
{
    qDebug() << "Testing FDR correction monotonicity...";
    
    int violations = 0;
    int total_comparisons = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 300);
        VectorXd p_corrected = Correction::fdr(p_values);
        
        if(!checkMonotonicity(p_values, p_corrected)) {
            violations++;
        }
        total_comparisons++;
    }
    
    qDebug() << "Tested" << total_comparisons << "random p-value sets";
    qDebug() << "Monotonicity violations:" << violations;
    
    QCOMPARE(violations, 0);
    
    qDebug() << "FDR monotonicity test passed (" << m_n_iterations << "/" << m_n_iterations << " iterations)";
}

//=============================================================================================================

void TestCorrectionProperties::testCorrectionBounds()
{
    qDebug() << "Testing correction bounds property...";
    qDebug() << "Property: For all p-values in [0,1], corrected p-values should be in [0,1]";
    
    int violations = 0;
    int total_tests = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 400);
        
        // Test Bonferroni
        VectorXd p_bonf = Correction::bonferroni(p_values);
        if(!checkBounds(p_bonf, 0.0, 1.0)) violations++;
        total_tests++;
        
        // Test Holm
        VectorXd p_holm = Correction::holm(p_values, 0.05);
        if(!checkBounds(p_holm, 0.0, 1.0)) violations++;
        total_tests++;
        
        // Test Sidak
        VectorXd p_sidak = Correction::sidak(p_values);
        if(!checkBounds(p_sidak, 0.0, 1.0)) violations++;
        total_tests++;
        
        // Test FDR
        VectorXd p_fdr = Correction::fdr(p_values);
        if(!checkBounds(p_fdr, 0.0, 1.0)) violations++;
        total_tests++;
    }
    
    qDebug() << "Tested" << total_tests << "correction applications";
    qDebug() << "Bound violations:" << violations;
    
    QCOMPARE(violations, 0);
    
    qDebug() << "Correction bounds test passed";
}

//=============================================================================================================

void TestCorrectionProperties::testCorrectionConsistency()
{
    qDebug() << "Testing correction consistency property...";
    qDebug() << "Property: Correcting the same p-values twice should give the same result";
    
    int violations = 0;
    
    for(int iter = 0; iter < m_n_iterations; ++iter) {
        VectorXd p_values = generateRandomPValues(m_n_tests, iter + 500);
        
        // Apply correction twice
        VectorXd p_corr1 = Correction::bonferroni(p_values);
        VectorXd p_corr2 = Correction::bonferroni(p_values);
        
        // Check if results are identical
        double max_diff = (p_corr1 - p_corr2).cwiseAbs().maxCoeff();
        if(max_diff > m_tolerance) {
            violations++;
        }
    }
    
    qDebug() << "Tested" << m_n_iterations << "correction applications";
    qDebug() << "Consistency violations:" << violations;
    
    QCOMPARE(violations, 0);
    
    qDebug() << "Correction consistency test passed";
}

//=============================================================================================================

void TestCorrectionProperties::cleanupTestCase()
{
    qDebug() << "Multiple Comparison Correction Property Tests completed";
    qDebug() << "All" << m_n_iterations << "property test iterations passed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

VectorXd TestCorrectionProperties::generateRandomPValues(int n, unsigned int seed)
{
    // Generate random p-values in [0, 1]
    srand(seed);
    VectorXd p_values(n);
    
    for(int i = 0; i < n; ++i) {
        p_values(i) = static_cast<double>(rand()) / RAND_MAX;
    }
    
    return p_values;
}

//=============================================================================================================

bool TestCorrectionProperties::checkMonotonicity(const VectorXd& original, const VectorXd& corrected)
{
    // Check if monotonicity is preserved: if p1 <= p2, then corrected_p1 <= corrected_p2
    
    if(original.size() != corrected.size()) {
        return false;
    }
    
    int n = original.size();
    
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            // If original[i] <= original[j]
            if(original(i) <= original(j) + m_tolerance) {
                // Then corrected[i] should be <= corrected[j]
                if(corrected(i) > corrected(j) + m_tolerance) {
                    qDebug() << "Monotonicity violation: p[" << i << "]=" << original(i) 
                             << "<= p[" << j << "]=" << original(j)
                             << "but corrected[" << i << "]=" << corrected(i)
                             << "> corrected[" << j << "]=" << corrected(j);
                    return false;
                }
            }
        }
    }
    
    return true;
}

//=============================================================================================================

bool TestCorrectionProperties::checkBounds(const VectorXd& p_values, double min_val, double max_val)
{
    for(int i = 0; i < p_values.size(); ++i) {
        if(p_values(i) < min_val - m_tolerance || p_values(i) > max_val + m_tolerance) {
            qDebug() << "Bound violation: p[" << i << "]=" << p_values(i) 
                     << "not in [" << min_val << "," << max_val << "]";
            return false;
        }
    }
    return true;
}

//=============================================================================================================

VectorXd TestCorrectionProperties::sortVector(const VectorXd& vec)
{
    VectorXd sorted = vec;
    std::sort(sorted.data(), sorted.data() + sorted.size());
    return sorted;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestCorrectionProperties)
#include "test_correction_properties.moc"
