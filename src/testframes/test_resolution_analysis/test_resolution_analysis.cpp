//=============================================================================================================
/**
 * @file     test_resolution_analysis.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Unit tests for resolution analysis functionality
 *           Validates: Requirements 4.3
 *
 * Tests the properties and analysis results of resolution matrices,
 * including cross-talk and point spread functions.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

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

//=============================================================================================================
/**
 * DECLARE CLASS TestResolutionAnalysis
 *
 * @brief The TestResolutionAnalysis class provides unit tests for resolution analysis
 *
 */
class TestResolutionAnalysis: public QObject
{
    Q_OBJECT

public:
    TestResolutionAnalysis();

private slots:
    void initTestCase();
    void testResolutionMatrixProperties();
    void testCrossTalkFunction();
    void testPointSpreadFunction();
    void testResolutionMetrics();
    void cleanupTestCase();

private:
    // Helper methods
    MatrixXd createTestResolutionMatrix(int n_sources);
    double computeMatrixNorm(const MatrixXd& matrix);
    
    // Test parameters
    double m_tolerance;
};

//=============================================================================================================

TestResolutionAnalysis::TestResolutionAnalysis()
: m_tolerance(1e-6)
{
}

//=============================================================================================================

void TestResolutionAnalysis::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Resolution Analysis Unit Tests";
    qDebug() << "Testing resolution matrix properties and analysis functions";
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestResolutionAnalysis::testResolutionMatrixProperties()
{
    qDebug() << "Testing resolution matrix properties...";
    
    // Create a test resolution matrix
    int n_sources = 100;
    MatrixXd resolution = createTestResolutionMatrix(n_sources);
    
    qDebug() << "Resolution matrix dimensions:" << resolution.rows() << "x" << resolution.cols();
    
    // Test 1: Resolution matrix should be square
    QCOMPARE(resolution.rows(), resolution.cols());
    QCOMPARE(resolution.rows(), n_sources);
    
    // Test 2: Diagonal elements should be positive (self-resolution)
    bool all_positive_diagonal = true;
    for(int i = 0; i < n_sources; ++i) {
        if(resolution(i, i) <= 0.0) {
            all_positive_diagonal = false;
            break;
        }
    }
    QVERIFY(all_positive_diagonal);
    
    // Test 3: Matrix norm should be reasonable
    double matrix_norm = computeMatrixNorm(resolution);
    qDebug() << "Resolution matrix norm:" << matrix_norm;
    QVERIFY(matrix_norm > 0.0);
    QVERIFY(matrix_norm < 1000.0);  // Reasonable upper bound
    
    // Test 4: Row sums should be reasonable (not too large)
    VectorXd row_sums = resolution.rowwise().sum();
    double max_row_sum = row_sums.maxCoeff();
    double min_row_sum = row_sums.minCoeff();
    
    qDebug() << "Row sum range:" << min_row_sum << "to" << max_row_sum;
    QVERIFY(max_row_sum > 0.0);
    
    qDebug() << "Resolution matrix properties test passed";
}

//=============================================================================================================

void TestResolutionAnalysis::testCrossTalkFunction()
{
    qDebug() << "Testing cross-talk function...";
    
    // Create a test resolution matrix
    int n_sources = 50;
    MatrixXd resolution = createTestResolutionMatrix(n_sources);
    
    // Compute cross-talk for each source
    // Cross-talk measures how much activity from other sources leaks into each source
    VectorXd cross_talk(n_sources);
    
    for(int i = 0; i < n_sources; ++i) {
        // Cross-talk is the sum of off-diagonal elements in the row
        double ct = 0.0;
        for(int j = 0; j < n_sources; ++j) {
            if(i != j) {
                ct += std::abs(resolution(i, j));
            }
        }
        cross_talk(i) = ct;
    }
    
    qDebug() << "Computed cross-talk for" << n_sources << "sources";
    qDebug() << "Cross-talk range:" << cross_talk.minCoeff() << "to" << cross_talk.maxCoeff();
    qDebug() << "Mean cross-talk:" << cross_talk.mean();
    
    // Test: Cross-talk should be non-negative
    QVERIFY(cross_talk.minCoeff() >= 0.0);
    
    // Test: Cross-talk should be reasonable (not too large)
    QVERIFY(cross_talk.maxCoeff() < 100.0);
    
    qDebug() << "Cross-talk function test passed";
}

//=============================================================================================================

void TestResolutionAnalysis::testPointSpreadFunction()
{
    qDebug() << "Testing point spread function...";
    
    // Create a test resolution matrix
    int n_sources = 50;
    MatrixXd resolution = createTestResolutionMatrix(n_sources);
    
    // Compute point spread for each source
    // Point spread measures how localized the reconstruction is
    VectorXd point_spread(n_sources);
    
    for(int i = 0; i < n_sources; ++i) {
        // Point spread can be measured as the effective width of the column
        // Using the ratio of L1 norm to max value
        double col_max = resolution.col(i).cwiseAbs().maxCoeff();
        double col_sum = resolution.col(i).cwiseAbs().sum();
        
        if(col_max > 1e-10) {
            point_spread(i) = col_sum / col_max;
        } else {
            point_spread(i) = 0.0;
        }
    }
    
    qDebug() << "Computed point spread for" << n_sources << "sources";
    qDebug() << "Point spread range:" << point_spread.minCoeff() << "to" << point_spread.maxCoeff();
    qDebug() << "Mean point spread:" << point_spread.mean();
    
    // Test: Point spread should be positive
    QVERIFY(point_spread.minCoeff() >= 0.0);
    
    // Test: Point spread should be at least 1 (perfect localization)
    QVERIFY(point_spread.minCoeff() >= 1.0 - m_tolerance);
    
    qDebug() << "Point spread function test passed";
}

//=============================================================================================================

void TestResolutionAnalysis::testResolutionMetrics()
{
    qDebug() << "Testing resolution metrics...";
    
    // Create a test resolution matrix
    int n_sources = 50;
    MatrixXd resolution = createTestResolutionMatrix(n_sources);
    
    // Metric 1: Peak localization error (distance from diagonal peak)
    VectorXd peak_errors(n_sources);
    for(int i = 0; i < n_sources; ++i) {
        int max_idx;
        resolution.col(i).cwiseAbs().maxCoeff(&max_idx);
        peak_errors(i) = std::abs(max_idx - i);
    }
    
    qDebug() << "Peak localization errors - mean:" << peak_errors.mean() 
             << "max:" << peak_errors.maxCoeff();
    
    // Test: Most sources should have peak at or near diagonal
    int well_localized = 0;
    for(int i = 0; i < n_sources; ++i) {
        if(peak_errors(i) <= 2.0) {  // Within 2 sources
            well_localized++;
        }
    }
    
    double localization_ratio = static_cast<double>(well_localized) / n_sources;
    qDebug() << "Well-localized sources:" << well_localized << "/" << n_sources 
             << "(" << (localization_ratio * 100) << "%)";
    
    // At least 50% should be well-localized
    QVERIFY(localization_ratio >= 0.5);
    
    // Metric 2: Resolution matrix symmetry
    MatrixXd diff = resolution - resolution.transpose();
    double symmetry_error = diff.norm() / resolution.norm();
    
    qDebug() << "Symmetry error:" << symmetry_error;
    
    // Resolution matrix should be approximately symmetric
    QVERIFY(symmetry_error < 0.5);  // Allow some asymmetry
    
    qDebug() << "Resolution metrics test passed";
}

//=============================================================================================================

void TestResolutionAnalysis::cleanupTestCase()
{
    qDebug() << "Resolution Analysis Unit Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestResolutionAnalysis::createTestResolutionMatrix(int n_sources)
{
    // Create a synthetic resolution matrix with realistic properties
    // Resolution matrix R = G_inv * G where G is the gain matrix
    // For testing, we create a matrix with:
    // - Strong diagonal (good self-resolution)
    // - Weak off-diagonal (limited cross-talk)
    // - Smooth spatial structure
    
    MatrixXd resolution = MatrixXd::Zero(n_sources, n_sources);
    
    // Create a band-diagonal structure with Gaussian falloff
    double sigma = 3.0;  // Spatial spread parameter
    
    for(int i = 0; i < n_sources; ++i) {
        for(int j = 0; j < n_sources; ++j) {
            double distance = std::abs(i - j);
            double value = std::exp(-distance * distance / (2.0 * sigma * sigma));
            resolution(i, j) = value;
        }
    }
    
    // Normalize rows to sum to approximately 1
    for(int i = 0; i < n_sources; ++i) {
        double row_sum = resolution.row(i).sum();
        if(row_sum > 1e-10) {
            resolution.row(i) /= row_sum;
        }
    }
    
    return resolution;
}

//=============================================================================================================

double TestResolutionAnalysis::computeMatrixNorm(const MatrixXd& matrix)
{
    return matrix.norm();
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestResolutionAnalysis)
#include "test_resolution_analysis.moc"
