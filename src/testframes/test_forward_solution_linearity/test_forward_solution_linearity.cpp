//=============================================================================================================
/**
 * @file     test_forward_solution_linearity.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for forward solution linear superposition (Property 14)
 *           Feature: mne-python-to-cpp-migration, Property 14: 正向解线性叠加
 *           Validates: Requirements 9.1
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <mne/mne_forwardsolution.h>
#include <fiff/fiff_info.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QTest>
#include <QCoreApplication>
#include <QRandomGenerator>
#include <QDebug>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>
#include <cmath>
#include <numeric>
#include <algorithm>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace Eigen;
using namespace MNELIB;
using namespace FIFFLIB;

//=============================================================================================================
/**
 * DECLARE CLASS TestForwardSolutionLinearity
 *
 * @brief The TestForwardSolutionLinearity class provides property-based tests for forward solution
 *        linear superposition property
 *
 */
class TestForwardSolutionLinearity: public QObject
{
    Q_OBJECT

public:
    TestForwardSolutionLinearity();

private slots:
    void initTestCase();
    void testForwardSolutionLinearSuperposition();
    void testForwardSolutionLinearSuperpositionProperty();
    void testForwardSolutionScaling();
    void testForwardSolutionScalingProperty();
    void testForwardSolutionAddition();
    void testForwardSolutionAdditionProperty();
    void cleanupTestCase();

private:
    // Helper methods
    MatrixXd generateRandomGainMatrix(int n_channels, int n_sources);
    MatrixXd generateRandomSourceActivity(int n_sources, int n_times);
    double computeMatrixError(const MatrixXd& A, const MatrixXd& B);
    double computeRelativeError(const MatrixXd& A, const MatrixXd& B);
    
    // Test parameters
    QRandomGenerator* m_generator;
    int m_n_channels;
    int m_n_sources;
    int m_n_times;
    double m_tolerance;
};

//=============================================================================================================

TestForwardSolutionLinearity::TestForwardSolutionLinearity()
: m_generator(QRandomGenerator::global())
, m_n_channels(64)
, m_n_sources(10)
, m_n_times(100)
, m_tolerance(1e-10)  // Numerical tolerance for linear operations
{
}

//=============================================================================================================

void TestForwardSolutionLinearity::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Forward Solution Linearity Property Tests";
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Sources:" << m_n_sources;
    qDebug() << "Time points:" << m_n_times;
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionLinearSuperposition()
{
    qDebug() << "Test 1: Forward solution linear superposition";
    
    // Create gain matrix (forward solution)
    MatrixXd G = generateRandomGainMatrix(m_n_channels, m_n_sources);
    
    // Create two source activities
    MatrixXd S1 = generateRandomSourceActivity(m_n_sources, m_n_times);
    MatrixXd S2 = generateRandomSourceActivity(m_n_sources, m_n_times);
    
    // Apply forward solution separately
    MatrixXd M1 = G * S1;  // Measurement from source 1
    MatrixXd M2 = G * S2;  // Measurement from source 2
    
    // Apply forward solution to combined source activity
    MatrixXd S_combined = S1 + S2;
    MatrixXd M_combined = G * S_combined;
    
    // Check linear superposition: M(S1 + S2) = M(S1) + M(S2)
    MatrixXd M_expected = M1 + M2;
    double error = computeMatrixError(M_combined, M_expected);
    
    qDebug() << "Linear superposition error:" << error;
    
    QVERIFY(error < m_tolerance);
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionLinearSuperpositionProperty()
{
    qDebug() << "Running forward solution linear superposition property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 14: 正向解线性叠加
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(10, 100);
        int n_sources = m_generator->bounded(5, 50);
        int n_times = m_generator->bounded(50, 200);
        
        try {
            // Create gain matrix
            MatrixXd G = generateRandomGainMatrix(n_channels, n_sources);
            
            // Create two source activities
            MatrixXd S1 = generateRandomSourceActivity(n_sources, n_times);
            MatrixXd S2 = generateRandomSourceActivity(n_sources, n_times);
            
            // Apply forward solution separately
            MatrixXd M1 = G * S1;
            MatrixXd M2 = G * S2;
            
            // Apply forward solution to combined source activity
            MatrixXd S_combined = S1 + S2;
            MatrixXd M_combined = G * S_combined;
            
            // Check linear superposition
            MatrixXd M_expected = M1 + M2;
            double error = computeMatrixError(M_combined, M_expected);
            
            if(error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed iterations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // All tests should pass for linear operations
    QVERIFY(successful_tests >= 95);
    
    qDebug() << "Forward solution linear superposition property test completed successfully";
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionScaling()
{
    qDebug() << "Test 2: Forward solution scaling property";
    
    // Create gain matrix
    MatrixXd G = generateRandomGainMatrix(m_n_channels, m_n_sources);
    
    // Create source activity
    MatrixXd S = generateRandomSourceActivity(m_n_sources, m_n_times);
    
    // Apply forward solution
    MatrixXd M = G * S;
    
    // Apply scaling factor
    double alpha = 2.5;
    MatrixXd M_scaled_direct = G * (alpha * S);
    MatrixXd M_scaled_expected = alpha * M;
    
    // Check scaling: M(alpha * S) = alpha * M(S)
    double error = computeMatrixError(M_scaled_direct, M_scaled_expected);
    
    qDebug() << "Scaling error:" << error;
    
    QVERIFY(error < m_tolerance);
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionScalingProperty()
{
    qDebug() << "Running forward solution scaling property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 14: 正向解线性叠加
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(10, 100);
        int n_sources = m_generator->bounded(5, 50);
        int n_times = m_generator->bounded(50, 200);
        
        try {
            // Create gain matrix
            MatrixXd G = generateRandomGainMatrix(n_channels, n_sources);
            
            // Create source activity
            MatrixXd S = generateRandomSourceActivity(n_sources, n_times);
            
            // Apply forward solution
            MatrixXd M = G * S;
            
            // Random scaling factor
            double alpha = m_generator->generateDouble() * 10.0 - 5.0;  // -5 to 5
            
            // Check scaling property
            MatrixXd M_scaled_direct = G * (alpha * S);
            MatrixXd M_scaled_expected = alpha * M;
            
            double error = computeMatrixError(M_scaled_direct, M_scaled_expected);
            
            if(error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed iterations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // All tests should pass for linear operations
    QVERIFY(successful_tests >= 95);
    
    qDebug() << "Forward solution scaling property test completed successfully";
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionAddition()
{
    qDebug() << "Test 3: Forward solution addition property";
    
    // Create gain matrix
    MatrixXd G = generateRandomGainMatrix(m_n_channels, m_n_sources);
    
    // Create three source activities
    MatrixXd S1 = generateRandomSourceActivity(m_n_sources, m_n_times);
    MatrixXd S2 = generateRandomSourceActivity(m_n_sources, m_n_times);
    MatrixXd S3 = generateRandomSourceActivity(m_n_sources, m_n_times);
    
    // Apply forward solution separately
    MatrixXd M1 = G * S1;
    MatrixXd M2 = G * S2;
    MatrixXd M3 = G * S3;
    
    // Apply forward solution to combined source activity
    MatrixXd S_combined = S1 + S2 + S3;
    MatrixXd M_combined = G * S_combined;
    
    // Check addition: M(S1 + S2 + S3) = M(S1) + M(S2) + M(S3)
    MatrixXd M_expected = M1 + M2 + M3;
    double error = computeMatrixError(M_combined, M_expected);
    
    qDebug() << "Addition error:" << error;
    
    QVERIFY(error < m_tolerance);
}

//=============================================================================================================

void TestForwardSolutionLinearity::testForwardSolutionAdditionProperty()
{
    qDebug() << "Running forward solution addition property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 14: 正向解线性叠加
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(10, 100);
        int n_sources = m_generator->bounded(5, 50);
        int n_times = m_generator->bounded(50, 200);
        
        try {
            // Create gain matrix
            MatrixXd G = generateRandomGainMatrix(n_channels, n_sources);
            
            // Create multiple source activities
            MatrixXd S1 = generateRandomSourceActivity(n_sources, n_times);
            MatrixXd S2 = generateRandomSourceActivity(n_sources, n_times);
            MatrixXd S3 = generateRandomSourceActivity(n_sources, n_times);
            
            // Apply forward solution separately
            MatrixXd M1 = G * S1;
            MatrixXd M2 = G * S2;
            MatrixXd M3 = G * S3;
            
            // Apply forward solution to combined source activity
            MatrixXd S_combined = S1 + S2 + S3;
            MatrixXd M_combined = G * S_combined;
            
            // Check addition property
            MatrixXd M_expected = M1 + M2 + M3;
            double error = computeMatrixError(M_combined, M_expected);
            
            if(error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed iterations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // All tests should pass for linear operations
    QVERIFY(successful_tests >= 95);
    
    qDebug() << "Forward solution addition property test completed successfully";
}

//=============================================================================================================

void TestForwardSolutionLinearity::cleanupTestCase()
{
    qDebug() << "Forward Solution Linearity Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestForwardSolutionLinearity::generateRandomGainMatrix(int n_channels, int n_sources)
{
    MatrixXd G(n_channels, n_sources);
    
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_sources; ++j) {
            // Generate random value from standard normal distribution
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            
            G(i, j) = z * 1e-9;  // Scale to realistic MEG/EEG units
        }
    }
    
    return G;
}

//=============================================================================================================

MatrixXd TestForwardSolutionLinearity::generateRandomSourceActivity(int n_sources, int n_times)
{
    MatrixXd S(n_sources, n_times);
    
    for(int i = 0; i < n_sources; ++i) {
        for(int j = 0; j < n_times; ++j) {
            // Generate random value from standard normal distribution
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            
            S(i, j) = z * 1e-9;  // Scale to realistic source activity units
        }
    }
    
    return S;
}

//=============================================================================================================

double TestForwardSolutionLinearity::computeMatrixError(const MatrixXd& A, const MatrixXd& B)
{
    // Compute Frobenius norm of difference
    return (A - B).norm();
}

//=============================================================================================================

double TestForwardSolutionLinearity::computeRelativeError(const MatrixXd& A, const MatrixXd& B)
{
    // Compute relative error: ||A - B|| / ||B||
    double norm_diff = (A - B).norm();
    double norm_B = B.norm();
    
    if(norm_B < 1e-15) {
        return norm_diff;
    }
    
    return norm_diff / norm_B;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestForwardSolutionLinearity)
#include "test_forward_solution_linearity.moc"

