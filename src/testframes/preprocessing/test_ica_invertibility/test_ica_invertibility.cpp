//=============================================================================================================
/**
 * @file     test_ica_invertibility.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for ICA invertibility (Property 6)
 *           Feature: mne-python-to-cpp-migration, Property 6: ICA解混可逆性
 *           Validates: Requirements 3.1
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <preprocessing/ica.h>

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
#include <Eigen/Dense>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace PREPROCESSINGLIB;
using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestICAInvertibility
 *
 * @brief The TestICAInvertibility class provides property-based tests for ICA invertibility
 *
 */
class TestICAInvertibility: public QObject
{
    Q_OBJECT

public:
    TestICAInvertibility();

private slots:
    void initTestCase();
    void testMixingUnmixingInverse();
    void testICAInvertibilityProperty();
    void testRoundTripReconstruction();
    void cleanupTestCase();

private:
    // Helper methods for property testing
    MatrixXd generateRandomData(int n_channels, int n_samples);
    MatrixXd generateMixedSources(int n_sources, int n_samples, int n_channels);
    double computeRelativeError(const MatrixXd& matrix1, const MatrixXd& matrix2);
    
    // Test parameters
    double m_tolerance;
    QRandomGenerator* m_generator;
};

//=============================================================================================================

TestICAInvertibility::TestICAInvertibility()
: m_tolerance(0.01)  // 1% tolerance for numerical errors
, m_generator(QRandomGenerator::global())
{
}

//=============================================================================================================

void TestICAInvertibility::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting ICA Invertibility Property Tests";
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestICAInvertibility::testMixingUnmixingInverse()
{
    qDebug() << "Testing mixing and unmixing matrices are inverses...";
    
    // Setup parameters
    int n_channels = 10;
    int n_samples = 500;
    int n_components = 10;
    
    // Generate random data
    MatrixXd data = generateRandomData(n_channels, n_samples);
    
    // Fit ICA
    ICA ica(n_components, "fastica");
    ica.fit(data);
    
    // Get mixing and unmixing matrices
    MatrixXd mixing = ica.get_mixing_matrix();
    MatrixXd unmixing = ica.get_unmixing_matrix();
    
    qDebug() << "Mixing matrix dimensions:" << mixing.rows() << "x" << mixing.cols();
    qDebug() << "Unmixing matrix dimensions:" << unmixing.rows() << "x" << unmixing.cols();
    
    // Verify W * A ≈ I
    MatrixXd product = unmixing * mixing;
    MatrixXd identity = MatrixXd::Identity(n_components, n_components);
    
    double error = computeRelativeError(product, identity);
    
    qDebug() << "Relative error (W * A vs I):" << error;
    QVERIFY(error < m_tolerance);
    
    qDebug() << "Mixing-unmixing inverse test passed";
}

//=============================================================================================================

void TestICAInvertibility::testICAInvertibilityProperty()
{
    qDebug() << "Running ICA invertibility property test (100 iterations)...";
    
    int successful_tests = 0;
    int total_iterations = 100;
    
    // Feature: mne-python-to-cpp-migration, Property 6: ICA解混可逆性
    for(int iteration = 0; iteration < total_iterations; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(5, 15);
        int n_samples = m_generator->bounded(200, 600);
        int n_components = n_channels;  // Use all components
        
        // Generate random data
        MatrixXd data = generateRandomData(n_channels, n_samples);
        
        try {
            // Fit ICA
            ICA ica(n_components, "fastica");
            ica.fit(data);
            
            // Get mixing and unmixing matrices
            MatrixXd mixing = ica.get_mixing_matrix();
            MatrixXd unmixing = ica.get_unmixing_matrix();
            
            // Verify W * A ≈ I
            MatrixXd product = unmixing * mixing;
            MatrixXd identity = MatrixXd::Identity(n_components, n_components);
            
            double error = computeRelativeError(product, identity);
            
            if(error < m_tolerance) {
                successful_tests++;
            } else {
                qDebug() << "Iteration" << iteration << ": Invertibility error too large:" << error;
            }
        } catch(const std::exception& e) {
            qDebug() << "Iteration" << iteration << ": Exception caught:" << e.what();
            continue;
        } catch(...) {
            qDebug() << "Iteration" << iteration << ": Unknown exception caught";
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/" << total_iterations;
    
    // At least 90% of tests should pass (ICA may not always converge perfectly)
    QVERIFY2(successful_tests >= 90, 
             QString("Only %1 out of %2 tests passed (expected >= 90)")
             .arg(successful_tests).arg(total_iterations).toUtf8());
    
    qDebug() << "ICA invertibility property test completed successfully";
}

//=============================================================================================================

void TestICAInvertibility::testRoundTripReconstruction()
{
    qDebug() << "Testing round-trip reconstruction (mix -> unmix -> mix)...";
    
    int n_channels = 8;
    int n_samples = 400;
    int n_components = 8;
    
    // Generate random data
    MatrixXd data = generateRandomData(n_channels, n_samples);
    
    // Fit ICA
    ICA ica(n_components, "fastica");
    ica.fit(data);
    
    // Get sources
    MatrixXd sources = ica.get_sources(data);
    
    // Reconstruct data from sources
    MatrixXd mixing = ica.get_mixing_matrix();
    MatrixXd reconstructed = mixing * sources;
    
    // Verify reconstruction
    double error = computeRelativeError(data, reconstructed);
    
    qDebug() << "Round-trip reconstruction error:" << error;
    QVERIFY(error < m_tolerance);
    
    qDebug() << "Round-trip reconstruction test passed";
}

//=============================================================================================================

void TestICAInvertibility::cleanupTestCase()
{
    qDebug() << "ICA Invertibility Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestICAInvertibility::generateRandomData(int n_channels, int n_samples)
{
    MatrixXd data(n_channels, n_samples);
    
    // Generate data with some structure (not pure noise)
    // Create independent sources
    MatrixXd sources(n_channels, n_samples);
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_samples; ++j) {
            // Mix of sinusoids and noise
            double t = static_cast<double>(j) / n_samples;
            sources(i, j) = std::sin(2.0 * M_PI * (i + 1) * t) + 
                           0.1 * (m_generator->generateDouble() - 0.5);
        }
    }
    
    // Create random mixing matrix
    MatrixXd mixing(n_channels, n_channels);
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_channels; ++j) {
            mixing(i, j) = m_generator->generateDouble() - 0.5;
        }
    }
    
    // Mix sources
    data = mixing * sources;
    
    return data;
}

//=============================================================================================================

MatrixXd TestICAInvertibility::generateMixedSources(int n_sources, int n_samples, int n_channels)
{
    // Generate independent sources
    MatrixXd sources(n_sources, n_samples);
    for(int i = 0; i < n_sources; ++i) {
        for(int j = 0; j < n_samples; ++j) {
            double t = static_cast<double>(j) / n_samples;
            sources(i, j) = std::sin(2.0 * M_PI * (i + 1) * t);
        }
    }
    
    // Create mixing matrix
    MatrixXd mixing(n_channels, n_sources);
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_sources; ++j) {
            mixing(i, j) = m_generator->generateDouble() - 0.5;
        }
    }
    
    // Mix sources
    MatrixXd mixed = mixing * sources;
    
    return mixed;
}

//=============================================================================================================

double TestICAInvertibility::computeRelativeError(const MatrixXd& matrix1, const MatrixXd& matrix2)
{
    if(matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double diff_norm = (matrix1 - matrix2).norm();
    double ref_norm = matrix1.norm();
    
    if(ref_norm < 1e-10) {
        return diff_norm;
    }
    
    return diff_norm / ref_norm;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestICAInvertibility)
#include "test_ica_invertibility.moc"
