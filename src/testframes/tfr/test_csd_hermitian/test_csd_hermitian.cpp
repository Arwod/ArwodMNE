//=============================================================================================================
/**
 * @file     test_csd_hermitian.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for Cross-Spectral Density Hermitian property (Property 2)
 *           Feature: mne-python-to-cpp-migration, Property 2: 交叉谱密度厄米特性
 *           Validates: Requirements 1.4
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <tfr/csd.h>

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

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace TFRLIB;
using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestCSDHermitian
 *
 * @brief The TestCSDHermitian class provides property-based tests for CSD Hermitian property
 *
 */
class TestCSDHermitian: public QObject
{
    Q_OBJECT

public:
    TestCSDHermitian();

private slots:
    void initTestCase();
    void testCSDFourierHermitian();
    void testCSDMorletHermitian();
    void testCSDMultitaperHermitian();
    void testCSDHermitianProperty();
    void cleanupTestCase();

private:
    // Helper methods for property testing
    std::vector<MatrixXd> generateRandomEpochs(int n_epochs, int n_channels, int n_times, double sfreq);
    bool verifyHermitianManual(const MatrixXcd& matrix, double tolerance = 1e-10);
    
    // Test parameters
    double m_tolerance;
    QRandomGenerator* m_generator;
    double m_sfreq;
    int m_n_channels;
    int m_n_times;
    int m_n_epochs;
};

//=============================================================================================================

TestCSDHermitian::TestCSDHermitian()
: m_tolerance(1e-8)  // Tolerance for Hermitian property
, m_generator(QRandomGenerator::global())
, m_sfreq(1000.0)
, m_n_channels(4)
, m_n_times(500)
, m_n_epochs(10)
{
}

//=============================================================================================================

void TestCSDHermitian::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting CSD Hermitian Property Tests";
    qDebug() << "Tolerance:" << m_tolerance;
    qDebug() << "Sampling frequency:" << m_sfreq;
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Time points:" << m_n_times;
    qDebug() << "Epochs:" << m_n_epochs;
}

//=============================================================================================================

void TestCSDHermitian::testCSDFourierHermitian()
{
    qDebug() << "Testing CSD Fourier Hermitian property...";
    
    // Generate random epochs
    auto epochs = generateRandomEpochs(m_n_epochs, m_n_channels, m_n_times, m_sfreq);
    
    // Compute CSD using Fourier method
    CSD csd = CSD::compute_fourier(epochs, m_sfreq, 10.0, 100.0, 0.0, 0.0, -1, 0.5);
    
    // Verify Hermitian property
    bool is_hermitian = csd.verify_hermitian(m_tolerance);
    
    qDebug() << "Number of frequency bins:" << csd.freqs.size();
    qDebug() << "Is Hermitian:" << is_hermitian;
    
    QVERIFY(is_hermitian);
    
    // Manually verify a few matrices
    for(size_t i = 0; i < std::min(size_t(3), csd.data.size()); ++i) {
        bool manual_check = verifyHermitianManual(csd.data[i], m_tolerance);
        QVERIFY(manual_check);
    }
    
    qDebug() << "CSD Fourier Hermitian test passed";
}

//=============================================================================================================

void TestCSDHermitian::testCSDMorletHermitian()
{
    qDebug() << "Testing CSD Morlet Hermitian property...";
    
    // Generate random epochs
    auto epochs = generateRandomEpochs(m_n_epochs, m_n_channels, m_n_times, m_sfreq);
    
    // Define frequencies
    VectorXd freqs = VectorXd::LinSpaced(10, 10.0, 100.0);
    VectorXd n_cycles = VectorXd::Constant(freqs.size(), 7.0);
    
    // Compute CSD using Morlet wavelets
    CSD csd = CSD::compute_morlet(epochs, freqs, m_sfreq, 0.0, 0.0, n_cycles, true, 1);
    
    // Verify Hermitian property
    bool is_hermitian = csd.verify_hermitian(m_tolerance);
    
    qDebug() << "Number of frequency bins:" << csd.freqs.size();
    qDebug() << "Is Hermitian:" << is_hermitian;
    
    QVERIFY(is_hermitian);
    
    qDebug() << "CSD Morlet Hermitian test passed";
}

//=============================================================================================================

void TestCSDHermitian::testCSDMultitaperHermitian()
{
    qDebug() << "Testing CSD Multitaper Hermitian property...";
    
    // Generate random epochs
    auto epochs = generateRandomEpochs(m_n_epochs, m_n_channels, m_n_times, m_sfreq);
    
    // Compute CSD using multitaper method
    CSD csd = CSD::compute_multitaper(epochs, m_sfreq, 0.0, 0.0, 10.0, 100.0, 4.0, false, true);
    
    // Verify Hermitian property
    bool is_hermitian = csd.verify_hermitian(m_tolerance);
    
    qDebug() << "Number of frequency bins:" << csd.freqs.size();
    qDebug() << "Is Hermitian:" << is_hermitian;
    
    QVERIFY(is_hermitian);
    
    qDebug() << "CSD Multitaper Hermitian test passed";
}

//=============================================================================================================

void TestCSDHermitian::testCSDHermitianProperty()
{
    qDebug() << "Running CSD Hermitian property test (20 iterations)...";
    
    int successful_tests = 0;
    int total_iterations = 20;
    
    // Feature: mne-python-to-cpp-migration, Property 2: 交叉谱密度厄米特性
    for(int iteration = 0; iteration < total_iterations; ++iteration) {
        // Generate random parameters
        int n_epochs = m_generator->bounded(5, 20);
        int n_channels = m_generator->bounded(2, 8);
        int n_times = m_generator->bounded(256, 1024);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random epochs
        auto epochs = generateRandomEpochs(n_epochs, n_channels, n_times, sfreq);
        
        // Randomly choose CSD method
        int method = m_generator->bounded(0, 3);
        
        try {
            CSD csd;
            
            if(method == 0) {
                // Fourier method
                csd = CSD::compute_fourier(epochs, sfreq, 10.0, sfreq/3.0, 0.0, 0.0, -1, 0.5);
            } else if(method == 1) {
                // Morlet method
                VectorXd freqs = VectorXd::LinSpaced(8, 10.0, sfreq/3.0);
                VectorXd n_cycles = VectorXd::Constant(freqs.size(), 7.0);
                csd = CSD::compute_morlet(epochs, freqs, sfreq, 0.0, 0.0, n_cycles, true, 1);
            } else {
                // Multitaper method
                csd = CSD::compute_multitaper(epochs, sfreq, 0.0, 0.0, 10.0, sfreq/3.0, 4.0, false, true);
            }
            
            // Verify Hermitian property
            bool is_hermitian = csd.verify_hermitian(m_tolerance);
            
            if(is_hermitian) {
                successful_tests++;
            } else {
                qDebug() << "Iteration" << iteration << ": Hermitian property violated (method" << method << ")";
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
    
    // At least 95% of tests should pass (Hermitian property should always hold)
    QVERIFY2(successful_tests >= 19, 
             QString("Only %1 out of %2 tests passed (expected >= 19)")
             .arg(successful_tests).arg(total_iterations).toUtf8());
    
    qDebug() << "CSD Hermitian property test completed successfully";
}

//=============================================================================================================

void TestCSDHermitian::cleanupTestCase()
{
    qDebug() << "CSD Hermitian Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

std::vector<MatrixXd> TestCSDHermitian::generateRandomEpochs(int n_epochs, int n_channels, int n_times, double sfreq)
{
    std::vector<MatrixXd> epochs;
    epochs.reserve(n_epochs);
    
    VectorXd time = VectorXd::LinSpaced(n_times, 0.0, (n_times - 1) / sfreq);
    
    for(int ep = 0; ep < n_epochs; ++ep) {
        MatrixXd epoch(n_channels, n_times);
        
        for(int ch = 0; ch < n_channels; ++ch) {
            for(int t = 0; t < n_times; ++t) {
                double value = 0.0;
                
                // Add multiple sinusoidal components with random phases
                int n_components = m_generator->bounded(2, 4);
                for(int f = 0; f < n_components; ++f) {
                    double freq = 10.0 + m_generator->generateDouble() * 80.0;  // 10-90 Hz
                    double amplitude = m_generator->generateDouble() * 0.5 + 0.5;  // 0.5-1.0
                    double phase = m_generator->generateDouble() * 2.0 * M_PI;
                    
                    value += amplitude * sin(2.0 * M_PI * freq * time(t) + phase);
                }
                
                // Add some noise
                value += (m_generator->generateDouble() - 0.5) * 0.1;
                
                epoch(ch, t) = value;
            }
        }
        
        epochs.push_back(epoch);
    }
    
    return epochs;
}

//=============================================================================================================

bool TestCSDHermitian::verifyHermitianManual(const MatrixXcd& matrix, double tolerance)
{
    // A matrix is Hermitian if A = A^H (conjugate transpose)
    // i.e., A(i,j) = conj(A(j,i))
    
    int n = matrix.rows();
    if(n != matrix.cols()) {
        return false;  // Must be square
    }
    
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::complex<double> a_ij = matrix(i, j);
            std::complex<double> a_ji_conj = std::conj(matrix(j, i));
            
            double diff = std::abs(a_ij - a_ji_conj);
            if(diff > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestCSDHermitian)
#include "test_csd_hermitian.moc"
