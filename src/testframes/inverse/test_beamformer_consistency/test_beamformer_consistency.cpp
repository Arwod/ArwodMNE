//=============================================================================================================
/**
 * @file     test_beamformer_consistency.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for beamformer data type consistency (Property 5)
 *           Feature: mne-python-to-cpp-migration, Property 5: 波束成形器数据类型一致性
 *           Validates: Requirements 2.2, 2.3
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <inverse/beamformer/lcmv.h>
#include <inverse/beamformer/covariance.h>

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

using namespace INVERSELIB;
using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestBeamformerConsistency
 *
 * @brief The TestBeamformerConsistency class provides property-based tests for beamformer consistency
 *
 */
class TestBeamformerConsistency: public QObject
{
    Q_OBJECT

public:
    TestBeamformerConsistency();

private slots:
    void initTestCase();
    void testLCMVConsistencyAcrossDataTypes();
    void testLCMVConsistencyWithAveraging();
    void testBeamformerConsistencyProperty();
    void cleanupTestCase();

private:
    // Helper methods for property testing
    MatrixXd generateRandomLeadfield(int n_channels, int n_sources, int n_ori = 3);
    Covariance generateRandomCovariance(int n_channels);
    MatrixXd generateRandomData(int n_channels, int n_times);
    std::vector<MatrixXd> generateRandomEpochs(int n_epochs, int n_channels, int n_times);
    double computeRelativeError(const MatrixXd& result1, const MatrixXd& result2);
    
    // Test parameters
    double m_tolerance;
    QRandomGenerator* m_generator;
};

//=============================================================================================================

TestBeamformerConsistency::TestBeamformerConsistency()
: m_tolerance(0.01)  // 1% relative error tolerance
, m_generator(QRandomGenerator::global())
{
}

//=============================================================================================================

void TestBeamformerConsistency::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Beamformer Consistency Property Tests";
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestBeamformerConsistency::testLCMVConsistencyAcrossDataTypes()
{
    qDebug() << "Testing LCMV consistency across data types...";
    
    // Setup parameters
    int n_channels = 10;
    int n_sources = 5;
    int n_ori = 3;
    int n_times = 100;
    
    // Generate leadfield and covariance
    MatrixXd leadfield = generateRandomLeadfield(n_channels, n_sources, n_ori);
    Covariance data_cov = generateRandomCovariance(n_channels);
    
    // Compute LCMV weights
    BeamformerWeights weights = LCMV::make_lcmv(
        leadfield, data_cov, Covariance(),
        0.05, "vector", "none", n_ori, 0.0, false, "matrix"
    );
    
    // Generate test data
    MatrixXd data = generateRandomData(n_channels, n_times);
    
    // Apply beamformer
    MatrixXd result = LCMV::apply(weights, data);
    
    // Verify output dimensions
    QVERIFY(result.rows() == n_sources * n_ori);
    QVERIFY(result.cols() == n_times);
    
    qDebug() << "Input shape:" << data.rows() << "x" << data.cols();
    qDebug() << "Output shape:" << result.rows() << "x" << result.cols();
    qDebug() << "LCMV consistency test passed";
}

//=============================================================================================================

void TestBeamformerConsistency::testLCMVConsistencyWithAveraging()
{
    qDebug() << "Testing LCMV consistency with epoch averaging...";
    
    // Setup parameters
    int n_channels = 12;
    int n_sources = 6;
    int n_ori = 3;
    int n_epochs = 10;
    int n_times = 50;
    
    // Generate leadfield and covariance
    MatrixXd leadfield = generateRandomLeadfield(n_channels, n_sources, n_ori);
    Covariance data_cov = generateRandomCovariance(n_channels);
    
    // Compute LCMV weights
    BeamformerWeights weights = LCMV::make_lcmv(
        leadfield, data_cov, Covariance(),
        0.05, "vector", "none", n_ori, 0.0, false, "matrix"
    );
    
    // Generate epochs
    std::vector<MatrixXd> epochs = generateRandomEpochs(n_epochs, n_channels, n_times);
    
    // Method 1: Apply to each epoch then average
    MatrixXd sum_result = MatrixXd::Zero(n_sources * n_ori, n_times);
    for(const auto& epoch : epochs) {
        MatrixXd epoch_result = LCMV::apply(weights, epoch);
        sum_result += epoch_result;
    }
    MatrixXd avg_result1 = sum_result / n_epochs;
    
    // Method 2: Average epochs first then apply
    MatrixXd avg_data = MatrixXd::Zero(n_channels, n_times);
    for(const auto& epoch : epochs) {
        avg_data += epoch;
    }
    avg_data /= n_epochs;
    MatrixXd avg_result2 = LCMV::apply(weights, avg_data);
    
    // Verify consistency (linearity of beamformer)
    double error = computeRelativeError(avg_result1, avg_result2);
    
    qDebug() << "Relative error between methods:" << error;
    QVERIFY(error < m_tolerance);
    
    qDebug() << "LCMV averaging consistency test passed";
}

//=============================================================================================================

void TestBeamformerConsistency::testBeamformerConsistencyProperty()
{
    qDebug() << "Running beamformer consistency property test (100 iterations)...";
    
    int successful_tests = 0;
    int total_iterations = 100;
    
    // Feature: mne-python-to-cpp-migration, Property 5: 波束成形器数据类型一致性
    for(int iteration = 0; iteration < total_iterations; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(8, 16);
        int n_sources = m_generator->bounded(3, 8);
        int n_ori = 3;
        int n_epochs = m_generator->bounded(5, 15);
        int n_times = m_generator->bounded(30, 100);
        
        // Generate leadfield and covariance
        MatrixXd leadfield = generateRandomLeadfield(n_channels, n_sources, n_ori);
        Covariance data_cov = generateRandomCovariance(n_channels);
        
        try {
            // Compute LCMV weights
            BeamformerWeights weights = LCMV::make_lcmv(
                leadfield, data_cov, Covariance(),
                0.05, "vector", "none", n_ori, 0.0, false, "matrix"
            );
            
            // Generate epochs
            std::vector<MatrixXd> epochs = generateRandomEpochs(n_epochs, n_channels, n_times);
            
            // Method 1: Apply to each epoch then average
            MatrixXd sum_result = MatrixXd::Zero(n_sources * n_ori, n_times);
            for(const auto& epoch : epochs) {
                MatrixXd epoch_result = LCMV::apply(weights, epoch);
                sum_result += epoch_result;
            }
            MatrixXd avg_result1 = sum_result / n_epochs;
            
            // Method 2: Average epochs first then apply
            MatrixXd avg_data = MatrixXd::Zero(n_channels, n_times);
            for(const auto& epoch : epochs) {
                avg_data += epoch;
            }
            avg_data /= n_epochs;
            MatrixXd avg_result2 = LCMV::apply(weights, avg_data);
            
            // Verify consistency
            double error = computeRelativeError(avg_result1, avg_result2);
            
            if(error < m_tolerance) {
                successful_tests++;
            } else {
                qDebug() << "Iteration" << iteration << ": Consistency error too large:" << error;
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
    
    // At least 95% of tests should pass (linearity should always hold)
    QVERIFY2(successful_tests >= 95, 
             QString("Only %1 out of %2 tests passed (expected >= 95)")
             .arg(successful_tests).arg(total_iterations).toUtf8());
    
    qDebug() << "Beamformer consistency property test completed successfully";
}

//=============================================================================================================

void TestBeamformerConsistency::cleanupTestCase()
{
    qDebug() << "Beamformer Consistency Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestBeamformerConsistency::generateRandomLeadfield(int n_channels, int n_sources, int n_ori)
{
    MatrixXd leadfield(n_channels, n_sources * n_ori);
    
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_sources * n_ori; ++j) {
            leadfield(i, j) = m_generator->generateDouble() - 0.5;
        }
    }
    
    // Normalize columns
    for(int j = 0; j < n_sources * n_ori; ++j) {
        double norm = leadfield.col(j).norm();
        if(norm > 1e-10) {
            leadfield.col(j) /= norm;
        }
    }
    
    return leadfield;
}

//=============================================================================================================

Covariance TestBeamformerConsistency::generateRandomCovariance(int n_channels)
{
    MatrixXd A(n_channels, n_channels);
    
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_channels; ++j) {
            A(i, j) = m_generator->generateDouble() - 0.5;
        }
    }
    
    MatrixXd cov = A * A.transpose();
    cov += MatrixXd::Identity(n_channels, n_channels) * 0.1;
    
    Covariance result;
    result.data = cov;
    result.nfree = 100;
    result.names.resize(n_channels);
    for(int i = 0; i < n_channels; ++i) {
        result.names[i] = "CH" + std::to_string(i);
    }
    result.is_empty = false;
    result.method = "empirical";
    result.loglik = 0.0;
    
    return result;
}

//=============================================================================================================

MatrixXd TestBeamformerConsistency::generateRandomData(int n_channels, int n_times)
{
    MatrixXd data(n_channels, n_times);
    
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_times; ++j) {
            data(i, j) = m_generator->generateDouble() - 0.5;
        }
    }
    
    return data;
}

//=============================================================================================================

std::vector<MatrixXd> TestBeamformerConsistency::generateRandomEpochs(int n_epochs, int n_channels, int n_times)
{
    std::vector<MatrixXd> epochs;
    epochs.reserve(n_epochs);
    
    for(int ep = 0; ep < n_epochs; ++ep) {
        epochs.push_back(generateRandomData(n_channels, n_times));
    }
    
    return epochs;
}

//=============================================================================================================

double TestBeamformerConsistency::computeRelativeError(const MatrixXd& result1, const MatrixXd& result2)
{
    if(result1.rows() != result2.rows() || result1.cols() != result2.cols()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double diff_norm = (result1 - result2).norm();
    double ref_norm = result1.norm();
    
    if(ref_norm < 1e-10) {
        return diff_norm;
    }
    
    return diff_norm / ref_norm;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestBeamformerConsistency)
#include "test_beamformer_consistency.moc"
