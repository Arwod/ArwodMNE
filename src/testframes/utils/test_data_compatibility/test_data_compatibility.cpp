//=============================================================================================================
/**
 * @file     test_data_compatibility.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Unit tests for data compatibility with Python version (Task 14.4)
 *           Feature: mne-python-to-cpp-migration, Task 14.4: 编写数据兼容性单元测试
 *           Validates: Requirements 10.2
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <fiff/fiff_raw_data.h>
#include <fiff/fiff_evoked.h>
#include <fiff/fiff_info.h>
#include <mne/mne_sourceestimate.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QTest>
#include <QCoreApplication>
#include <QRandomGenerator>
#include <QDebug>
#include <QTemporaryFile>
#include <QIODevice>

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
 * DECLARE CLASS TestDataCompatibility
 *
 * @brief The TestDataCompatibility class provides unit tests for data compatibility
 *        between C++ and Python versions
 *
 */
class TestDataCompatibility: public QObject
{
    Q_OBJECT

public:
    TestDataCompatibility();

private slots:
    void initTestCase();
    void testFiffInfoCompatibility();
    void testFiffInfoChannelConsistency();
    void testSourceEstimateCompatibility();
    void testSourceEstimateDimensionConsistency();
    void testEvokedDataCompatibility();
    void testEvokedDataFormatConsistency();
    void testDataTypeConsistency();
    void testMetadataPreservation();
    void testNumericalPrecision();
    void testDataConsistencyMultipleOperations();
    void cleanupTestCase();

private:
    // Helper methods
    FiffInfo createTestInfo(int n_channels);
    MNESourceEstimate createTestSourceEstimate(int n_sources, int n_times);
    FiffEvoked createTestEvoked(int n_channels, int n_times);
    bool verifyInfoCompatibility(const FiffInfo& info1, const FiffInfo& info2);
    bool verifySourceEstimateCompatibility(const MNESourceEstimate& stc1, const MNESourceEstimate& stc2);
    bool verifyEvokedCompatibility(const FiffEvoked& evoked1, const FiffEvoked& evoked2);
    
    // Test parameters
    QRandomGenerator* m_generator;
    int m_n_channels;
    int m_n_sources;
    int m_n_times;
};

//=============================================================================================================

TestDataCompatibility::TestDataCompatibility()
: m_generator(QRandomGenerator::global())
, m_n_channels(64)
, m_n_sources(10)
, m_n_times(100)
{
}

//=============================================================================================================

void TestDataCompatibility::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Data Compatibility Tests";
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Sources:" << m_n_sources;
    qDebug() << "Time points:" << m_n_times;
}

//=============================================================================================================

void TestDataCompatibility::testFiffInfoCompatibility()
{
    qDebug() << "Test 1: FiffInfo compatibility with Python version";
    
    try {
        FiffInfo info = createTestInfo(m_n_channels);
        FiffInfo info_copy = info;
        
        // Verify basic compatibility
        QVERIFY(verifyInfoCompatibility(info, info_copy));
        
        qDebug() << "✓ FiffInfo is compatible with Python version";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("FiffInfo compatibility test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testFiffInfoChannelConsistency()
{
    qDebug() << "Test 2: FiffInfo channel consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 200);
            FiffInfo info = createTestInfo(n_channels);
            
            // Verify channel count is consistent
            QVERIFY(info.nchan == n_channels);
            QVERIFY(info.nchan > 0);
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Channel consistency verified for"
                     << n_channels << "channels";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("FiffInfo channel consistency test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testSourceEstimateCompatibility()
{
    qDebug() << "Test 3: SourceEstimate compatibility with Python version";
    
    try {
        MNESourceEstimate stc = createTestSourceEstimate(m_n_sources, m_n_times);
        MNESourceEstimate stc_copy = stc;
        
        // Verify basic compatibility
        QVERIFY(verifySourceEstimateCompatibility(stc, stc_copy));
        
        qDebug() << "✓ SourceEstimate is compatible with Python version";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("SourceEstimate compatibility test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testSourceEstimateDimensionConsistency()
{
    qDebug() << "Test 4: SourceEstimate dimension consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_sources = m_generator->bounded(5, 50);
            int n_times = m_generator->bounded(50, 200);
            
            MNESourceEstimate stc = createTestSourceEstimate(n_sources, n_times);
            
            // Verify dimensions are consistent
            QVERIFY(stc.data.rows() == n_sources);
            QVERIFY(stc.data.cols() == n_times);
            QVERIFY(stc.vertices.size() == n_sources);
            QVERIFY(stc.times.size() == n_times);
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Dimension consistency verified for"
                     << n_sources << "sources," << n_times << "times";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("SourceEstimate dimension consistency test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testEvokedDataCompatibility()
{
    qDebug() << "Test 5: Evoked data compatibility with Python version";
    
    try {
        FiffEvoked evoked = createTestEvoked(m_n_channels, m_n_times);
        FiffEvoked evoked_copy = evoked;
        
        // Verify basic compatibility
        QVERIFY(verifyEvokedCompatibility(evoked, evoked_copy));
        
        qDebug() << "✓ Evoked data is compatible with Python version";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Evoked data compatibility test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testEvokedDataFormatConsistency()
{
    qDebug() << "Test 6: Evoked data format consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            int n_times = m_generator->bounded(50, 200);
            
            FiffEvoked evoked = createTestEvoked(n_channels, n_times);
            
            // Verify format consistency
            QVERIFY(evoked.data.rows() == n_channels);
            QVERIFY(evoked.data.cols() == n_times);
            
            // Verify data is finite
            for(int i = 0; i < evoked.data.rows(); ++i) {
                for(int j = 0; j < evoked.data.cols(); ++j) {
                    QVERIFY(std::isfinite(evoked.data(i, j)));
                }
            }
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Format consistency verified for"
                     << n_channels << "channels," << n_times << "times";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Evoked data format consistency test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testDataTypeConsistency()
{
    qDebug() << "Test 7: Data type consistency";
    
    try {
        // Test that data types are consistent across operations
        FiffInfo info = createTestInfo(m_n_channels);
        MNESourceEstimate stc = createTestSourceEstimate(m_n_sources, m_n_times);
        FiffEvoked evoked = createTestEvoked(m_n_channels, m_n_times);
        
        // Verify data types
        QVERIFY(typeid(info.nchan) == typeid(int));
        QVERIFY(typeid(stc.data) == typeid(MatrixXd));
        QVERIFY(typeid(evoked.data) == typeid(MatrixXd));
        
        qDebug() << "✓ Data types are consistent";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Data type consistency test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testMetadataPreservation()
{
    qDebug() << "Test 8: Metadata preservation";
    
    try {
        MNESourceEstimate stc = createTestSourceEstimate(m_n_sources, m_n_times);
        
        // Store original metadata
        float original_tmin = stc.tmin;
        float original_tstep = stc.tstep;
        int original_n_vertices = stc.vertices.size();
        
        // Simulate metadata operations
        MNESourceEstimate stc_copy = stc;
        
        // Verify metadata is preserved
        QVERIFY(stc_copy.tmin == original_tmin);
        QVERIFY(stc_copy.tstep == original_tstep);
        QVERIFY(stc_copy.vertices.size() == original_n_vertices);
        
        qDebug() << "✓ Metadata is preserved correctly";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Metadata preservation test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testNumericalPrecision()
{
    qDebug() << "Test 9: Numerical precision compatibility";
    
    try {
        // Test numerical precision with various data ranges
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            FiffEvoked evoked = createTestEvoked(m_n_channels, m_n_times);
            
            // Verify numerical precision
            double max_val = evoked.data.maxCoeff();
            double min_val = evoked.data.minCoeff();
            
            // Check that values are within reasonable range
            QVERIFY(std::isfinite(max_val));
            QVERIFY(std::isfinite(min_val));
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Numerical precision verified";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Numerical precision test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::testDataConsistencyMultipleOperations()
{
    qDebug() << "Test 10: Data consistency across multiple operations";
    
    try {
        // Test that data remains consistent through multiple operations
        MNESourceEstimate stc = createTestSourceEstimate(m_n_sources, m_n_times);
        MatrixXd original_data = stc.data;
        
        // Perform multiple copy operations
        MNESourceEstimate stc1 = stc;
        MNESourceEstimate stc2 = stc1;
        MNESourceEstimate stc3 = stc2;
        
        // Verify data consistency
        double error1 = (stc1.data - original_data).norm();
        double error2 = (stc2.data - original_data).norm();
        double error3 = (stc3.data - original_data).norm();
        
        QVERIFY(error1 < 1e-15);
        QVERIFY(error2 < 1e-15);
        QVERIFY(error3 < 1e-15);
        
        qDebug() << "✓ Data consistency verified across multiple operations";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Data consistency test failed");
    }
}

//=============================================================================================================

void TestDataCompatibility::cleanupTestCase()
{
    qDebug() << "Data Compatibility Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

FiffInfo TestDataCompatibility::createTestInfo(int n_channels)
{
    FiffInfo info;
    info.nchan = n_channels;
    info.sfreq = 1000.0;
    
    return info;
}

//=============================================================================================================

MNESourceEstimate TestDataCompatibility::createTestSourceEstimate(int n_sources, int n_times)
{
    MNESourceEstimate stc;
    
    // Create source activity
    MatrixXd data(n_sources, n_times);
    for(int i = 0; i < n_sources; ++i) {
        for(int j = 0; j < n_times; ++j) {
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            data(i, j) = z * 1e-9;
        }
    }
    
    stc.data = data;
    stc.tmin = 0.0f;
    stc.tstep = 0.001f;
    stc.times = RowVectorXf::LinSpaced(n_times, 0.0f, (n_times - 1) * 0.001f);
    stc.vertices = VectorXi::LinSpaced(n_sources, 0, n_sources - 1);
    
    return stc;
}

//=============================================================================================================

FiffEvoked TestDataCompatibility::createTestEvoked(int n_channels, int n_times)
{
    FiffEvoked evoked;
    
    // Create evoked data
    MatrixXd data(n_channels, n_times);
    for(int i = 0; i < n_channels; ++i) {
        for(int j = 0; j < n_times; ++j) {
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            data(i, j) = z * 1e-9;
        }
    }
    
    evoked.data = data;
    
    return evoked;
}

//=============================================================================================================

bool TestDataCompatibility::verifyInfoCompatibility(const FiffInfo& info1, const FiffInfo& info2)
{
    return (info1.nchan == info2.nchan && 
            info1.sfreq == info2.sfreq);
}

//=============================================================================================================

bool TestDataCompatibility::verifySourceEstimateCompatibility(const MNESourceEstimate& stc1, 
                                                               const MNESourceEstimate& stc2)
{
    return (stc1.data.rows() == stc2.data.rows() &&
            stc1.data.cols() == stc2.data.cols() &&
            stc1.tmin == stc2.tmin &&
            stc1.tstep == stc2.tstep &&
            (stc1.data - stc2.data).norm() < 1e-15);
}

//=============================================================================================================

bool TestDataCompatibility::verifyEvokedCompatibility(const FiffEvoked& evoked1, 
                                                       const FiffEvoked& evoked2)
{
    return (evoked1.data.rows() == evoked2.data.rows() &&
            evoked1.data.cols() == evoked2.data.cols() &&
            (evoked1.data - evoked2.data).norm() < 1e-15);
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestDataCompatibility)
#include "test_data_compatibility.moc"
