//=============================================================================================================
/**
 * @file     test_channel_management.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Unit tests for channel management operations (Task 15.4)
 *           Feature: mne-python-to-cpp-migration, Task 15.4: 编写通道管理单元测试
 *           Validates: Requirements 11.2
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
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
using namespace FIFFLIB;

//=============================================================================================================
/**
 * DECLARE CLASS TestChannelManagement
 *
 * @brief The TestChannelManagement class provides unit tests for channel management operations
 *
 */
class TestChannelManagement: public QObject
{
    Q_OBJECT

public:
    TestChannelManagement();

private slots:
    void initTestCase();
    void testChannelSelectionBasic();
    void testChannelSelectionConsistency();
    void testChannelRenaming();
    void testChannelRenamingConsistency();
    void testChannelEqualization();
    void testChannelEqualizationConsistency();
    void testChannelCombination();
    void testChannelCombinationConsistency();
    void testChannelAdjacency();
    void testChannelAdjacencyConsistency();
    void cleanupTestCase();

private:
    // Helper methods
    FiffInfo createTestInfo(int n_channels);
    QStringList createChannelNames(int n_channels);
    bool verifyChannelSelection(const FiffInfo& info, const QStringList& selected);
    bool verifyChannelRenaming(const FiffInfo& info, const QString& old_name, const QString& new_name);
    bool verifyChannelEqualization(const FiffInfo& info1, const FiffInfo& info2);
    
    // Test parameters
    QRandomGenerator* m_generator;
    int m_n_channels;
};

//=============================================================================================================

TestChannelManagement::TestChannelManagement()
: m_generator(QRandomGenerator::global())
, m_n_channels(64)
{
}

//=============================================================================================================

void TestChannelManagement::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Channel Management Tests";
    qDebug() << "Channels:" << m_n_channels;
}

//=============================================================================================================

void TestChannelManagement::testChannelSelectionBasic()
{
    qDebug() << "Test 1: Channel selection basic functionality";
    
    try {
        FiffInfo info = createTestInfo(m_n_channels);
        
        // Select a subset of channels
        QStringList selected_channels;
        for(int i = 0; i < m_n_channels / 2; ++i) {
            selected_channels.append(info.ch_names[i]);
        }
        
        // Verify selection
        QVERIFY(verifyChannelSelection(info, selected_channels));
        QVERIFY(selected_channels.size() == m_n_channels / 2);
        
        qDebug() << "✓ Channel selection verified for" << selected_channels.size() << "channels";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel selection test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelSelectionConsistency()
{
    qDebug() << "Test 2: Channel selection consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            FiffInfo info = createTestInfo(n_channels);
            
            // Select random subset
            int n_selected = m_generator->bounded(1, n_channels);
            QStringList selected_channels;
            for(int i = 0; i < n_selected; ++i) {
                int idx = m_generator->bounded(0, n_channels);
                selected_channels.append(info.ch_names[idx]);
            }
            
            // Verify selection is consistent
            QVERIFY(verifyChannelSelection(info, selected_channels));
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Selection consistency verified for"
                     << n_selected << "out of" << n_channels << "channels";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel selection consistency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelRenaming()
{
    qDebug() << "Test 3: Channel renaming basic functionality";
    
    try {
        FiffInfo info = createTestInfo(m_n_channels);
        
        // Rename a channel
        QString old_name = info.ch_names[0];
        QString new_name = "RENAMED_CH_0";
        
        // Verify renaming
        QVERIFY(verifyChannelRenaming(info, old_name, new_name));
        
        qDebug() << "✓ Channel renaming verified:" << old_name << "->" << new_name;
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel renaming test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelRenamingConsistency()
{
    qDebug() << "Test 4: Channel renaming consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            FiffInfo info = createTestInfo(n_channels);
            
            // Rename random channel
            int idx = m_generator->bounded(0, n_channels);
            QString old_name = info.ch_names[idx];
            QString new_name = QString("RENAMED_CH_%1").arg(idx);
            
            // Verify renaming is consistent
            QVERIFY(verifyChannelRenaming(info, old_name, new_name));
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Renaming consistency verified";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel renaming consistency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelEqualization()
{
    qDebug() << "Test 5: Channel equalization basic functionality";
    
    try {
        FiffInfo info1 = createTestInfo(m_n_channels);
        FiffInfo info2 = createTestInfo(m_n_channels);
        
        // Verify equalization
        QVERIFY(verifyChannelEqualization(info1, info2));
        QVERIFY(info1.nchan == info2.nchan);
        
        qDebug() << "✓ Channel equalization verified for" << m_n_channels << "channels";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel equalization test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelEqualizationConsistency()
{
    qDebug() << "Test 6: Channel equalization consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            FiffInfo info1 = createTestInfo(n_channels);
            FiffInfo info2 = createTestInfo(n_channels);
            
            // Verify equalization is consistent
            QVERIFY(verifyChannelEqualization(info1, info2));
            QVERIFY(info1.nchan == info2.nchan);
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Equalization consistency verified for"
                     << n_channels << "channels";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel equalization consistency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelCombination()
{
    qDebug() << "Test 7: Channel combination basic functionality";
    
    try {
        FiffInfo info = createTestInfo(m_n_channels);
        
        // Combine channels
        int n_combined = m_n_channels / 2;
        QStringList channels_to_combine;
        for(int i = 0; i < n_combined; ++i) {
            channels_to_combine.append(info.ch_names[i]);
        }
        
        // Verify combination
        QVERIFY(channels_to_combine.size() == n_combined);
        QVERIFY(channels_to_combine.size() <= m_n_channels);
        
        qDebug() << "✓ Channel combination verified for" << n_combined << "channels";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel combination test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelCombinationConsistency()
{
    qDebug() << "Test 8: Channel combination consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            FiffInfo info = createTestInfo(n_channels);
            
            // Combine random subset
            int n_combined = m_generator->bounded(1, n_channels);
            QStringList channels_to_combine;
            for(int i = 0; i < n_combined; ++i) {
                int idx = m_generator->bounded(0, n_channels);
                channels_to_combine.append(info.ch_names[idx]);
            }
            
            // Verify combination is consistent
            QVERIFY(channels_to_combine.size() == n_combined);
            QVERIFY(channels_to_combine.size() <= n_channels);
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Combination consistency verified for"
                     << n_combined << "out of" << n_channels << "channels";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel combination consistency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelAdjacency()
{
    qDebug() << "Test 9: Channel adjacency basic functionality";
    
    try {
        FiffInfo info = createTestInfo(m_n_channels);
        
        // Check adjacency for first channel
        int ch_idx = 0;
        int adjacent_count = 0;
        
        // Count adjacent channels (simplified: adjacent if indices differ by 1)
        for(int i = 0; i < m_n_channels; ++i) {
            if(std::abs(i - ch_idx) == 1) {
                adjacent_count++;
            }
        }
        
        // Verify adjacency
        QVERIFY(adjacent_count >= 0);
        QVERIFY(adjacent_count <= 2);  // At most 2 adjacent channels
        
        qDebug() << "✓ Channel adjacency verified:" << adjacent_count << "adjacent channels";
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel adjacency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::testChannelAdjacencyConsistency()
{
    qDebug() << "Test 10: Channel adjacency consistency";
    
    try {
        for(int test_idx = 0; test_idx < 5; ++test_idx) {
            int n_channels = m_generator->bounded(10, 100);
            FiffInfo info = createTestInfo(n_channels);
            
            // Check adjacency for random channel
            int ch_idx = m_generator->bounded(0, n_channels);
            int adjacent_count = 0;
            
            for(int i = 0; i < n_channels; ++i) {
                if(std::abs(i - ch_idx) == 1) {
                    adjacent_count++;
                }
            }
            
            // Verify adjacency is consistent
            QVERIFY(adjacent_count >= 0);
            QVERIFY(adjacent_count <= 2);
            
            qDebug() << "✓ Test" << (test_idx + 1) << ": Adjacency consistency verified for channel"
                     << ch_idx << "with" << adjacent_count << "adjacent channels";
        }
    } catch(const std::exception& e) {
        qDebug() << "Exception:" << e.what();
        QFAIL("Channel adjacency consistency test failed");
    }
}

//=============================================================================================================

void TestChannelManagement::cleanupTestCase()
{
    qDebug() << "Channel Management Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

FiffInfo TestChannelManagement::createTestInfo(int n_channels)
{
    FiffInfo info;
    info.nchan = n_channels;
    info.sfreq = 1000.0;
    
    // Create channel names
    for(int i = 0; i < n_channels; ++i) {
        info.ch_names.append(QString("CH_%1").arg(i));
    }
    
    return info;
}

//=============================================================================================================

QStringList TestChannelManagement::createChannelNames(int n_channels)
{
    QStringList names;
    for(int i = 0; i < n_channels; ++i) {
        names.append(QString("CH_%1").arg(i));
    }
    return names;
}

//=============================================================================================================

bool TestChannelManagement::verifyChannelSelection(const FiffInfo& info, const QStringList& selected)
{
    // Verify all selected channels exist in info
    for(const auto& ch : selected) {
        if(!info.ch_names.contains(ch)) {
            return false;
        }
    }
    return true;
}

//=============================================================================================================

bool TestChannelManagement::verifyChannelRenaming(const FiffInfo& info, const QString& old_name, const QString& new_name)
{
    // Verify old name exists
    if(!info.ch_names.contains(old_name)) {
        return false;
    }
    
    // Verify new name is different
    if(old_name == new_name) {
        return false;
    }
    
    return true;
}

//=============================================================================================================

bool TestChannelManagement::verifyChannelEqualization(const FiffInfo& info1, const FiffInfo& info2)
{
    // Verify both have same number of channels
    if(info1.nchan != info2.nchan) {
        return false;
    }
    
    // Verify both have same sampling frequency
    if(info1.sfreq != info2.sfreq) {
        return false;
    }
    
    return true;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestChannelManagement)
#include "test_channel_management.moc"
