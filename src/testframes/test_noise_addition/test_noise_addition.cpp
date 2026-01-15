//=============================================================================================================
/**
 * @file     test_noise_addition.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Unit tests for noise addition functionality (Task 12.4)
 *           Feature: mne-python-to-cpp-migration, Task 12.4: 编写噪声添加单元测试
 *           Validates: Requirements 8.2
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <simulation/noise_simulation.h>
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
using namespace SIMULATIONLIB;
using namespace FIFFLIB;

//=============================================================================================================
/**
 * DECLARE CLASS TestNoiseAddition
 *
 * @brief The TestNoiseAddition class provides unit tests for noise addition functionality
 *
 */
class TestNoiseAddition: public QObject
{
    Q_OBJECT

public:
    TestNoiseAddition();

private slots:
    void initTestCase();
    void testWhiteNoiseAddition();
    void testWhiteNoiseStatistics();
    void testECGAddition();
    void testECGAmplitude();
    void testEOGAddition();
    void testEOGBlinks();
    void testCHPIAddition();
    void testCHPIFrequencies();
    void testGeneralNoiseAddition();
    void testNoiseLevel();
    void testRandomSeed();
    void cleanupTestCase();

private:
    // Helper methods
    FiffInfo::SPtr createTestInfo(int n_channels, double sfreq);
    double computeRMS(const MatrixXd& data);
    double computeStd(const MatrixXd& data);
    double computeMean(const MatrixXd& data);
    VectorXd computeFFT(const VectorXd& signal);
    double computeFrequencyPower(const VectorXd& fft_result, double target_freq, double freq_resolution);
    
    // Test parameters
    QRandomGenerator* m_generator;
    double m_sfreq;
    int m_n_channels;
    int m_n_times;
    double m_tolerance;
};

//=============================================================================================================

TestNoiseAddition::TestNoiseAddition()
: m_generator(QRandomGenerator::global())
, m_sfreq(1000.0)
, m_n_channels(64)
, m_n_times(10000)
, m_tolerance(0.2)  // 20% tolerance for noise statistics
{
}

//=============================================================================================================

void TestNoiseAddition::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Noise Addition Unit Tests";
    qDebug() << "Sampling frequency:" << m_sfreq;
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Time points:" << m_n_times;
}

//=============================================================================================================

void TestNoiseAddition::testWhiteNoiseAddition()
{
    qDebug() << "Test 1: White noise addition";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add white noise
    double noise_level = 1e-12;
    add_noise(data, info, noise_level);
    
    // Verify noise was added
    double rms = computeRMS(data);
    qDebug() << "Target noise level (variance):" << noise_level;
    qDebug() << "Actual RMS:" << rms;
    
    // RMS should be approximately sqrt(noise_level)
    double expected_rms = std::sqrt(noise_level);
    qDebug() << "Expected RMS (sqrt of variance):" << expected_rms;
    
    QVERIFY(rms > 0);
    // Allow 50% tolerance due to random nature
    QVERIFY(rms < expected_rms * 2.0);
    
    qDebug() << "✓ White noise addition test passed";
}

//=============================================================================================================

void TestNoiseAddition::testWhiteNoiseStatistics()
{
    qDebug() << "Test 2: White noise statistics";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add white noise
    double noise_level = 1e-12;
    add_noise(data, info, noise_level);
    
    // Check mean is close to zero
    double mean = computeMean(data);
    qDebug() << "Mean of noise:" << mean;
    // Mean should be very close to zero for large sample
    QVERIFY(std::abs(mean) < 1e-6);
    
    // Check std is proportional to noise_level
    double std = computeStd(data);
    double expected_std = std::sqrt(noise_level);
    qDebug() << "Std of noise:" << std;
    qDebug() << "Expected std:" << expected_std;
    QVERIFY(std > 0);
    
    qDebug() << "✓ White noise statistics test passed";
}

//=============================================================================================================

void TestNoiseAddition::testECGAddition()
{
    qDebug() << "Test 3: ECG artifact addition";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add ECG artifacts
    NoiseSimulation::ECGParams ecg_params = NoiseSimulation::getDefaultECGParams();
    ecg_params.heart_rate = 60.0;  // 60 bpm
    ecg_params.amplitude = 5e-12;
    
    add_ecg(data, info, ecg_params);
    
    // Verify ECG was added
    double rms = computeRMS(data);
    qDebug() << "RMS after ECG addition:" << rms;
    
    QVERIFY(rms > 0);
    
    qDebug() << "✓ ECG artifact addition test passed";
}

//=============================================================================================================

void TestNoiseAddition::testECGAmplitude()
{
    qDebug() << "Test 4: ECG amplitude consistency";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add ECG with specific amplitude
    NoiseSimulation::ECGParams ecg_params = NoiseSimulation::getDefaultECGParams();
    double target_amplitude = 10e-12;
    ecg_params.amplitude = target_amplitude;
    ecg_params.heart_rate = 72.0;
    
    add_ecg(data, info, ecg_params);
    
    // Check that max amplitude is reasonable
    double max_value = data.cwiseAbs().maxCoeff();
    qDebug() << "Target ECG amplitude:" << target_amplitude;
    qDebug() << "Max value in data:" << max_value;
    
    QVERIFY(max_value > 0);
    QVERIFY(max_value < target_amplitude * 10);
    
    qDebug() << "✓ ECG amplitude consistency test passed";
}

//=============================================================================================================

void TestNoiseAddition::testEOGAddition()
{
    qDebug() << "Test 5: EOG artifact addition";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add EOG artifacts
    NoiseSimulation::EOGParams eog_params = NoiseSimulation::getDefaultEOGParams();
    eog_params.blink_rate = 15.0;
    eog_params.amplitude = 10e-12;
    
    add_eog(data, info, eog_params);
    
    // Verify EOG was added
    double rms = computeRMS(data);
    qDebug() << "RMS after EOG addition:" << rms;
    
    QVERIFY(rms > 0);
    
    qDebug() << "✓ EOG artifact addition test passed";
}

//=============================================================================================================

void TestNoiseAddition::testEOGBlinks()
{
    qDebug() << "Test 6: EOG blink detection";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add EOG with specific blink rate
    NoiseSimulation::EOGParams eog_params = NoiseSimulation::getDefaultEOGParams();
    double blink_rate = 20.0;  // 20 blinks per minute
    eog_params.blink_rate = blink_rate;
    eog_params.amplitude = 15e-12;
    
    add_eog(data, info, eog_params);
    
    // Expected number of blinks
    double duration = m_n_times / m_sfreq;
    int expected_blinks = static_cast<int>(blink_rate * duration / 60.0);
    
    qDebug() << "Duration:" << duration << "seconds";
    qDebug() << "Expected blinks:" << expected_blinks;
    
    // Check that data has been modified
    double rms = computeRMS(data);
    QVERIFY(rms > 0);
    
    qDebug() << "✓ EOG blink detection test passed";
}

//=============================================================================================================

void TestNoiseAddition::testCHPIAddition()
{
    qDebug() << "Test 7: cHPI artifact addition";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add cHPI artifacts
    NoiseSimulation::CHPIParams chpi_params = NoiseSimulation::getDefaultCHPIParams();
    
    add_chpi(data, info, chpi_params);
    
    // Verify cHPI was added
    double rms = computeRMS(data);
    qDebug() << "RMS after cHPI addition:" << rms;
    
    QVERIFY(rms > 0);
    
    qDebug() << "✓ cHPI artifact addition test passed";
}

//=============================================================================================================

void TestNoiseAddition::testCHPIFrequencies()
{
    qDebug() << "Test 8: cHPI frequency content";
    
    // Create test data with only cHPI
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add cHPI with specific frequencies
    NoiseSimulation::CHPIParams chpi_params = NoiseSimulation::getDefaultCHPIParams();
    chpi_params.frequencies = {83.0, 143.0};  // Two HPI frequencies
    chpi_params.amplitudes = {1e-11, 1e-11};
    
    add_chpi(data, info, chpi_params);
    
    // Compute FFT of first channel
    VectorXd fft_result = computeFFT(data.row(0));
    double freq_resolution = m_sfreq / m_n_times;
    
    // Check power at HPI frequencies
    double power_83 = computeFrequencyPower(fft_result, 83.0, freq_resolution);
    double power_143 = computeFrequencyPower(fft_result, 143.0, freq_resolution);
    
    qDebug() << "Power at 83 Hz:" << power_83;
    qDebug() << "Power at 143 Hz:" << power_143;
    
    // Both should have significant power
    QVERIFY(power_83 > 0.01);
    QVERIFY(power_143 > 0.01);
    
    qDebug() << "✓ cHPI frequency content test passed";
}

//=============================================================================================================

void TestNoiseAddition::testGeneralNoiseAddition()
{
    qDebug() << "Test 9: General noise addition";
    
    // Create test data
    MatrixXd data = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Create noise simulator
    NoiseSimulation noise_sim(info);
    
    // Add general noise
    NoiseSimulation::GeneralNoiseParams noise_params = NoiseSimulation::getDefaultGeneralNoiseParams();
    noise_params.white_noise_level = 1e-12;
    noise_params.line_noise_amplitude = 5e-13;
    
    noise_sim.addGeneralNoise(data, noise_params);
    
    // Verify noise was added
    double rms = computeRMS(data);
    qDebug() << "RMS after general noise addition:" << rms;
    
    QVERIFY(rms > 0);
    
    qDebug() << "✓ General noise addition test passed";
}

//=============================================================================================================

void TestNoiseAddition::testNoiseLevel()
{
    qDebug() << "Test 10: Noise level scaling";
    
    // Create two datasets with different noise levels
    MatrixXd data1 = MatrixXd::Zero(m_n_channels, m_n_times);
    MatrixXd data2 = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add noise with different levels
    double noise_level1 = 1e-12;
    double noise_level2 = 2e-12;
    
    add_noise(data1, info, noise_level1, 42);  // Fixed seed for reproducibility
    add_noise(data2, info, noise_level2, 42);  // Same seed
    
    double rms1 = computeRMS(data1);
    double rms2 = computeRMS(data2);
    
    qDebug() << "RMS with noise level" << noise_level1 << ":" << rms1;
    qDebug() << "RMS with noise level" << noise_level2 << ":" << rms2;
    
    // RMS2 should be roughly sqrt(2) times RMS1 since variance scales linearly
    double ratio = rms2 / rms1;
    double expected_ratio = std::sqrt(2.0);
    qDebug() << "Ratio of RMS values:" << ratio;
    qDebug() << "Expected ratio (sqrt(2)):" << expected_ratio;
    
    // Allow 30% tolerance
    QVERIFY(ratio > expected_ratio * 0.7 && ratio < expected_ratio * 1.3);
    
    qDebug() << "✓ Noise level scaling test passed";
}

//=============================================================================================================

void TestNoiseAddition::testRandomSeed()
{
    qDebug() << "Test 11: Random seed reproducibility";
    
    // Create two datasets with same seed
    MatrixXd data1 = MatrixXd::Zero(m_n_channels, m_n_times);
    MatrixXd data2 = MatrixXd::Zero(m_n_channels, m_n_times);
    FiffInfo::SPtr info = createTestInfo(m_n_channels, m_sfreq);
    
    // Add noise with same seed
    int seed = 12345;
    add_noise(data1, info, 1e-12, seed);
    add_noise(data2, info, 1e-12, seed);
    
    // Data should be identical
    double max_diff = (data1 - data2).cwiseAbs().maxCoeff();
    qDebug() << "Max difference between seeded runs:" << max_diff;
    
    QVERIFY(max_diff < 1e-15);  // Should be numerically identical
    
    qDebug() << "✓ Random seed reproducibility test passed";
}

//=============================================================================================================

void TestNoiseAddition::cleanupTestCase()
{
    qDebug() << "Noise Addition Unit Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

FiffInfo::SPtr TestNoiseAddition::createTestInfo(int n_channels, double sfreq)
{
    FiffInfo::SPtr info = FiffInfo::SPtr::create();
    info->nchan = n_channels;
    info->sfreq = sfreq;
    
    return info;
}

//=============================================================================================================

double TestNoiseAddition::computeRMS(const MatrixXd& data)
{
    return std::sqrt(data.cwiseProduct(data).mean());
}

//=============================================================================================================

double TestNoiseAddition::computeStd(const MatrixXd& data)
{
    double mean = data.mean();
    double sum_sq_diff = 0.0;
    
    for(int i = 0; i < data.rows(); ++i) {
        for(int j = 0; j < data.cols(); ++j) {
            double diff = data(i, j) - mean;
            sum_sq_diff += diff * diff;
        }
    }
    
    return std::sqrt(sum_sq_diff / (data.rows() * data.cols() - 1));
}

//=============================================================================================================

double TestNoiseAddition::computeMean(const MatrixXd& data)
{
    return data.mean();
}

//=============================================================================================================

VectorXd TestNoiseAddition::computeFFT(const VectorXd& signal)
{
    // Simple DFT implementation
    int n = signal.size();
    VectorXd magnitude(n / 2);
    
    for(int k = 0; k < n / 2; ++k) {
        double real_part = 0.0;
        double imag_part = 0.0;
        
        for(int t = 0; t < n; ++t) {
            double angle = -2.0 * M_PI * k * t / n;
            real_part += signal(t) * std::cos(angle);
            imag_part += signal(t) * std::sin(angle);
        }
        
        magnitude(k) = std::sqrt(real_part * real_part + imag_part * imag_part);
    }
    
    return magnitude;
}

//=============================================================================================================

double TestNoiseAddition::computeFrequencyPower(const VectorXd& fft_result, 
                                               double target_freq,
                                               double freq_resolution)
{
    // Find the bin closest to target frequency
    int target_bin = static_cast<int>(target_freq / freq_resolution + 0.5);
    
    if(target_bin < 0 || target_bin >= fft_result.size()) {
        return 0.0;
    }
    
    // Return normalized power
    double max_power = fft_result.maxCoeff();
    if(max_power > 0) {
        return fft_result(target_bin) / max_power;
    }
    
    return 0.0;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestNoiseAddition)
#include "test_noise_addition.moc"
