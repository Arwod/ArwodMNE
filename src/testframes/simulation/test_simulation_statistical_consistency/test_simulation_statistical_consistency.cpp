//=============================================================================================================
/**
 * @file     test_simulation_statistical_consistency.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for simulated data statistical consistency (Property 13)
 *           Feature: mne-python-to-cpp-migration, Property 13: 仿真数据统计一致性
 *           Validates: Requirements 8.1
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

//=============================================================================================================
/**
 * DECLARE CLASS TestSimulationStatisticalConsistency
 *
 * @brief The TestSimulationStatisticalConsistency class provides property-based tests for simulated data
 *        statistical consistency
 *
 */
class TestSimulationStatisticalConsistency: public QObject
{
    Q_OBJECT

public:
    TestSimulationStatisticalConsistency();

private slots:
    void initTestCase();
    void testSimulatedEvokedMeanConsistency();
    void testSimulatedEvokedMeanConsistencyProperty();
    void testSimulatedEvokedVarianceConsistency();
    void testSimulatedEvokedVarianceConsistencyProperty();
    void testSimulatedRawNoiseStatistics();
    void testSimulatedRawNoiseStatisticsProperty();
    void testSimulatedDataFrequencyContent();
    void testSimulatedDataFrequencyContentProperty();
    void cleanupTestCase();

private:
    // Simulation helper methods
    MatrixXd simulateEvoked(int n_channels, int n_times, double sfreq, 
                           const VectorXd& signal_mean, double signal_std);
    MatrixXd simulateRaw(int n_channels, int n_times, double sfreq, 
                        double noise_std, const VectorXd& signal_freqs);
    
    // Statistical analysis methods
    double computeMean(const MatrixXd& data);
    double computeStd(const MatrixXd& data);
    double computeVariance(const MatrixXd& data);
    VectorXd computeChannelMeans(const MatrixXd& data);
    VectorXd computeChannelStds(const MatrixXd& data);
    VectorXd computeFFT(const VectorXd& signal);
    double computeFrequencyPower(const VectorXd& fft_result, double target_freq, 
                                double freq_resolution);
    
    // Test parameters
    QRandomGenerator* m_generator;
    double m_sfreq;
    int m_n_channels;
    int m_n_times;
    double m_tolerance;
};

//=============================================================================================================

TestSimulationStatisticalConsistency::TestSimulationStatisticalConsistency()
: m_generator(QRandomGenerator::global())
, m_sfreq(1000.0)
, m_n_channels(4)
, m_n_times(2000)
, m_tolerance(0.15)  // 15% tolerance for statistical properties
{
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Simulated Data Statistical Consistency Property Tests";
    qDebug() << "Sampling frequency:" << m_sfreq;
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Time points:" << m_n_times;
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedEvokedMeanConsistency()
{
    qDebug() << "Testing simulated evoked data mean consistency...";
    
    // Define target mean and std
    double target_mean = 2.5;
    double target_std = 1.0;
    VectorXd signal_mean = VectorXd::Constant(m_n_channels, target_mean);
    
    // Simulate evoked data
    MatrixXd simulated = simulateEvoked(m_n_channels, m_n_times, m_sfreq, 
                                       signal_mean, target_std);
    
    // Compute actual mean
    double actual_mean = computeMean(simulated);
    
    qDebug() << "Target mean:" << target_mean;
    qDebug() << "Actual mean:" << actual_mean;
    qDebug() << "Difference:" << std::abs(actual_mean - target_mean);
    
    // Check mean consistency (within tolerance)
    double mean_error = std::abs(actual_mean - target_mean) / std::abs(target_mean);
    QVERIFY(mean_error < m_tolerance);
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedEvokedMeanConsistencyProperty()
{
    qDebug() << "Running simulated evoked mean consistency property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 13: 仿真数据统计一致性
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(1, 8);
        int n_times = m_generator->bounded(1000, 3000);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random target mean and std
        double target_mean = (m_generator->generateDouble() - 0.5) * 10.0;  // -5 to 5
        double target_std = m_generator->generateDouble() * 2.0 + 0.5;      // 0.5 to 2.5
        
        VectorXd signal_mean = VectorXd::Constant(n_channels, target_mean);
        
        try {
            // Simulate evoked data
            MatrixXd simulated = simulateEvoked(n_channels, n_times, sfreq, 
                                               signal_mean, target_std);
            
            // Compute actual mean
            double actual_mean = computeMean(simulated);
            
            // Check mean consistency
            double mean_error = std::abs(actual_mean - target_mean) / 
                               (std::abs(target_mean) + 1e-6);
            
            if(mean_error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed simulations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 85% of tests should pass
    QVERIFY(successful_tests >= 85);
    
    qDebug() << "Simulated evoked mean consistency property test completed successfully";
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedEvokedVarianceConsistency()
{
    qDebug() << "Testing simulated evoked data variance consistency...";
    
    // Define target mean and std
    double target_mean = 1.0;
    double target_std = 2.0;
    VectorXd signal_mean = VectorXd::Constant(m_n_channels, target_mean);
    
    // Simulate evoked data
    MatrixXd simulated = simulateEvoked(m_n_channels, m_n_times, m_sfreq, 
                                       signal_mean, target_std);
    
    // Compute actual std
    double actual_std = computeStd(simulated);
    
    qDebug() << "Target std:" << target_std;
    qDebug() << "Actual std:" << actual_std;
    qDebug() << "Difference:" << std::abs(actual_std - target_std);
    
    // Check std consistency (within tolerance)
    double std_error = std::abs(actual_std - target_std) / target_std;
    QVERIFY(std_error < m_tolerance);
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedEvokedVarianceConsistencyProperty()
{
    qDebug() << "Running simulated evoked variance consistency property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 13: 仿真数据统计一致性
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(1, 8);
        int n_times = m_generator->bounded(1000, 3000);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random target mean and std
        double target_mean = (m_generator->generateDouble() - 0.5) * 10.0;
        double target_std = m_generator->generateDouble() * 2.0 + 0.5;
        
        VectorXd signal_mean = VectorXd::Constant(n_channels, target_mean);
        
        try {
            // Simulate evoked data
            MatrixXd simulated = simulateEvoked(n_channels, n_times, sfreq, 
                                               signal_mean, target_std);
            
            // Compute actual std
            double actual_std = computeStd(simulated);
            
            // Check std consistency
            double std_error = std::abs(actual_std - target_std) / target_std;
            
            if(std_error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed simulations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 85% of tests should pass
    QVERIFY(successful_tests >= 85);
    
    qDebug() << "Simulated evoked variance consistency property test completed successfully";
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedRawNoiseStatistics()
{
    qDebug() << "Testing simulated raw data noise statistics...";
    
    // Define noise parameters
    double noise_std = 1.5;
    VectorXd signal_freqs = VectorXd::Zero(0);  // No signal, just noise
    
    // Simulate raw data
    MatrixXd simulated = simulateRaw(m_n_channels, m_n_times, m_sfreq, 
                                    noise_std, signal_freqs);
    
    // Compute actual std
    double actual_std = computeStd(simulated);
    
    qDebug() << "Target noise std:" << noise_std;
    qDebug() << "Actual std:" << actual_std;
    qDebug() << "Difference:" << std::abs(actual_std - noise_std);
    
    // Check noise std consistency (within tolerance)
    double std_error = std::abs(actual_std - noise_std) / noise_std;
    QVERIFY(std_error < m_tolerance);
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedRawNoiseStatisticsProperty()
{
    qDebug() << "Running simulated raw noise statistics property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 13: 仿真数据统计一致性
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(1, 8);
        int n_times = m_generator->bounded(1000, 3000);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random noise std
        double noise_std = m_generator->generateDouble() * 2.0 + 0.5;  // 0.5 to 2.5
        
        try {
            // Simulate raw data with noise only
            VectorXd signal_freqs = VectorXd::Zero(0);
            MatrixXd simulated = simulateRaw(n_channels, n_times, sfreq, 
                                            noise_std, signal_freqs);
            
            // Compute actual std
            double actual_std = computeStd(simulated);
            
            // Check noise std consistency
            double std_error = std::abs(actual_std - noise_std) / noise_std;
            
            if(std_error < m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed simulations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 85% of tests should pass
    QVERIFY(successful_tests >= 85);
    
    qDebug() << "Simulated raw noise statistics property test completed successfully";
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedDataFrequencyContent()
{
    qDebug() << "Testing simulated data frequency content...";
    
    // Define signal frequency
    double signal_freq = 20.0;  // 20 Hz
    double noise_std = 0.5;
    VectorXd signal_freqs(1);
    signal_freqs(0) = signal_freq;
    
    // Simulate raw data with signal
    MatrixXd simulated = simulateRaw(m_n_channels, m_n_times, m_sfreq, 
                                    noise_std, signal_freqs);
    
    // Compute FFT of first channel
    VectorXd fft_result = computeFFT(simulated.row(0));
    
    // Compute power at target frequency
    double freq_resolution = m_sfreq / m_n_times;
    double power_at_freq = computeFrequencyPower(fft_result, signal_freq, freq_resolution);
    
    qDebug() << "Target frequency:" << signal_freq << "Hz";
    qDebug() << "Power at target frequency:" << power_at_freq;
    
    // Power at target frequency should be significant
    QVERIFY(power_at_freq > 0.1);
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::testSimulatedDataFrequencyContentProperty()
{
    qDebug() << "Running simulated data frequency content property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 13: 仿真数据统计一致性
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_channels = m_generator->bounded(1, 4);
        int n_times = m_generator->bounded(2000, 4000);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random signal frequency (avoid DC and Nyquist)
        double signal_freq = m_generator->bounded(5, static_cast<int>(sfreq / 4));
        double noise_std = m_generator->generateDouble() * 0.5 + 0.1;
        
        try {
            // Simulate raw data with signal
            VectorXd signal_freqs(1);
            signal_freqs(0) = signal_freq;
            
            MatrixXd simulated = simulateRaw(n_channels, n_times, sfreq, 
                                            noise_std, signal_freqs);
            
            // Compute FFT
            VectorXd fft_result = computeFFT(simulated.row(0));
            
            // Compute power at target frequency
            double freq_resolution = sfreq / n_times;
            double power_at_freq = computeFrequencyPower(fft_result, signal_freq, freq_resolution);
            
            // Power should be significant
            if(power_at_freq > 0.05) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed simulations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 80% of tests should pass
    QVERIFY(successful_tests >= 80);
    
    qDebug() << "Simulated data frequency content property test completed successfully";
}

//=============================================================================================================

void TestSimulationStatisticalConsistency::cleanupTestCase()
{
    qDebug() << "Simulated Data Statistical Consistency Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestSimulationStatisticalConsistency::simulateEvoked(int n_channels, int n_times, 
                                                             double sfreq,
                                                             const VectorXd& signal_mean, 
                                                             double signal_std)
{
    MatrixXd data(n_channels, n_times);
    
    // Generate time vector
    VectorXd time = VectorXd::LinSpaced(n_times, 0.0, (n_times - 1) / sfreq);
    
    for(int ch = 0; ch < n_channels; ++ch) {
        for(int t = 0; t < n_times; ++t) {
            // Generate Gaussian noise with specified mean and std
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            
            data(ch, t) = signal_mean(ch) + signal_std * z;
        }
    }
    
    return data;
}

//=============================================================================================================

MatrixXd TestSimulationStatisticalConsistency::simulateRaw(int n_channels, int n_times, 
                                                          double sfreq,
                                                          double noise_std, 
                                                          const VectorXd& signal_freqs)
{
    MatrixXd data(n_channels, n_times);
    
    // Generate time vector
    VectorXd time = VectorXd::LinSpaced(n_times, 0.0, (n_times - 1) / sfreq);
    
    for(int ch = 0; ch < n_channels; ++ch) {
        for(int t = 0; t < n_times; ++t) {
            double value = 0.0;
            
            // Add signal components
            for(int f = 0; f < signal_freqs.size(); ++f) {
                double freq = signal_freqs(f);
                double amplitude = m_generator->generateDouble() * 0.5 + 0.5;
                double phase = m_generator->generateDouble() * 2.0 * M_PI;
                
                value += amplitude * std::sin(2.0 * M_PI * freq * time(t) + phase);
            }
            
            // Add Gaussian noise
            double u1 = m_generator->generateDouble();
            double u2 = m_generator->generateDouble();
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            
            value += noise_std * z;
            
            data(ch, t) = value;
        }
    }
    
    return data;
}

//=============================================================================================================

double TestSimulationStatisticalConsistency::computeMean(const MatrixXd& data)
{
    return data.mean();
}

//=============================================================================================================

double TestSimulationStatisticalConsistency::computeStd(const MatrixXd& data)
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

double TestSimulationStatisticalConsistency::computeVariance(const MatrixXd& data)
{
    double std = computeStd(data);
    return std * std;
}

//=============================================================================================================

VectorXd TestSimulationStatisticalConsistency::computeChannelMeans(const MatrixXd& data)
{
    VectorXd means(data.rows());
    
    for(int ch = 0; ch < data.rows(); ++ch) {
        means(ch) = data.row(ch).mean();
    }
    
    return means;
}

//=============================================================================================================

VectorXd TestSimulationStatisticalConsistency::computeChannelStds(const MatrixXd& data)
{
    VectorXd stds(data.rows());
    
    for(int ch = 0; ch < data.rows(); ++ch) {
        VectorXd channel = data.row(ch);
        double mean = channel.mean();
        double sum_sq_diff = 0.0;
        
        for(int t = 0; t < channel.size(); ++t) {
            double diff = channel(t) - mean;
            sum_sq_diff += diff * diff;
        }
        
        stds(ch) = std::sqrt(sum_sq_diff / (channel.size() - 1));
    }
    
    return stds;
}

//=============================================================================================================

VectorXd TestSimulationStatisticalConsistency::computeFFT(const VectorXd& signal)
{
    // Simple DFT implementation (not optimized, but sufficient for testing)
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

double TestSimulationStatisticalConsistency::computeFrequencyPower(const VectorXd& fft_result, 
                                                                  double target_freq,
                                                                  double freq_resolution)
{
    // Find the bin closest to target frequency
    int target_bin = static_cast<int>(target_freq / freq_resolution + 0.5);
    
    if(target_bin < 0 || target_bin >= fft_result.size()) {
        return 0.0;
    }
    
    // Return normalized power
    return fft_result(target_bin) / fft_result.maxCoeff();
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestSimulationStatisticalConsistency)
#include "test_simulation_statistical_consistency.moc"
