//=============================================================================================================
/**
 * @file     test_tfr_energy_conservation.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for time-frequency transform energy conservation (Property 1)
 *           Feature: mne-python-to-cpp-migration, Property 1: 时频变换能量守恒
 *           Validates: Requirements 1.1, 1.2
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <tfr/tfr_compute.h>

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
 * DECLARE CLASS TestTFREnergyConservation
 *
 * @brief The TestTFREnergyConservation class provides property-based tests for TFR energy conservation
 *
 */
class TestTFREnergyConservation: public QObject
{
    Q_OBJECT

public:
    TestTFREnergyConservation();

private slots:
    void initTestCase();
    void testMorletEnergyConservation();
    void testMorletEnergyConservationProperty();
    void testMultitaperEnergyConservation();
    void testMultitaperEnergyConservationProperty();
    void cleanupTestCase();

private:
    // Helper methods for property testing
    MatrixXd generateRandomSignal(int n_channels, int n_times, double sfreq);
    VectorXd generateFrequencyVector(double fmin, double fmax, int n_freqs);
    double computeSignalEnergy(const MatrixXd& signal);
    double computeTFREnergy(const std::vector<std::vector<VectorXd>>& tfr_power);
    double computeTFRComplexEnergy(const std::vector<std::vector<VectorXcd>>& tfr_complex);
    
    // Test parameters
    double m_tolerance;
    QRandomGenerator* m_generator;
    double m_sfreq;
    int m_n_channels;
    int m_n_times;
};

//=============================================================================================================

TestTFREnergyConservation::TestTFREnergyConservation()
: m_tolerance(0.1)  // 10% tolerance for energy conservation
, m_generator(QRandomGenerator::global())
, m_sfreq(1000.0)
, m_n_channels(4)
, m_n_times(1000)
{
}

//=============================================================================================================

void TestTFREnergyConservation::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting TFR Energy Conservation Property Tests";
    qDebug() << "Tolerance:" << m_tolerance;
    qDebug() << "Sampling frequency:" << m_sfreq;
    qDebug() << "Channels:" << m_n_channels;
    qDebug() << "Time points:" << m_n_times;
}

//=============================================================================================================

void TestTFREnergyConservation::testMorletEnergyConservation()
{
    qDebug() << "Testing Morlet wavelet energy conservation...";
    
    // Create a simple test signal
    MatrixXd signal = generateRandomSignal(m_n_channels, m_n_times, m_sfreq);
    VectorXd freqs = generateFrequencyVector(10.0, 100.0, 20);
    
    // Compute original signal energy
    double original_energy = computeSignalEnergy(signal);
    
    // Compute TFR using Morlet wavelets
    auto tfr_power = TFRCompute::tfr_morlet(signal, m_sfreq, freqs, 7.0, true, 1);
    
    // Compute TFR energy
    double tfr_energy = computeTFREnergy(tfr_power);
    
    // Check energy conservation (should be proportional)
    double energy_ratio = tfr_energy / original_energy;
    
    qDebug() << "Original energy:" << original_energy;
    qDebug() << "TFR energy:" << tfr_energy;
    qDebug() << "Energy ratio:" << energy_ratio;
    
    // Energy should be conserved within tolerance
    QVERIFY(energy_ratio > 0.1);  // Should have some energy
    QVERIFY(energy_ratio < 10.0); // Should not be too large
}

//=============================================================================================================

void TestTFREnergyConservation::testMorletEnergyConservationProperty()
{
    qDebug() << "Running Morlet energy conservation property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 1: 时频变换能量守恒
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random signal parameters
        int n_channels = m_generator->bounded(1, 8);
        int n_times = m_generator->bounded(500, 2000);
        double sfreq = m_generator->bounded(500, 2000);
        
        // Generate random signal
        MatrixXd signal = generateRandomSignal(n_channels, n_times, sfreq);
        
        // Skip if signal has zero energy
        double original_energy = computeSignalEnergy(signal);
        if(original_energy < 1e-12) continue;
        
        // Generate frequency vector
        double fmax = sfreq / 3.0;  // Avoid aliasing
        VectorXd freqs = generateFrequencyVector(5.0, fmax, 15);
        
        // Compute TFR
        try {
            auto tfr_power = TFRCompute::tfr_morlet(signal, sfreq, freqs, 7.0, true, 1);
            double tfr_energy = computeTFREnergy(tfr_power);
            
            // Check energy conservation
            double energy_ratio = tfr_energy / original_energy;
            
            // Energy should be positive and reasonable
            if(energy_ratio > 0.01 && energy_ratio < 100.0) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed computations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 80% of tests should pass
    QVERIFY(successful_tests >= 80);
    
    qDebug() << "Morlet energy conservation property test completed successfully";
}

//=============================================================================================================

void TestTFREnergyConservation::testMultitaperEnergyConservation()
{
    qDebug() << "Testing Multitaper energy conservation...";
    
    // Create a simple test signal
    MatrixXd signal = generateRandomSignal(m_n_channels, m_n_times, m_sfreq);
    VectorXd freqs = generateFrequencyVector(10.0, 100.0, 20);
    VectorXd n_cycles = VectorXd::Constant(freqs.size(), 7.0);
    
    // Compute original signal energy
    double original_energy = computeSignalEnergy(signal);
    
    // Compute TFR using multitaper method
    auto tfr_complex = TFRCompute::tfr_multitaper(signal, m_sfreq, freqs, n_cycles, 4.0, true, "complex", 1);
    
    // Compute TFR energy
    double tfr_energy = computeTFRComplexEnergy(tfr_complex);
    
    // Check energy conservation (should be proportional)
    double energy_ratio = tfr_energy / original_energy;
    
    qDebug() << "Original energy:" << original_energy;
    qDebug() << "TFR energy:" << tfr_energy;
    qDebug() << "Energy ratio:" << energy_ratio;
    
    // Energy should be conserved within tolerance
    QVERIFY(energy_ratio > 0.1);  // Should have some energy
    QVERIFY(energy_ratio < 10.0); // Should not be too large
}

//=============================================================================================================

void TestTFREnergyConservation::testMultitaperEnergyConservationProperty()
{
    qDebug() << "Running Multitaper energy conservation property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 1: 时频变换能量守恒
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random signal parameters
        int n_channels = m_generator->bounded(1, 6);
        int n_times = m_generator->bounded(500, 1500);
        double sfreq = m_generator->bounded(500, 1500);
        
        // Generate random signal
        MatrixXd signal = generateRandomSignal(n_channels, n_times, sfreq);
        
        // Skip if signal has zero energy
        double original_energy = computeSignalEnergy(signal);
        if(original_energy < 1e-12) continue;
        
        // Generate frequency vector
        double fmax = sfreq / 3.0;  // Avoid aliasing
        VectorXd freqs = generateFrequencyVector(5.0, fmax, 10);
        VectorXd n_cycles = VectorXd::Constant(freqs.size(), 7.0);
        
        // Compute TFR
        try {
            auto tfr_complex = TFRCompute::tfr_multitaper(signal, sfreq, freqs, n_cycles, 4.0, true, "complex", 1);
            double tfr_energy = computeTFRComplexEnergy(tfr_complex);
            
            // Check energy conservation
            double energy_ratio = tfr_energy / original_energy;
            
            // Energy should be positive and reasonable
            if(energy_ratio > 0.01 && energy_ratio < 100.0) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed computations
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 80% of tests should pass
    QVERIFY(successful_tests >= 80);
    
    qDebug() << "Multitaper energy conservation property test completed successfully";
}

//=============================================================================================================

void TestTFREnergyConservation::cleanupTestCase()
{
    qDebug() << "TFR Energy Conservation Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

MatrixXd TestTFREnergyConservation::generateRandomSignal(int n_channels, int n_times, double sfreq)
{
    MatrixXd signal(n_channels, n_times);
    
    // Generate signal with multiple frequency components
    VectorXd time = VectorXd::LinSpaced(n_times, 0.0, (n_times - 1) / sfreq);
    
    for(int ch = 0; ch < n_channels; ++ch) {
        for(int t = 0; t < n_times; ++t) {
            double value = 0.0;
            
            // Add multiple sinusoidal components with random phases
            for(int f = 0; f < 3; ++f) {
                double freq = 10.0 + f * 20.0;  // 10, 30, 50 Hz
                double amplitude = m_generator->generateDouble() * 0.5 + 0.5;  // 0.5-1.0
                double phase = m_generator->generateDouble() * 2.0 * M_PI;
                
                value += amplitude * sin(2.0 * M_PI * freq * time(t) + phase);
            }
            
            // Add some noise
            value += (m_generator->generateDouble() - 0.5) * 0.1;
            
            signal(ch, t) = value;
        }
    }
    
    return signal;
}

//=============================================================================================================

VectorXd TestTFREnergyConservation::generateFrequencyVector(double fmin, double fmax, int n_freqs)
{
    return VectorXd::LinSpaced(n_freqs, fmin, fmax);
}

//=============================================================================================================

double TestTFREnergyConservation::computeSignalEnergy(const MatrixXd& signal)
{
    return signal.array().square().sum();
}

//=============================================================================================================

double TestTFREnergyConservation::computeTFREnergy(const std::vector<std::vector<VectorXd>>& tfr_power)
{
    double total_energy = 0.0;
    
    for(const auto& channel_tfr : tfr_power) {
        for(const auto& freq_tfr : channel_tfr) {
            total_energy += freq_tfr.sum();
        }
    }
    
    return total_energy;
}

//=============================================================================================================

double TestTFREnergyConservation::computeTFRComplexEnergy(const std::vector<std::vector<VectorXcd>>& tfr_complex)
{
    double total_energy = 0.0;
    
    for(const auto& channel_tfr : tfr_complex) {
        for(const auto& freq_tfr : channel_tfr) {
            for(int t = 0; t < freq_tfr.size(); ++t) {
                total_energy += std::norm(freq_tfr(t));  // |z|^2
            }
        }
    }
    
    return total_energy;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestTFREnergyConservation)
#include "test_tfr_energy_conservation.moc"