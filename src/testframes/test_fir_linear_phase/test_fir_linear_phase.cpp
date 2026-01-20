//=============================================================================================================
/**
 * @file     test_fir_linear_phase.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MNE-CPP AUTHORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief     Property-based test for FIR filter linear phase response
 *            Feature: mne-python-to-cpp-migration, Property 18: FIR滤波器线性相位
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <rtprocessing/helpers/firfilter.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <Eigen/Dense>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QtCore/QCoreApplication>
#include <QtTest>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace RTPROCESSINGLIB;
using namespace UTILSLIB;
using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestFIRLinearPhase
 *
 * @brief The TestFIRLinearPhase class provides property-based tests for FIR filter linear phase response
 *
 */
class TestFIRLinearPhase: public QObject
{
    Q_OBJECT

public:
    TestFIRLinearPhase();

private slots:
    void initTestCase();
    void testLinearPhaseProperty();
    void testLinearPhasePropertyLowpass();
    void testLinearPhasePropertyHighpass();
    void testLinearPhasePropertyBandpass();
    void testLinearPhasePropertyBandstop();
    void testLinearPhaseConsistency();
    void testGroupDelayConstancy();
    void testSymmetricCoefficients();
    void cleanupTestCase();

private:
    // Helper functions
    VectorXd generateRandomFrequencies(double fmin, double fmax, int nFreqs);
    FIRFilter::FilterDesign generateRandomFilterDesign();
    bool isLinearPhase(const VectorXd& frequencies, const VectorXd& phase, double tolerance = 0.1);
    double computePhaseLinearity(const VectorXd& frequencies, const VectorXd& phase);
    bool isSymmetric(const VectorXd& coefficients, double tolerance = 1e-10);
    
    // Test parameters
    static constexpr double TOLERANCE = 0.1;  // Phase linearity tolerance in radians
    static constexpr double COEFF_TOLERANCE = 1e-10;  // Coefficient symmetry tolerance
    static constexpr int NUM_PROPERTY_ITERATIONS = 100;  // Number of property test iterations
    
    std::mt19937 m_rng;
};

//=============================================================================================================

TestFIRLinearPhase::TestFIRLinearPhase()
: m_rng(std::random_device{}())
{
}

//=============================================================================================================

void TestFIRLinearPhase::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting FIR Linear Phase Property Tests";
    qDebug() << "Phase linearity tolerance:" << TOLERANCE << "radians";
    qDebug() << "Coefficient symmetry tolerance:" << COEFF_TOLERANCE;
    qDebug() << "Property test iterations:" << NUM_PROPERTY_ITERATIONS;
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhaseProperty()
{
    // Feature: mne-python-to-cpp-migration, Property 18: FIR滤波器线性相位
    // For any FIR filter with linear phase design, the phase response should be linear in the passband
    
    int passedIterations = 0;
    
    // Debug: Test with a simple known case first
    FIRFilter::FilterDesign debugDesign;
    debugDesign.type = FIRFilter::LOWPASS;
    debugDesign.linearPhase = true;
    debugDesign.samplingRate = 1000.0;
    debugDesign.order = 64;
    debugDesign.window = FIRFilter::HAMMING;
    debugDesign.cutoffFreqs.resize(1);
    debugDesign.cutoffFreqs[0] = 200.0;
    
    FIRFilter debugFilter;
    bool designSuccess = debugFilter.designFilter(debugDesign);
    
    if (designSuccess) {
        VectorXd frequencies(10);
        for (int i = 0; i < 10; ++i) {
            frequencies[i] = 10.0 + i * 15.0;  // 10, 25, 40, ..., 145 Hz
        }
        
        VectorXd magnitude, phase;
        debugFilter.frequencyResponse(frequencies, magnitude, phase);
        
        qDebug() << "Debug frequencies:";
        for (int i = 0; i < frequencies.size(); ++i) {
            qDebug() << "  " << i << ":" << frequencies[i];
        }
        qDebug() << "Debug phase:";
        for (int i = 0; i < phase.size(); ++i) {
            qDebug() << "  " << i << ":" << phase[i];
        }
        
        // Check if coefficients are symmetric
        VectorXd coeffs = debugFilter.getCoefficients();
        bool symmetric = isSymmetric(coeffs, COEFF_TOLERANCE);
        qDebug() << "Coefficients symmetric:" << symmetric;
        
        // Check linearity
        bool linear = isLinearPhase(frequencies, phase, TOLERANCE);
        qDebug() << "Linear phase result:" << linear;
        
        // Also check unwrapped phase manually
        VectorXd unwrappedPhase = phase;
        for (int i = 1; i < unwrappedPhase.size(); ++i) {
            double diff = unwrappedPhase[i] - unwrappedPhase[i-1];
            while (diff > M_PI) {
                unwrappedPhase[i] -= 2 * M_PI;
                diff = unwrappedPhase[i] - unwrappedPhase[i-1];
            }
            while (diff < -M_PI) {
                unwrappedPhase[i] += 2 * M_PI;
                diff = unwrappedPhase[i] - unwrappedPhase[i-1];
            }
        }
        qDebug() << "Unwrapped phase:";
        for (int i = 0; i < unwrappedPhase.size(); ++i) {
            qDebug() << "  " << i << ":" << unwrappedPhase[i];
        }
        double unwrappedLinearity = computePhaseLinearity(frequencies, unwrappedPhase);
        qDebug() << "Unwrapped phase linearity:" << unwrappedLinearity;
    }
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        // Generate random filter design with linear phase enabled
        FIRFilter::FilterDesign design = generateRandomFilterDesign();
        design.linearPhase = true;  // Ensure linear phase is enabled
        
        // Create and design the filter
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;  // Skip invalid designs
        }
        
        double nyquist = design.samplingRate / 2.0;
        
        // Generate frequency points for evaluation (only in passband)
        VectorXd frequencies;
        switch (design.type) {
            case FIRFilter::LOWPASS:
                frequencies = generateRandomFrequencies(1.0, design.cutoffFreqs[0] * 0.8, 200);
                break;
            case FIRFilter::HIGHPASS:
                frequencies = generateRandomFrequencies(design.cutoffFreqs[0] * 1.2, nyquist * 0.9, 200);
                break;
            case FIRFilter::BANDPASS:
                if (design.cutoffFreqs[0] < design.cutoffFreqs[1]) {
                    double passbandStart = design.cutoffFreqs[0] * 1.1;
                    double passbandEnd = design.cutoffFreqs[1] * 0.9;
                    if (passbandStart < passbandEnd) {
                        frequencies = generateRandomFrequencies(passbandStart, passbandEnd, 200);
                    } else {
                        continue; // Skip invalid designs
                    }
                } else {
                    continue; // Skip invalid designs
                }
                break;
            case FIRFilter::BANDSTOP:
                // Test in lower passband
                frequencies = generateRandomFrequencies(1.0, design.cutoffFreqs[0] * 0.8, 200);
                break;
            default:
                frequencies = generateRandomFrequencies(1.0, nyquist * 0.9, 200);
                break;
        }
        
        // Compute frequency response
        VectorXd magnitude, phase;
        filter.frequencyResponse(frequencies, magnitude, phase);
        
        // Test linear phase property
        bool isLinear = isLinearPhase(frequencies, phase, TOLERANCE);
        
        if (isLinear) {
            passedIterations++;
        }
    }
    
    // Verify that most iterations passed (allowing for some numerical errors)
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Linear phase property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    qDebug() << "Pass rate:" << passRate * 100 << "%";
    
    QVERIFY(passRate >= 0.95);  // At least 95% should pass
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhasePropertyLowpass()
{
    // Test linear phase property specifically for lowpass filters
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        // Create lowpass filter design
        FIRFilter::FilterDesign design;
        design.type = FIRFilter::LOWPASS;
        design.linearPhase = true;
        design.samplingRate = 1000.0;
        design.order = 64;
        design.window = FIRFilter::HAMMING;
        
        // Random cutoff frequency
        std::uniform_real_distribution<double> cutoffDist(50.0, 400.0);
        design.cutoffFreqs.resize(1);
        design.cutoffFreqs[0] = cutoffDist(m_rng);
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        // Test in passband (up to cutoff frequency)
        // Increase number of points to avoid phase unwrapping aliasing
        // For order 64, delay is 32 samples. Phase slope ~ 0.2 rad/Hz.
        // Max df < pi/0.2 ~ 15 Hz. With 400Hz range, need > 26 points.
        // Using 200 points to be safe.
        VectorXd frequencies = generateRandomFrequencies(1.0, design.cutoffFreqs[0] * 0.8, 200);
        
        VectorXd magnitude, phase;
        filter.frequencyResponse(frequencies, magnitude, phase);
        
        if (isLinearPhase(frequencies, phase, TOLERANCE)) {
            passedIterations++;
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Lowpass linear phase property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.95);
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhasePropertyHighpass()
{
    // Test linear phase property specifically for highpass filters
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        FIRFilter::FilterDesign design;
        design.type = FIRFilter::HIGHPASS;
        design.linearPhase = true;
        design.samplingRate = 1000.0;
        design.order = 64;
        design.window = FIRFilter::HAMMING;
        
        std::uniform_real_distribution<double> cutoffDist(100.0, 300.0);
        design.cutoffFreqs.resize(1);
        design.cutoffFreqs[0] = cutoffDist(m_rng);
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        // Test in passband (above cutoff frequency)
        // Using 200 points to be safe.
        VectorXd frequencies = generateRandomFrequencies(design.cutoffFreqs[0] * 1.2, 450.0, 200);
        
        VectorXd magnitude, phase;
        filter.frequencyResponse(frequencies, magnitude, phase);
        
        if (isLinearPhase(frequencies, phase, TOLERANCE)) {
            passedIterations++;
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Highpass linear phase property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.95);
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhasePropertyBandpass()
{
    // Test linear phase property specifically for bandpass filters
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        FIRFilter::FilterDesign design;
        design.type = FIRFilter::BANDPASS;
        design.linearPhase = true;
        design.samplingRate = 1000.0;
        design.order = 64;
        design.window = FIRFilter::HAMMING;
        
        std::uniform_real_distribution<double> lowCutoffDist(50.0, 150.0);
        std::uniform_real_distribution<double> highCutoffDist(200.0, 350.0);
        
        design.cutoffFreqs.resize(2);
        design.cutoffFreqs[0] = lowCutoffDist(m_rng);
        design.cutoffFreqs[1] = highCutoffDist(m_rng);
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        // Test in passband (between cutoff frequencies)
        double passbandStart = design.cutoffFreqs[0] * 1.1;
        double passbandEnd = design.cutoffFreqs[1] * 0.9;
        
        if (passbandStart < passbandEnd) {
            // Using 200 points to be safe.
            VectorXd frequencies = generateRandomFrequencies(passbandStart, passbandEnd, 200);
            
            VectorXd magnitude, phase;
            filter.frequencyResponse(frequencies, magnitude, phase);
            
            if (isLinearPhase(frequencies, phase, TOLERANCE)) {
                passedIterations++;
            }
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Bandpass linear phase property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.90);  // Slightly lower threshold for bandpass due to complexity
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhasePropertyBandstop()
{
    // Test linear phase property specifically for bandstop filters
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        FIRFilter::FilterDesign design;
        design.type = FIRFilter::BANDSTOP;
        design.linearPhase = true;
        design.samplingRate = 1000.0;
        design.order = 64;
        design.window = FIRFilter::HAMMING;
        
        std::uniform_real_distribution<double> lowCutoffDist(100.0, 150.0);
        std::uniform_real_distribution<double> highCutoffDist(200.0, 250.0);
        
        design.cutoffFreqs.resize(2);
        design.cutoffFreqs[0] = lowCutoffDist(m_rng);
        design.cutoffFreqs[1] = highCutoffDist(m_rng);
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        // Test in lower passband (below first cutoff)
        // Using 200 points to be safe.
        VectorXd frequencies = generateRandomFrequencies(10.0, design.cutoffFreqs[0] * 0.8, 200);
        
        VectorXd magnitude, phase;
        filter.frequencyResponse(frequencies, magnitude, phase);
        
        if (isLinearPhase(frequencies, phase, TOLERANCE)) {
            passedIterations++;
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Bandstop linear phase property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.90);
}

//=============================================================================================================

void TestFIRLinearPhase::testLinearPhaseConsistency()
{
    // Test that linear phase property is consistent across different window types
    
    std::vector<FIRFilter::WindowType> windows = {
        FIRFilter::HAMMING,
        FIRFilter::HANNING,
        FIRFilter::BLACKMAN,
        FIRFilter::BARTLETT
    };
    
    for (auto window : windows) {
        int passedIterations = 0;
        
        for (int iter = 0; iter < 20; ++iter) {  // Fewer iterations per window
            FIRFilter::FilterDesign design;
            design.type = FIRFilter::LOWPASS;
            design.linearPhase = true;
            design.samplingRate = 1000.0;
            design.order = 64;
            design.window = window;
            design.cutoffFreqs.resize(1);
            design.cutoffFreqs[0] = 200.0;
            
            FIRFilter filter;
            bool designSuccess = filter.designFilter(design);
            
            if (!designSuccess) {
                continue;
            }
            
            // Using 200 points to be safe.
            VectorXd frequencies = generateRandomFrequencies(10.0, 160.0, 200);
            
            VectorXd magnitude, phase;
            filter.frequencyResponse(frequencies, magnitude, phase);
            
            if (isLinearPhase(frequencies, phase, TOLERANCE)) {
                passedIterations++;
            }
        }
        
        double passRate = static_cast<double>(passedIterations) / 20;
        qDebug() << "Window type" << window << "linear phase consistency:" << passRate * 100 << "%";
        
        QVERIFY(passRate >= 0.90);
    }
}

//=============================================================================================================

void TestFIRLinearPhase::testGroupDelayConstancy()
{
    // Test that group delay is constant for linear phase FIR filters
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        FIRFilter::FilterDesign design = generateRandomFilterDesign();
        design.linearPhase = true;
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        // Generate frequency points
        double nyquist = design.samplingRate / 2.0;
        VectorXd frequencies = generateRandomFrequencies(10.0, nyquist * 0.8, 50);
        
        // Compute group delay
        VectorXd groupDelay = filter.groupDelay(frequencies);
        
        if (groupDelay.size() > 1) {
            // Check if group delay is approximately constant
            double meanDelay = groupDelay.mean();
            double maxDeviation = (groupDelay.array() - meanDelay).abs().maxCoeff();
            
            // For linear phase filters, group delay should be constant
            if (maxDeviation < 0.5) {  // Allow small numerical variations
                passedIterations++;
            }
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Group delay constancy passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.95);
}

//=============================================================================================================

void TestFIRLinearPhase::testSymmetricCoefficients()
{
    // Test that linear phase FIR filters have symmetric coefficients
    
    int passedIterations = 0;
    
    for (int iter = 0; iter < NUM_PROPERTY_ITERATIONS; ++iter) {
        FIRFilter::FilterDesign design = generateRandomFilterDesign();
        design.linearPhase = true;
        
        FIRFilter filter;
        bool designSuccess = filter.designFilter(design);
        
        if (!designSuccess) {
            continue;
        }
        
        VectorXd coefficients = filter.getCoefficients();
        
        if (isSymmetric(coefficients, COEFF_TOLERANCE)) {
            passedIterations++;
        }
    }
    
    double passRate = static_cast<double>(passedIterations) / NUM_PROPERTY_ITERATIONS;
    qDebug() << "Symmetric coefficients property passed in" << passedIterations << "out of" << NUM_PROPERTY_ITERATIONS << "iterations";
    
    QVERIFY(passRate >= 0.98);  // Should be very high for this property
}

//=============================================================================================================

void TestFIRLinearPhase::cleanupTestCase()
{
    qDebug() << "FIR Linear Phase Property Tests completed";
}

//=============================================================================================================
// HELPER FUNCTIONS
//=============================================================================================================

VectorXd TestFIRLinearPhase::generateRandomFrequencies(double fmin, double fmax, int nFreqs)
{
    std::uniform_real_distribution<double> freqDist(fmin, fmax);
    VectorXd frequencies(nFreqs);
    
    for (int i = 0; i < nFreqs; ++i) {
        frequencies[i] = freqDist(m_rng);
    }
    
    // Sort frequencies for better phase analysis
    std::sort(frequencies.data(), frequencies.data() + frequencies.size());
    
    return frequencies;
}

//=============================================================================================================

FIRFilter::FilterDesign TestFIRLinearPhase::generateRandomFilterDesign()
{
    FIRFilter::FilterDesign design;
    
    // Random filter type
    std::uniform_int_distribution<int> typeDist(0, 3);  // LOWPASS to BANDSTOP
    design.type = static_cast<FIRFilter::FilterType>(typeDist(m_rng));
    
    // Random window type
    std::uniform_int_distribution<int> windowDist(0, 4);  // Common windows
    design.window = static_cast<FIRFilter::WindowType>(windowDist(m_rng));
    
    // Random parameters
    std::uniform_int_distribution<int> orderDist(32, 128);
    std::uniform_real_distribution<double> sfreqDist(500.0, 2000.0);
    
    design.order = orderDist(m_rng);
    design.samplingRate = sfreqDist(m_rng);
    design.linearPhase = true;
    
    // Set cutoff frequencies based on filter type
    double nyquist = design.samplingRate / 2.0;
    
    switch (design.type) {
        case FIRFilter::LOWPASS:
        case FIRFilter::HIGHPASS:
        {
            design.cutoffFreqs.resize(1);
            std::uniform_real_distribution<double> cutoffDist(nyquist * 0.1, nyquist * 0.8);
            design.cutoffFreqs[0] = cutoffDist(m_rng);
            break;
        }
            
        case FIRFilter::BANDPASS:
        case FIRFilter::BANDSTOP:
        {
            design.cutoffFreqs.resize(2);
            std::uniform_real_distribution<double> lowDist(nyquist * 0.1, nyquist * 0.4);
            std::uniform_real_distribution<double> highDist(nyquist * 0.5, nyquist * 0.8);
            design.cutoffFreqs[0] = lowDist(m_rng);
            design.cutoffFreqs[1] = highDist(m_rng);
            break;
        }
            
        default:
        {
            design.type = FIRFilter::LOWPASS;
            design.cutoffFreqs.resize(1);
            design.cutoffFreqs[0] = nyquist * 0.5;
            break;
        }
    }
    
    return design;
}

//=============================================================================================================

bool TestFIRLinearPhase::isLinearPhase(const VectorXd& frequencies, const VectorXd& phase, double tolerance)
{
    if (frequencies.size() < 3 || phase.size() != frequencies.size()) {
        return false;
    }
    
    // Unwrap phase to handle discontinuities
    VectorXd unwrappedPhase = phase;
    for (int i = 1; i < unwrappedPhase.size(); ++i) {
        double diff = unwrappedPhase[i] - unwrappedPhase[i-1];
        while (diff > M_PI) {
            unwrappedPhase[i] -= 2 * M_PI;
            diff = unwrappedPhase[i] - unwrappedPhase[i-1];
        }
        while (diff < -M_PI) {
            unwrappedPhase[i] += 2 * M_PI;
            diff = unwrappedPhase[i] - unwrappedPhase[i-1];
        }
    }
    
    // Compute linearity measure
    double linearity = computePhaseLinearity(frequencies, unwrappedPhase);
    
    return linearity <= tolerance;
}

//=============================================================================================================

double TestFIRLinearPhase::computePhaseLinearity(const VectorXd& frequencies, const VectorXd& phase)
{
    if (frequencies.size() < 3) {
        return std::numeric_limits<double>::max();
    }
    
    // Fit a line to the phase vs frequency data using least squares
    int n = frequencies.size();
    double sumX = frequencies.sum();
    double sumY = phase.sum();
    double sumXY = frequencies.dot(phase);
    double sumX2 = frequencies.dot(frequencies);
    
    // Linear regression: y = ax + b
    double a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double b = (sumY - a * sumX) / n;
    
    // Compute residual sum of squares
    double rss = 0.0;
    for (int i = 0; i < n; ++i) {
        double predicted = a * frequencies[i] + b;
        double residual = phase[i] - predicted;
        rss += residual * residual;
    }
    
    // Return RMS error as linearity measure
    return std::sqrt(rss / n);
}

//=============================================================================================================

bool TestFIRLinearPhase::isSymmetric(const VectorXd& coefficients, double tolerance)
{
    int n = coefficients.size();
    
    for (int i = 0; i < n / 2; ++i) {
        double diff = std::abs(coefficients[i] - coefficients[n - 1 - i]);
        if (diff > tolerance) {
            return false;
        }
    }
    
    return true;
}

//=============================================================================================================
// MAIN
//=============================================================================================================

QTEST_GUILESS_MAIN(TestFIRLinearPhase)
#include "test_fir_linear_phase.moc"