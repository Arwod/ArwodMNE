//=============================================================================================================
/**
 * @file     main.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Integration test for all MNE-CPP algorithm modules.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <integration.h>
#include <benchmark.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QCoreApplication>
#include <QDebug>
#include <QTimer>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace MNELIB;

//=============================================================================================================
// MAIN
//=============================================================================================================

/**
 * The function main marks the entry point of the program.
 * By default, main has the storage class extern.
 *
 * @param[in] argc (argument count) is an integer that indicates how many arguments were entered on the command line when the program was started.
 * @param[in] argv (argument vector) is an array of pointers to arrays of character objects. The array objects are null-terminated strings, representing the arguments that were entered on the command line when the program was started.
 *
 * @return the value that was set to the exit() function (which is 0 if exit() is called via quit()).
 */
int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    qDebug() << "==============================================";
    qDebug() << "MNE-CPP Algorithms Integration Test";
    qDebug() << "==============================================";

    // Create the module integration manager
    Integration::SPtr integration = Integration::SPtr::create();

    // Initialize all modules
    qDebug() << "\n[TEST] Initializing algorithm modules...";
    bool initSuccess = integration->initializeModules();
    
    if (!initSuccess) {
        qCritical() << "[ERROR] Failed to initialize algorithm modules";
        return -1;
    }

    qDebug() << "[SUCCESS] Algorithm modules initialized successfully";

    // Get available modules
    qDebug() << "\n[TEST] Checking available modules...";
    QStringList availableModules = integration->getAvailableModules();
    qDebug() << "[INFO] Available modules:" << availableModules.size();
    for (const QString& module : availableModules) {
        bool isAvailable = integration->isModuleAvailable(module);
        qDebug() << "  -" << module << (isAvailable ? "[AVAILABLE]" : "[NOT AVAILABLE]");
    }

    // Verify module compatibility
    qDebug() << "\n[TEST] Verifying module compatibility...";
    bool compatibilityOk = integration->verifyModuleCompatibility();
    
    if (!compatibilityOk) {
        qWarning() << "[WARNING] Some module compatibility issues detected";
    } else {
        qDebug() << "[SUCCESS] All modules are compatible";
    }

    // Display version information
    qDebug() << "\n[INFO] Version Information:";
    qDebug() << integration->getVersionInfo();

    // Display performance information
    qDebug() << "\n[INFO] Performance Information:";
    qDebug() << integration->getPerformanceInfo();

    // Test specific module availability
    qDebug() << "\n[TEST] Testing specific algorithm modules...";
    
    QStringList criticalModules = {
        "mne_connectivity",    // Task 16: Connectivity analysis
        "mne_dataio",         // Task 14: Data I/O module
        "mne_channels",       // Task 15: Channel and montage management
        "mne_rtprocessing",   // Task 17: Filter algorithm enhancement
        "mne_decoding",       // Task 9: Decoding and machine learning
        "mne_stats"           // Task 8: Statistical analysis
    };

    int successCount = 0;
    for (const QString& module : criticalModules) {
        bool available = integration->isModuleAvailable(module);
        qDebug() << QString("[%1] %2").arg(available ? "PASS" : "FAIL").arg(module);
        if (available) successCount++;
    }

    qDebug() << "\n==============================================";
    qDebug() << QString("Integration Test Results: %1/%2 modules available")
                .arg(successCount).arg(criticalModules.size());
    
    if (successCount == criticalModules.size()) {
        qDebug() << "[SUCCESS] All critical algorithm modules are integrated and available";
        qDebug() << "[SUCCESS] Task 18.1 - Module Integration: COMPLETED";
    } else {
        qWarning() << "[WARNING] Some critical modules are not available";
    }
    
    // Run performance benchmarks
    qDebug() << "\n[TEST] Running performance benchmarks...";
    Benchmark::SPtr benchmark = Benchmark::SPtr::create();
    
    QMap<QString, Benchmark::BenchmarkResult> benchmarkResults = benchmark->runComprehensiveBenchmarks();
    
    qDebug() << "\n[INFO] Performance Report:";
    qDebug() << benchmark->generatePerformanceReport(benchmarkResults);
    
    qDebug() << "\n[INFO] Performance Comparison:";
    qDebug() << benchmark->comparePerformance(benchmarkResults);
    
    // Check if performance benchmarks passed
    int benchmarksPassed = 0;
    int totalBenchmarks = benchmarkResults.size();
    
    for (auto it = benchmarkResults.begin(); it != benchmarkResults.end(); ++it) {
        if (it.value().success) {
            benchmarksPassed++;
        }
    }
    
    qDebug() << "\n==============================================";
    qDebug() << QString("Performance Test Results: %1/%2 benchmarks passed")
                .arg(benchmarksPassed).arg(totalBenchmarks);
    
    if (benchmarksPassed == totalBenchmarks) {
        qDebug() << "[SUCCESS] All performance benchmarks completed successfully";
        qDebug() << "[SUCCESS] Task 18.3 - Performance Benchmarking: COMPLETED";
    } else {
        qWarning() << "[WARNING] Some performance benchmarks failed";
    }
    
    qDebug() << "==============================================";

    // Exit after a short delay to allow output to be displayed
    QTimer::singleShot(100, &app, &QCoreApplication::quit);
    
    return app.exec();
}