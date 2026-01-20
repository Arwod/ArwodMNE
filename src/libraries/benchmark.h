//=============================================================================================================
/**
 * @file     performance_benchmark.h
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Performance benchmarking utilities for MNE-CPP algorithms.
 *
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QObject>
#include <QSharedPointer>
#include <QString>
#include <QStringList>
#include <QElapsedTimer>
#include <QMap>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Dense>

//=============================================================================================================
// DEFINE NAMESPACE MNELIB
//=============================================================================================================

namespace MNELIB
{

//=============================================================================================================
/**
 * Performance benchmarking utilities for MNE-CPP algorithms.
 * This class provides tools to measure and compare algorithm performance
 * against reference implementations.
 *
 * @brief Performance benchmarking utilities
 */
class Benchmark : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<Benchmark> SPtr;            /**< Shared pointer type for Benchmark. */
    typedef QSharedPointer<const Benchmark> ConstSPtr; /**< Const shared pointer type for Benchmark. */

    //=========================================================================================================
    /**
     * Benchmark result structure.
     */
    struct BenchmarkResult {
        QString algorithmName;      /**< Name of the algorithm. */
        double executionTime;       /**< Execution time in milliseconds. */
        double memoryUsage;         /**< Memory usage in MB. */
        double throughput;          /**< Throughput (operations per second). */
        bool success;               /**< Whether the benchmark completed successfully. */
        QString errorMessage;       /**< Error message if benchmark failed. */
    };

    //=========================================================================================================
    /**
     * Constructs the performance benchmark utility.
     *
     * @param[in] parent     Parent QObject (optional).
     */
    explicit Benchmark(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destructor.
     */
    ~Benchmark();

    //=========================================================================================================
    /**
     * Run comprehensive performance benchmarks for all algorithm modules.
     *
     * @return QMap of algorithm names to benchmark results.
     */
    QMap<QString, BenchmarkResult> runComprehensiveBenchmarks();

    //=========================================================================================================
    /**
     * Benchmark connectivity analysis algorithms.
     *
     * @return BenchmarkResult for connectivity algorithms.
     */
    BenchmarkResult benchmarkConnectivityAnalysis();

    //=========================================================================================================
    /**
     * Benchmark filtering algorithms.
     *
     * @return BenchmarkResult for filtering algorithms.
     */
    BenchmarkResult benchmarkFilteringAlgorithms();

    //=========================================================================================================
    /**
     * Benchmark decoding algorithms.
     *
     * @return BenchmarkResult for decoding algorithms.
     */
    BenchmarkResult benchmarkDecodingAlgorithms();

    //=========================================================================================================
    /**
     * Benchmark statistical analysis algorithms.
     *
     * @return BenchmarkResult for statistical algorithms.
     */
    BenchmarkResult benchmarkStatisticalAnalysis();

    //=========================================================================================================
    /**
     * Generate performance report.
     *
     * @param[in] results    Benchmark results to include in report.
     *
     * @return QString containing formatted performance report.
     */
    QString generatePerformanceReport(const QMap<QString, BenchmarkResult>& results);

    //=========================================================================================================
    /**
     * Compare performance against reference values.
     *
     * @param[in] results    Current benchmark results.
     *
     * @return QString containing performance comparison.
     */
    QString comparePerformance(const QMap<QString, BenchmarkResult>& results);

private:
    //=========================================================================================================
    /**
     * Generate synthetic test data for benchmarking.
     *
     * @param[in] channels    Number of channels.
     * @param[in] samples     Number of samples.
     *
     * @return Eigen::MatrixXd containing synthetic data.
     */
    Eigen::MatrixXd generateSyntheticData(int channels, int samples);

    //=========================================================================================================
    /**
     * Measure memory usage.
     *
     * @return Current memory usage in MB.
     */
    double measureMemoryUsage();

    QElapsedTimer           m_timer;                /**< Timer for measuring execution time. */
    QMap<QString, double>   m_referenceValues;     /**< Reference performance values. */
};

} // namespace MNELIB

#endif // BENCHMARK_H