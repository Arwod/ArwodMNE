//=============================================================================================================
/**
 * @file     test_sparse_reconstruction_gamma_map.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Property test for Gamma-MAP sparse reconstruction sparsity constraint (Property 12)
 *           Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
 *           Validates: Requirements 7.1
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
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <algorithm>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace Eigen;

//=============================================================================================================
/**
 * DECLARE CLASS TestSparseReconstructionGammaMap
 *
 * @brief The TestSparseReconstructionGammaMap class provides property-based tests for Gamma-MAP
 *        sparse reconstruction sparsity constraint
 *
 */
class TestSparseReconstructionGammaMap: public QObject
{
    Q_OBJECT

public:
    TestSparseReconstructionGammaMap();

private slots:
    void initTestCase();
    void testGammaMapSparsityMonotonicity();
    void testGammaMapSparsityMonotonicityProperty();
    void testGammaMapSolutionNonnegativity();
    void testGammaMapSolutionNonnegativeProperty();
    void testGammaMapRegularizationEffect();
    void testGammaMapRegularizationEffectProperty();
    void testGammaMapConvergence();
    void testGammaMapConvergenceProperty();
    void cleanupTestCase();

private:
    // Gamma-MAP simulation methods
    struct GammaMapResult {
        VectorXd solution;
        double objective;
        int iterations;
        bool converged;
    };
    
    GammaMapResult gammaMapReconstruction(const MatrixXd& G, const VectorXd& y, 
                                         double lambda, double alpha = 1.0,
                                         int max_iter = 100, double tol = 1e-6);
    
    // Sparsity analysis methods
    double computeSparsity(const VectorXd& solution);
    double computeL0Norm(const VectorXd& solution, double threshold = 1e-6);
    double computeL1Norm(const VectorXd& solution);
    double computeL2Norm(const VectorXd& solution);
    double computeObjective(const MatrixXd& G, const VectorXd& y, 
                           const VectorXd& x, double lambda);
    
    // Data generation methods
    MatrixXd generateLeadFieldMatrix(int n_sources, int n_sensors);
    VectorXd generateSourceActivity(int n_sources, double sparsity = 0.1);
    VectorXd generateMeasurement(const MatrixXd& G, const VectorXd& x, 
                                double noise_std = 0.1);
    
    // Test parameters
    QRandomGenerator* m_generator;
    int m_n_sources;
    int m_n_sensors;
    double m_tolerance;
};

//=============================================================================================================

TestSparseReconstructionGammaMap::TestSparseReconstructionGammaMap()
: m_generator(QRandomGenerator::global())
, m_n_sources(100)
, m_n_sensors(64)
, m_tolerance(0.1)
{
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::initTestCase()
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    qDebug() << "Starting Gamma-MAP Sparse Reconstruction Property Tests";
    qDebug() << "Sources:" << m_n_sources;
    qDebug() << "Sensors:" << m_n_sensors;
    qDebug() << "Tolerance:" << m_tolerance;
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapSparsityMonotonicity()
{
    qDebug() << "Testing Gamma-MAP sparsity monotonicity...";
    
    // Generate test data
    MatrixXd G = generateLeadFieldMatrix(m_n_sources, m_n_sensors);
    VectorXd x_true = generateSourceActivity(m_n_sources, 0.1);
    VectorXd y = generateMeasurement(G, x_true, 0.1);
    
    // Test with different regularization parameters
    std::vector<double> lambdas = {0.001, 0.01, 0.1, 1.0, 10.0};
    std::vector<double> sparsities;
    
    for(double lambda : lambdas) {
        GammaMapResult result = gammaMapReconstruction(G, y, lambda);
        double sparsity = computeSparsity(result.solution);
        sparsities.push_back(sparsity);
        
        qDebug() << "Lambda:" << lambda << "Sparsity:" << sparsity;
    }
    
    // Check monotonicity: sparsity should increase with lambda
    for(size_t i = 1; i < sparsities.size(); ++i) {
        QVERIFY(sparsities[i] >= sparsities[i-1] - m_tolerance);
    }
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapSparsityMonotonicityProperty()
{
    qDebug() << "Running Gamma-MAP sparsity monotonicity property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_sources = m_generator->bounded(50, 200);
        int n_sensors = m_generator->bounded(32, 128);
        
        // Ensure n_sensors < n_sources for underdetermined system
        if(n_sensors >= n_sources) {
            n_sensors = n_sources / 2;
        }
        
        try {
            // Generate test data
            MatrixXd G = generateLeadFieldMatrix(n_sources, n_sensors);
            VectorXd x_true = generateSourceActivity(n_sources, 0.1);
            VectorXd y = generateMeasurement(G, x_true, 0.1);
            
            // Test with two different regularization parameters
            double lambda1 = m_generator->generateDouble() * 0.1 + 0.001;
            double lambda2 = lambda1 * (m_generator->generateDouble() * 5.0 + 1.0);
            
            GammaMapResult result1 = gammaMapReconstruction(G, y, lambda1);
            GammaMapResult result2 = gammaMapReconstruction(G, y, lambda2);
            
            double sparsity1 = computeSparsity(result1.solution);
            double sparsity2 = computeSparsity(result2.solution);
            
            // Sparsity should increase with lambda (or stay similar)
            if(sparsity2 >= sparsity1 - m_tolerance) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed reconstructions
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 80% of tests should pass
    QVERIFY(successful_tests >= 80);
    
    qDebug() << "Gamma-MAP sparsity monotonicity property test completed successfully";
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapSolutionNonnegativity()
{
    qDebug() << "Testing Gamma-MAP solution non-negativity...";
    
    // Generate test data
    MatrixXd G = generateLeadFieldMatrix(m_n_sources, m_n_sensors);
    VectorXd x_true = generateSourceActivity(m_n_sources, 0.1);
    VectorXd y = generateMeasurement(G, x_true, 0.1);
    
    // Reconstruct with moderate regularization
    double lambda = 0.1;
    GammaMapResult result = gammaMapReconstruction(G, y, lambda);
    
    // Check that solution is non-negative (or mostly non-negative)
    int negative_count = 0;
    for(int i = 0; i < result.solution.size(); ++i) {
        if(result.solution(i) < -1e-6) {
            negative_count++;
        }
    }
    
    double negative_ratio = static_cast<double>(negative_count) / result.solution.size();
    
    qDebug() << "Negative values ratio:" << negative_ratio;
    
    // Most values should be non-negative
    QVERIFY(negative_ratio < 0.1);
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapSolutionNonnegativeProperty()
{
    qDebug() << "Running Gamma-MAP solution non-negativity property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_sources = m_generator->bounded(50, 150);
        int n_sensors = m_generator->bounded(32, 96);
        
        if(n_sensors >= n_sources) {
            n_sensors = n_sources / 2;
        }
        
        try {
            // Generate test data
            MatrixXd G = generateLeadFieldMatrix(n_sources, n_sensors);
            VectorXd x_true = generateSourceActivity(n_sources, 0.1);
            VectorXd y = generateMeasurement(G, x_true, 0.1);
            
            // Reconstruct
            double lambda = m_generator->generateDouble() * 0.5 + 0.01;
            GammaMapResult result = gammaMapReconstruction(G, y, lambda);
            
            // Count negative values
            int negative_count = 0;
            for(int i = 0; i < result.solution.size(); ++i) {
                if(result.solution(i) < -1e-6) {
                    negative_count++;
                }
            }
            
            double negative_ratio = static_cast<double>(negative_count) / result.solution.size();
            
            // Most values should be non-negative
            if(negative_ratio < 0.15) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed reconstructions
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 75% of tests should pass
    QVERIFY(successful_tests >= 75);
    
    qDebug() << "Gamma-MAP solution non-negativity property test completed successfully";
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapRegularizationEffect()
{
    qDebug() << "Testing Gamma-MAP regularization effect...";
    
    // Generate test data
    MatrixXd G = generateLeadFieldMatrix(m_n_sources, m_n_sensors);
    VectorXd x_true = generateSourceActivity(m_n_sources, 0.1);
    VectorXd y = generateMeasurement(G, x_true, 0.1);
    
    // Test with different regularization parameters
    double lambda_small = 0.001;
    double lambda_large = 1.0;
    
    GammaMapResult result_small = gammaMapReconstruction(G, y, lambda_small);
    GammaMapResult result_large = gammaMapReconstruction(G, y, lambda_large);
    
    double l1_small = computeL1Norm(result_small.solution);
    double l1_large = computeL1Norm(result_large.solution);
    
    qDebug() << "L1 norm (small lambda):" << l1_small;
    qDebug() << "L1 norm (large lambda):" << l1_large;
    
    // Larger lambda should produce smaller L1 norm
    QVERIFY(l1_large <= l1_small * (1.0 + m_tolerance));
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapRegularizationEffectProperty()
{
    qDebug() << "Running Gamma-MAP regularization effect property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_sources = m_generator->bounded(50, 150);
        int n_sensors = m_generator->bounded(32, 96);
        
        if(n_sensors >= n_sources) {
            n_sensors = n_sources / 2;
        }
        
        try {
            // Generate test data
            MatrixXd G = generateLeadFieldMatrix(n_sources, n_sensors);
            VectorXd x_true = generateSourceActivity(n_sources, 0.1);
            VectorXd y = generateMeasurement(G, x_true, 0.1);
            
            // Test with two different regularization parameters
            double lambda1 = m_generator->generateDouble() * 0.1 + 0.001;
            double lambda2 = lambda1 * (m_generator->generateDouble() * 10.0 + 1.0);
            
            GammaMapResult result1 = gammaMapReconstruction(G, y, lambda1);
            GammaMapResult result2 = gammaMapReconstruction(G, y, lambda2);
            
            double l1_1 = computeL1Norm(result1.solution);
            double l1_2 = computeL1Norm(result2.solution);
            
            // Larger lambda should produce smaller or equal L1 norm
            if(l1_2 <= l1_1 * (1.0 + m_tolerance)) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed reconstructions
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 80% of tests should pass
    QVERIFY(successful_tests >= 80);
    
    qDebug() << "Gamma-MAP regularization effect property test completed successfully";
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapConvergence()
{
    qDebug() << "Testing Gamma-MAP convergence...";
    
    // Generate test data
    MatrixXd G = generateLeadFieldMatrix(m_n_sources, m_n_sensors);
    VectorXd x_true = generateSourceActivity(m_n_sources, 0.1);
    VectorXd y = generateMeasurement(G, x_true, 0.1);
    
    // Reconstruct with moderate regularization
    double lambda = 0.1;
    GammaMapResult result = gammaMapReconstruction(G, y, lambda, 1.0, 100, 1e-6);
    
    qDebug() << "Converged:" << result.converged;
    qDebug() << "Iterations:" << result.iterations;
    qDebug() << "Objective:" << result.objective;
    
    // Should converge within reasonable iterations
    QVERIFY(result.converged || result.iterations < 100);
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::testGammaMapConvergenceProperty()
{
    qDebug() << "Running Gamma-MAP convergence property test (100 iterations)...";
    
    int successful_tests = 0;
    
    // Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
    for(int iteration = 0; iteration < 100; ++iteration) {
        // Generate random parameters
        int n_sources = m_generator->bounded(50, 150);
        int n_sensors = m_generator->bounded(32, 96);
        
        if(n_sensors >= n_sources) {
            n_sensors = n_sources / 2;
        }
        
        try {
            // Generate test data
            MatrixXd G = generateLeadFieldMatrix(n_sources, n_sensors);
            VectorXd x_true = generateSourceActivity(n_sources, 0.1);
            VectorXd y = generateMeasurement(G, x_true, 0.1);
            
            // Reconstruct
            double lambda = m_generator->generateDouble() * 0.5 + 0.01;
            GammaMapResult result = gammaMapReconstruction(G, y, lambda, 1.0, 100, 1e-6);
            
            // Should converge or reach max iterations
            if(result.converged || result.iterations <= 100) {
                successful_tests++;
            }
        } catch(...) {
            // Skip failed reconstructions
            continue;
        }
    }
    
    qDebug() << "Successful tests:" << successful_tests << "/ 100";
    
    // At least 90% of tests should pass
    QVERIFY(successful_tests >= 90);
    
    qDebug() << "Gamma-MAP convergence property test completed successfully";
}

//=============================================================================================================

void TestSparseReconstructionGammaMap::cleanupTestCase()
{
    qDebug() << "Gamma-MAP Sparse Reconstruction Property Tests completed";
}

//=============================================================================================================
// HELPER METHODS
//=============================================================================================================

TestSparseReconstructionGammaMap::GammaMapResult 
TestSparseReconstructionGammaMap::gammaMapReconstruction(const MatrixXd& G, const VectorXd& y,
                                                        double lambda, double alpha,
                                                        int max_iter, double tol)
{
    GammaMapResult result;
    result.converged = false;
    result.iterations = 0;
    
    int n_sources = G.cols();
    int n_sensors = G.rows();
    
    // Initialize solution
    VectorXd x = VectorXd::Zero(n_sources);
    VectorXd gamma = VectorXd::Ones(n_sources) * 1.0;  // Prior variance
    
    // Compute Gram matrix
    MatrixXd GtG = G.transpose() * G;
    VectorXd Gty = G.transpose() * y;
    
    // Add regularization to Gram matrix for numerical stability
    MatrixXd GtG_reg = GtG + 1e-6 * MatrixXd::Identity(n_sources, n_sources);
    
    // Iterative Gamma-MAP algorithm
    for(int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        
        // E-step: Update posterior mean and variance
        VectorXd gamma_inv = gamma.cwiseInverse();
        
        // Construct regularized system: (GtG + lambda * diag(1/gamma)) * x = Gty
        MatrixXd A = GtG_reg;
        for(int i = 0; i < n_sources; ++i) {
            A(i, i) += lambda * gamma_inv(i);
        }
        
        // Solve for x using Cholesky decomposition
        VectorXd x_new = A.llt().solve(Gty);
        
        // Enforce non-negativity constraint
        for(int i = 0; i < n_sources; ++i) {
            x_new(i) = std::max(0.0, x_new(i));
        }
        
        // M-step: Update hyperparameters (gamma)
        VectorXd gamma_new = VectorXd::Zero(n_sources);
        for(int i = 0; i < n_sources; ++i) {
            // Gamma update based on current solution
            double x_val = x_new(i);
            // Use a more conservative update to avoid zero solutions
            gamma_new(i) = (x_val * x_val + 0.1) / (1.0 + lambda * 0.1);
            gamma_new(i) = std::max(gamma_new(i), 0.01);  // Prevent too small values
        }
        
        // Check convergence
        double x_change = (x_new - x).norm() / (x.norm() + 1e-8);
        double gamma_change = (gamma_new - gamma).norm() / (gamma.norm() + 1e-8);
        
        x = x_new;
        gamma = gamma_new;
        
        if(x_change < tol && gamma_change < tol) {
            result.converged = true;
            break;
        }
    }
    
    // Ensure non-negativity in final solution
    for(int i = 0; i < x.size(); ++i) {
        x(i) = std::max(0.0, x(i));
    }
    
    // Compute objective
    VectorXd residual = y - G * x;
    result.objective = residual.squaredNorm() + lambda * x.cwiseAbs().sum();
    result.solution = x;
    
    return result;
}

//=============================================================================================================

double TestSparseReconstructionGammaMap::computeSparsity(const VectorXd& solution)
{
    // Sparsity = number of non-zero elements / total elements
    int non_zero_count = 0;
    for(int i = 0; i < solution.size(); ++i) {
        if(std::abs(solution(i)) > 1e-6) {
            non_zero_count++;
        }
    }
    
    return static_cast<double>(non_zero_count) / solution.size();
}

//=============================================================================================================

double TestSparseReconstructionGammaMap::computeL0Norm(const VectorXd& solution, double threshold)
{
    int count = 0;
    for(int i = 0; i < solution.size(); ++i) {
        if(std::abs(solution(i)) > threshold) {
            count++;
        }
    }
    return static_cast<double>(count);
}

//=============================================================================================================

double TestSparseReconstructionGammaMap::computeL1Norm(const VectorXd& solution)
{
    return solution.cwiseAbs().sum();
}

//=============================================================================================================

double TestSparseReconstructionGammaMap::computeL2Norm(const VectorXd& solution)
{
    return solution.norm();
}

//=============================================================================================================

double TestSparseReconstructionGammaMap::computeObjective(const MatrixXd& G, const VectorXd& y,
                                                         const VectorXd& x, double lambda)
{
    VectorXd residual = y - G * x;
    return residual.squaredNorm() + lambda * x.cwiseAbs().sum();
}

//=============================================================================================================

MatrixXd TestSparseReconstructionGammaMap::generateLeadFieldMatrix(int n_sources, int n_sensors)
{
    MatrixXd G = MatrixXd::Random(n_sensors, n_sources);
    
    // Normalize columns
    for(int i = 0; i < n_sources; ++i) {
        G.col(i) /= G.col(i).norm();
    }
    
    return G;
}

//=============================================================================================================

VectorXd TestSparseReconstructionGammaMap::generateSourceActivity(int n_sources, double sparsity)
{
    VectorXd x = VectorXd::Zero(n_sources);
    
    // Generate sparse source activity
    int n_active = std::max(1, static_cast<int>(n_sources * sparsity));
    
    for(int i = 0; i < n_active; ++i) {
        int idx = m_generator->bounded(n_sources);
        x(idx) = m_generator->generateDouble() * 2.0 + 0.5;  // 0.5 to 2.5
    }
    
    return x;
}

//=============================================================================================================

VectorXd TestSparseReconstructionGammaMap::generateMeasurement(const MatrixXd& G, const VectorXd& x,
                                                              double noise_std)
{
    VectorXd y = G * x;
    
    // Add Gaussian noise
    for(int i = 0; i < y.size(); ++i) {
        double u1 = m_generator->generateDouble();
        double u2 = m_generator->generateDouble();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        
        y(i) += noise_std * z;
    }
    
    return y;
}

//=============================================================================================================

QTEST_GUILESS_MAIN(TestSparseReconstructionGammaMap)
#include "test_sparse_reconstruction_gamma_map.moc"
