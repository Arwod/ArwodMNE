#include <QtTest/QtTest>
#include <inverse/sparseInverse/gamma_map.h>
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace INVERSELIB;
using namespace Eigen;

/**
 * @brief Property-Based Test for Sparse Reconstruction Algorithms
 * 
 * Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
 * Validates: Requirements 7.1
 * 
 * This test verifies that sparse inverse algorithms produce increasingly sparse
 * solutions as the regularization parameter increases. For Gamma-MAP, this is
 * controlled by the beta parameter (rate parameter of the Gamma prior).
 * 
 * Property: For any sparse reconstruction algorithm and input data, increasing
 * the regularization parameter should result in increased sparsity (more sources
 * with near-zero variance).
 */
class TestSparseReconstruction : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testGammaMapSparsityConstraint();

private:
    std::mt19937 m_rng;
    
    // Helper function to generate synthetic leadfield matrix
    MatrixXd generateLeadfield(int n_sensors, int n_sources);
    
    // Helper function to generate synthetic sensor data with sparse sources
    MatrixXd generateSparseSourceData(const MatrixXd& leadfield, int n_active_sources, 
                                     int n_times, double noise_level);
    
    // Helper function to measure sparsity of source variances
    double computeSparsity(const VectorXd& source_variances, double threshold = 1e-6);
};

void TestSparseReconstruction::initTestCase()
{
    // Initialize random number generator with a fixed seed for reproducibility
    m_rng.seed(42);
}

void TestSparseReconstruction::cleanupTestCase()
{
}

MatrixXd TestSparseReconstruction::generateLeadfield(int n_sensors, int n_sources)
{
    // Generate a random leadfield matrix with reasonable conditioning
    std::normal_distribution<double> dist(0.0, 1.0);
    
    MatrixXd leadfield(n_sensors, n_sources);
    for (int i = 0; i < n_sensors; ++i) {
        for (int j = 0; j < n_sources; ++j) {
            leadfield(i, j) = dist(m_rng);
        }
    }
    
    // Normalize columns to have unit norm
    for (int j = 0; j < n_sources; ++j) {
        double norm = leadfield.col(j).norm();
        if (norm > 1e-12) {
            leadfield.col(j) /= norm;
        }
    }
    
    return leadfield;
}

MatrixXd TestSparseReconstruction::generateSparseSourceData(const MatrixXd& leadfield, 
                                                           int n_active_sources,
                                                           int n_times, 
                                                           double noise_level)
{
    int n_sensors = leadfield.rows();
    int n_sources = leadfield.cols();
    
    // Ensure n_active_sources is valid
    n_active_sources = std::min(n_active_sources, n_sources);
    
    // Generate sparse source activity
    MatrixXd sources = MatrixXd::Zero(n_sources, n_times);
    
    // Randomly select active sources
    std::vector<int> all_indices(n_sources);
    for (int i = 0; i < n_sources; ++i) {
        all_indices[i] = i;
    }
    std::shuffle(all_indices.begin(), all_indices.end(), m_rng);
    
    std::normal_distribution<double> signal_dist(0.0, 1.0);
    for (int i = 0; i < n_active_sources; ++i) {
        int source_idx = all_indices[i];
        for (int t = 0; t < n_times; ++t) {
            sources(source_idx, t) = signal_dist(m_rng);
        }
    }
    
    // Generate sensor data: data = leadfield * sources + noise
    MatrixXd data = leadfield * sources;
    
    // Add noise
    std::normal_distribution<double> noise_dist(0.0, noise_level);
    for (int i = 0; i < n_sensors; ++i) {
        for (int t = 0; t < n_times; ++t) {
            data(i, t) += noise_dist(m_rng);
        }
    }
    
    return data;
}

double TestSparseReconstruction::computeSparsity(const VectorXd& source_variances, 
                                                double threshold)
{
    int sparse_count = 0;
    for (int i = 0; i < source_variances.size(); ++i) {
        if (source_variances(i) < threshold) {
            sparse_count++;
        }
    }
    
    return (double)sparse_count / (double)source_variances.size();
}

void TestSparseReconstruction::testGammaMapSparsityConstraint()
{
    // Feature: mne-python-to-cpp-migration, Property 12: 稀疏重建稀疏性约束
    // Run 100 iterations with random data
    const int n_iterations = 100;
    int passed = 0;
    int monotonicity_violations = 0;
    
    qDebug() << "Testing Gamma-MAP Sparsity Constraint Property...";
    qDebug() << "Property: Sparsity should increase with regularization parameter (beta)";
    
    for (int iter = 0; iter < n_iterations; ++iter) {
        // Generate random problem parameters
        std::uniform_int_distribution<int> sensor_dist(10, 30);
        std::uniform_int_distribution<int> source_dist(20, 50);
        std::uniform_int_distribution<int> time_dist(10, 50);
        std::uniform_int_distribution<int> active_dist(2, 10);
        std::uniform_real_distribution<double> noise_dist(0.01, 0.5);
        
        int n_sensors = sensor_dist(m_rng);
        int n_sources = source_dist(m_rng);
        int n_times = time_dist(m_rng);
        int n_active = active_dist(m_rng);
        double noise_level = noise_dist(m_rng);
        
        // Generate synthetic data
        MatrixXd leadfield = generateLeadfield(n_sensors, n_sources);
        MatrixXd data = generateSparseSourceData(leadfield, n_active, n_times, noise_level);
        
        // Test with increasing beta values (regularization parameter)
        std::vector<double> beta_values = {0.1, 0.5, 1.0, 2.0, 5.0};
        std::vector<double> sparsity_levels;
        
        bool iteration_passed = true;
        
        for (double beta : beta_values) {
            // Configure Gamma-MAP with current beta
            GammaMAP::Hyperparameters hyperparams;
            hyperparams.alpha = 1.0;
            hyperparams.beta = beta;
            hyperparams.noise_variance = noise_level * noise_level;
            hyperparams.update_noise = false; // Keep noise fixed for fair comparison
            hyperparams.use_ard = true;
            
            GammaMAP::ConvergenceCriteria convergence;
            convergence.tolerance = 1e-6;
            convergence.max_iterations = 50; // Reduced for speed
            convergence.check_cost_function = true;
            convergence.cost_tolerance = 1e-8;
            
            // Create and fit Gamma-MAP
            GammaMAP solver(leadfield, hyperparams, convergence);
            bool fit_success = solver.fit(data);
            
            if (!fit_success) {
                qDebug() << "Iteration" << iter << ": Gamma-MAP fit failed for beta =" << beta;
                iteration_passed = false;
                break;
            }
            
            // Get sparsity level
            double sparsity = solver.getSparsityLevel(1e-6);
            sparsity_levels.push_back(sparsity);
        }
        
        if (!iteration_passed) {
            continue;
        }
        
        // Verify monotonicity: sparsity should increase (or stay same) with beta
        bool monotonic = true;
        for (size_t i = 1; i < sparsity_levels.size(); ++i) {
            if (sparsity_levels[i] < sparsity_levels[i-1] - 1e-6) {
                monotonic = false;
                monotonicity_violations++;
                
                if (iter < 5) { // Only print details for first few failures
                    qDebug() << "Iteration" << iter << ": Monotonicity violation";
                    qDebug() << "Beta values:" << beta_values[i-1] << "->" << beta_values[i];
                    qDebug() << "Sparsity:" << sparsity_levels[i-1] << "->" << sparsity_levels[i];
                }
                break;
            }
        }
        
        if (monotonic) {
            passed++;
        }
        
        // Additional verification: Check that sparsity increases overall
        double first_sparsity = sparsity_levels.front();
        double last_sparsity = sparsity_levels.back();
        
        QVERIFY2(last_sparsity >= first_sparsity - 1e-6,
                qPrintable(QString("Iteration %1: Overall sparsity decreased from %2 to %3")
                          .arg(iter).arg(first_sparsity).arg(last_sparsity)));
        
        // Verify sparsity is in valid range [0, 1]
        for (size_t i = 0; i < sparsity_levels.size(); ++i) {
            QVERIFY2(sparsity_levels[i] >= 0.0 && sparsity_levels[i] <= 1.0,
                    qPrintable(QString("Iteration %1: Sparsity out of range [0,1]: %2")
                              .arg(iter).arg(sparsity_levels[i])));
        }
    }
    
    qDebug() << "\nGamma-MAP Sparsity Constraint Property Test Results:";
    qDebug() << "Passed:" << passed << "/" << n_iterations << "iterations";
    qDebug() << "Monotonicity violations:" << monotonicity_violations;
    
    // Require at least 90% success rate (allowing some numerical instability)
    QVERIFY2(passed >= n_iterations * 0.90, 
            qPrintable(QString("Only %1/%2 iterations passed (< 90%)")
                      .arg(passed).arg(n_iterations)));
    
    qDebug() << "\nProperty verified: Sparsity increases with regularization parameter";
}

QTEST_GUILESS_MAIN(TestSparseReconstruction)
#include "test_sparse_reconstruction.moc"
