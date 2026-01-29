#include <QtTest/QtTest>
#include <decoding/csp.h>
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace DECODINGLIB;
using namespace Eigen;

/**
 * @brief Property-Based Test for CSP Algorithm
 * 
 * Feature: mne-python-to-cpp-migration, Property 11: CSP方差比最大化
 * Validates: Requirements 6.1
 * 
 * This test verifies that CSP filters maximize the variance ratio between two classes.
 * For any two classes of data, the CSP filters should satisfy the generalized eigenvalue
 * problem: C_a * w = lambda * (C_a + C_b) * w, where C_a and C_b are the covariance
 * matrices of the two classes.
 */
class TestCSPProperties : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCSPVarianceRatioMaximization();

private:
    std::mt19937 m_rng;
    
    // Helper function to generate random epochs with controlled covariance structure
    std::vector<MatrixXd> generateRandomEpochs(int n_epochs, int n_channels, int n_times, 
                                               const MatrixXd& cov_structure);
    
    // Helper function to compute covariance matrix from epochs
    MatrixXd computeAverageCovariance(const std::vector<MatrixXd>& epochs);
    
    // Helper function to verify eigenvalue problem
    bool verifyGeneralizedEigenvalueProblem(const MatrixXd& cov_a, const MatrixXd& cov_b,
                                           const MatrixXd& filters, const VectorXd& eigenvalues,
                                           double tolerance = 1e-6);
};

void TestCSPProperties::initTestCase()
{
    // Initialize random number generator with a fixed seed for reproducibility
    m_rng.seed(42);
}

void TestCSPProperties::cleanupTestCase()
{
}

std::vector<MatrixXd> TestCSPProperties::generateRandomEpochs(int n_epochs, int n_channels, 
                                                              int n_times, const MatrixXd& cov_structure)
{
    std::vector<MatrixXd> epochs;
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Compute Cholesky decomposition of covariance structure
    LLT<MatrixXd> llt(cov_structure);
    MatrixXd L = llt.matrixL();
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd white_noise(n_channels, n_times);
        for (int c = 0; c < n_channels; ++c) {
            for (int t = 0; t < n_times; ++t) {
                white_noise(c, t) = dist(m_rng);
            }
        }
        
        // Apply covariance structure: X = L * white_noise
        MatrixXd epoch = L * white_noise;
        epochs.push_back(epoch);
    }
    
    return epochs;
}

MatrixXd TestCSPProperties::computeAverageCovariance(const std::vector<MatrixXd>& epochs)
{
    if (epochs.empty()) {
        return MatrixXd();
    }
    
    int n_channels = epochs[0].rows();
    MatrixXd avg_cov = MatrixXd::Zero(n_channels, n_channels);
    
    for (const auto& epoch : epochs) {
        MatrixXd centered = epoch.colwise() - epoch.rowwise().mean();
        MatrixXd cov = (centered * centered.transpose()) / (double)(epoch.cols() - 1);
        avg_cov += cov;
    }
    
    avg_cov /= (double)epochs.size();
    return avg_cov;
}

bool TestCSPProperties::verifyGeneralizedEigenvalueProblem(const MatrixXd& cov_a, const MatrixXd& cov_b,
                                                          const MatrixXd& filters, const VectorXd& eigenvalues,
                                                          double tolerance)
{
    MatrixXd cov_sum = cov_a + cov_b;
    int n_channels = cov_a.rows();
    
    // For each eigenvector, verify: C_a * w = lambda * (C_a + C_b) * w
    for (int i = 0; i < n_channels; ++i) {
        VectorXd w = filters.col(i);
        double lambda = eigenvalues(i);
        
        VectorXd lhs = cov_a * w;
        VectorXd rhs = lambda * (cov_sum * w);
        
        double error = (lhs - rhs).norm() / (lhs.norm() + 1e-12);
        
        if (error > tolerance) {
            qDebug() << "Eigenvalue problem verification failed for component" << i;
            qDebug() << "Error:" << error << "Tolerance:" << tolerance;
            return false;
        }
    }
    
    return true;
}

void TestCSPProperties::testCSPVarianceRatioMaximization()
{
    // Feature: mne-python-to-cpp-migration, Property 11: CSP方差比最大化
    // Run 100 iterations with random data
    const int n_iterations = 100;
    int passed = 0;
    
    for (int iter = 0; iter < n_iterations; ++iter) {
        // Generate random parameters
        std::uniform_int_distribution<int> channel_dist(2, 8);
        std::uniform_int_distribution<int> epoch_dist(10, 30);
        std::uniform_int_distribution<int> time_dist(50, 200);
        
        int n_channels = channel_dist(m_rng);
        int n_epochs_per_class = epoch_dist(m_rng);
        int n_times = time_dist(m_rng);
        int n_components = std::min(4, n_channels);
        
        // Generate random covariance structures for two classes
        // Class A: Random positive definite matrix
        MatrixXd A = MatrixXd::Random(n_channels, n_channels);
        MatrixXd cov_structure_a = A * A.transpose() + MatrixXd::Identity(n_channels, n_channels);
        
        // Class B: Different random positive definite matrix
        MatrixXd B = MatrixXd::Random(n_channels, n_channels);
        MatrixXd cov_structure_b = B * B.transpose() + MatrixXd::Identity(n_channels, n_channels);
        
        // Generate epochs for both classes
        std::vector<MatrixXd> epochs_a = generateRandomEpochs(n_epochs_per_class, n_channels, 
                                                              n_times, cov_structure_a);
        std::vector<MatrixXd> epochs_b = generateRandomEpochs(n_epochs_per_class, n_channels, 
                                                              n_times, cov_structure_b);
        
        // Combine epochs and labels
        std::vector<MatrixXd> all_epochs;
        std::vector<int> labels;
        
        for (const auto& epoch : epochs_a) {
            all_epochs.push_back(epoch);
            labels.push_back(0);
        }
        
        for (const auto& epoch : epochs_b) {
            all_epochs.push_back(epoch);
            labels.push_back(1);
        }
        
        // Fit CSP
        CSP csp(n_components, false, false); // No normalization, no log for property testing
        bool fit_success = csp.fit(all_epochs, labels);
        
        if (!fit_success) {
            qDebug() << "Iteration" << iter << ": CSP fit failed";
            continue;
        }
        
        // Get filters and eigenvalues
        MatrixXd filters = csp.getFilters();
        VectorXd eigenvalues = csp.getEigenValues();
        
        // Compute average covariance matrices for each class
        MatrixXd cov_a = computeAverageCovariance(epochs_a);
        MatrixXd cov_b = computeAverageCovariance(epochs_b);
        
        // Verify the generalized eigenvalue problem
        bool property_holds = verifyGeneralizedEigenvalueProblem(cov_a, cov_b, filters, 
                                                                 eigenvalues, 1e-4);
        
        if (property_holds) {
            passed++;
        } else {
            qDebug() << "Iteration" << iter << ": Property verification failed";
            qDebug() << "n_channels:" << n_channels << "n_epochs:" << n_epochs_per_class 
                     << "n_times:" << n_times;
        }
        
        // Additional check: Verify eigenvalues are in [0, 1]
        // For the generalized eigenvalue problem C_a * w = lambda * (C_a + C_b) * w,
        // eigenvalues should be in the range [0, 1]
        for (int i = 0; i < eigenvalues.size(); ++i) {
            double lambda = eigenvalues(i);
            QVERIFY2(lambda >= -1e-6 && lambda <= 1.0 + 1e-6, 
                    qPrintable(QString("Eigenvalue %1 out of range [0,1]: %2").arg(i).arg(lambda)));
        }
        
        // Verify filters are orthogonal with respect to (C_a + C_b)
        // W^T * (C_a + C_b) * W should be diagonal
        MatrixXd cov_sum = cov_a + cov_b;
        MatrixXd orthogonality = filters.transpose() * cov_sum * filters;
        
        // Check if off-diagonal elements are small
        for (int i = 0; i < orthogonality.rows(); ++i) {
            for (int j = 0; j < orthogonality.cols(); ++j) {
                if (i != j) {
                    double off_diag = std::abs(orthogonality(i, j));
                    double diag_scale = std::sqrt(orthogonality(i, i) * orthogonality(j, j));
                    if (diag_scale > 1e-12) {
                        double relative_error = off_diag / diag_scale;
                        QVERIFY2(relative_error < 1e-3,
                                qPrintable(QString("Filters not orthogonal: (%1,%2) = %3")
                                          .arg(i).arg(j).arg(relative_error)));
                    }
                }
            }
        }
    }
    
    qDebug() << "CSP Variance Ratio Maximization Property Test:";
    qDebug() << "Passed:" << passed << "/" << n_iterations << "iterations";
    
    // Require at least 95% success rate
    QVERIFY2(passed >= n_iterations * 0.95, 
            qPrintable(QString("Only %1/%2 iterations passed (< 95%)").arg(passed).arg(n_iterations)));
}

QTEST_GUILESS_MAIN(TestCSPProperties)
#include "test_csp_properties.moc"
