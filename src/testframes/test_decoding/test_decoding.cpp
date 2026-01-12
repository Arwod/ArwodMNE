#include <QtTest/QtTest>
#include <decoding/csp.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace DECODINGLIB;
using namespace Eigen;

class TestDecoding : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCSP();
};

void TestDecoding::initTestCase()
{
}

void TestDecoding::cleanupTestCase()
{
}

void TestDecoding::testCSP()
{
    // 1. Generate Simulated Data
    // 2 Classes, 20 Epochs each, 2 Channels, 100 Samples
    int n_epochs_per_class = 20;
    int n_channels = 2;
    int n_times = 100;
    
    std::vector<MatrixXd> epochs;
    std::vector<int> labels;
    
    // Class 0: Ch0 High Var, Ch1 Low Var
    for (int i = 0; i < n_epochs_per_class; ++i) {
        MatrixXd epoch(n_channels, n_times);
        epoch.row(0) = VectorXd::Random(n_times) * 5.0; // High var
        epoch.row(1) = VectorXd::Random(n_times) * 1.0; // Low var
        epochs.push_back(epoch);
        labels.push_back(0);
    }
    
    // Class 1: Ch0 Low Var, Ch1 High Var
    for (int i = 0; i < n_epochs_per_class; ++i) {
        MatrixXd epoch(n_channels, n_times);
        epoch.row(0) = VectorXd::Random(n_times) * 1.0; // Low var
        epoch.row(1) = VectorXd::Random(n_times) * 5.0; // High var
        epochs.push_back(epoch);
        labels.push_back(1);
    }
    
    // 2. Setup CSP
    // n_components = 2 (Keep all)
    CSP csp(2);
    
    // 3. Fit
    bool success = csp.fit(epochs, labels);
    QVERIFY(success);
    
    // Check Eigenvalues
    // Should be close to [0, 1] or [1, 0] roughly sum to 1?
    // MNE-Python eigh(cov_a, cov_a + cov_b) -> eigenvalues in [0, 1].
    // Since we have perfectly distinct variances, one lambda should be close to 1 (high ratio for A), 
    // one close to 0 (low ratio for A, high for B).
    // Or if we solve C_a w = lambda C_b w, then lambda can be anything.
    // Our implementation: es(cov_a, cov_a + cov_b) -> lambdas in [0, 1].
    
    VectorXd eigen_vals = csp.getEigenValues();
    qDebug() << "Eigenvalues:" << eigen_vals[0] << eigen_vals[1];
    
    QVERIFY(eigen_vals.size() == 2);
    // With our data, Class 0 has high var on Ch0.
    // Cov_0 ~ diag(25, 1), Cov_1 ~ diag(1, 25)
    // Cov_sum ~ diag(26, 26)
    // C_0 * w = lambda * C_sum * w
    // lambda_0 ~ 25/26 ~ 0.96
    // lambda_1 ~ 1/26 ~ 0.04
    // Eigen sorts ascending. So vals should be [0.04, 0.96].
    
    QVERIFY(std::abs(eigen_vals[0] - 0.04) < 0.1);
    QVERIFY(std::abs(eigen_vals[1] - 0.96) < 0.1);
    
    // 4. Transform
    MatrixXd X_transformed = csp.transform(epochs);
    
    QVERIFY(X_transformed.rows() == 40);
    QVERIFY(X_transformed.cols() == 2);
    
    // Check discrimination
    // Transform order: Top (Largest Lambda) then Bottom (Smallest Lambda)
    
    // Feature 0 (corresponding to lambda ~ 0.96, Class 0 dominant)
    // Should be High for Class 0, Low for Class 1.
    
    double mean_f0_c0 = X_transformed.block(0, 0, 20, 1).mean();
    double mean_f0_c1 = X_transformed.block(20, 0, 20, 1).mean();
    
    qDebug() << "Feature 0 Mean (Class 0):" << mean_f0_c0;
    qDebug() << "Feature 0 Mean (Class 1):" << mean_f0_c1;
    
    // Since we use Log variance:
    // Var(Ch0) ~ 25 -> log(25) ~ 3.2 (If normalized, it's different)
    // The values we saw were -0.04 vs -3.22.
    // -0.04 is higher than -3.22.
    
    QVERIFY(mean_f0_c0 > mean_f0_c1); // Feature 0 dominates Class 0
    
    // Feature 1 (corresponding to lambda ~ 0.04, Class 1 dominant)
    // Should be Low for Class 0, High for Class 1.
    
    double mean_f1_c0 = X_transformed.block(0, 1, 20, 1).mean();
    double mean_f1_c1 = X_transformed.block(20, 1, 20, 1).mean();
    
    qDebug() << "Feature 1 Mean (Class 0):" << mean_f1_c0;
    qDebug() << "Feature 1 Mean (Class 1):" << mean_f1_c1;
    
    QVERIFY(mean_f1_c1 > mean_f1_c0); // Feature 1 dominates Class 1
}

QTEST_GUILESS_MAIN(TestDecoding)
#include "test_decoding.moc"
