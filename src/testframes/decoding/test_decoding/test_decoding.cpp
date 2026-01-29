#include <QtTest/QtTest>
#include <decoding/csp.h>
#include <decoding/temporal_decoding.h>
#include <decoding/spatial_filtering.h>
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
    
    // Time decoder tests
    void testSlidingEstimatorFit();
    void testSlidingEstimatorPredict();
    void testSlidingEstimatorScore();
    void testSlidingEstimatorConsistency();
    void testGeneralizingEstimatorFit();
    void testGeneralizingEstimatorScore();
    void testGeneralizingEstimatorDiagonal();
    void testTemporalGeneralizationMatrix();
    
    // Spatial filter tests
    void testPCAFilterFit();
    void testPCAFilterTransform();
    void testPCAFilterDimensionReduction();
    void testPCAFilterExplainedVariance();
    void testWhiteningFilterFit();
    void testWhiteningFilterCovariance();
    void testLaplacianFilterFit();
    void testXdawnFilterFit();
    void testXdawnFilterTransform();
    void testXdawnFilterEvokedResponse();
    void testSurfaceLaplacianFilterFit();
    void testSpatialFilterConsistency();
    
private:
    // Helper function to generate synthetic temporal data
    std::vector<MatrixXd> generateTemporalData(int n_epochs, int n_channels, 
                                                int n_times, int label,
                                                double signal_strength = 1.0);
    
    // Helper function to generate synthetic spatial data
    std::vector<MatrixXd> generateSpatialData(int n_epochs, int n_channels, 
                                              int n_times, double noise_level = 1.0);
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

//=============================================================================================================
// Helper function to generate synthetic temporal data
//=============================================================================================================

std::vector<MatrixXd> TestDecoding::generateTemporalData(int n_epochs, int n_channels, 
                                                          int n_times, int label,
                                                          double signal_strength)
{
    std::vector<MatrixXd> epochs;
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd epoch(n_channels, n_times);
        
        // Generate time-varying signal with class-specific pattern
        for (int t = 0; t < n_times; ++t) {
            for (int ch = 0; ch < n_channels; ++ch) {
                // Base noise
                double noise = (double)rand() / RAND_MAX - 0.5;
                
                // Add class-specific temporal pattern
                // Class 0: Signal increases over time
                // Class 1: Signal decreases over time
                double temporal_pattern = 0.0;
                if (label == 0) {
                    temporal_pattern = signal_strength * (double)t / n_times;
                } else {
                    temporal_pattern = signal_strength * (1.0 - (double)t / n_times);
                }
                
                epoch(ch, t) = noise + temporal_pattern;
            }
        }
        
        epochs.push_back(epoch);
    }
    
    return epochs;
}

//=============================================================================================================
// Test SlidingEstimator Fit
//=============================================================================================================

void TestDecoding::testSlidingEstimatorFit()
{
    // Generate synthetic data with temporal patterns
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    // Class 0
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    // Class 1
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    // Create and fit SlidingEstimator
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator);
    
    bool success = sliding_est.fit(epochs, labels);
    QVERIFY(success);
    
    // Verify estimators were created for each time point
    auto estimators = sliding_est.getEstimators();
    QVERIFY(estimators.size() == n_times);
    
    qDebug() << "SlidingEstimator fit test passed: Created" << estimators.size() << "estimators";
}

//=============================================================================================================
// Test SlidingEstimator Predict
//=============================================================================================================

void TestDecoding::testSlidingEstimatorPredict()
{
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator);
    sliding_est.fit(epochs, labels);
    
    // Predict on training data
    MatrixXi predictions = sliding_est.predict(epochs);
    
    // Verify prediction dimensions
    QVERIFY(predictions.rows() == n_epochs_per_class * 2);
    QVERIFY(predictions.cols() == n_times);
    
    // Check that predictions are valid labels (0 or 1)
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int t = 0; t < predictions.cols(); ++t) {
            QVERIFY(predictions(i, t) == 0 || predictions(i, t) == 1);
        }
    }
    
    qDebug() << "SlidingEstimator predict test passed: Predictions shape" 
             << predictions.rows() << "x" << predictions.cols();
}

//=============================================================================================================
// Test SlidingEstimator Score
//=============================================================================================================

void TestDecoding::testSlidingEstimatorScore()
{
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator);
    sliding_est.fit(epochs, labels);
    
    // Score on training data
    VectorXd scores = sliding_est.score(epochs, labels);
    
    // Verify score dimensions
    QVERIFY(scores.size() == n_times);
    
    // Scores should be between 0 and 1 (accuracy)
    for (int t = 0; t < scores.size(); ++t) {
        QVERIFY(scores(t) >= 0.0 && scores(t) <= 1.0);
    }
    
    // With strong temporal patterns, accuracy should be reasonable
    double mean_accuracy = scores.mean();
    QVERIFY(mean_accuracy > 0.5); // Better than chance
    
    qDebug() << "SlidingEstimator score test passed: Mean accuracy" << mean_accuracy;
}

//=============================================================================================================
// Test SlidingEstimator Sliding Window Consistency
//=============================================================================================================

void TestDecoding::testSlidingEstimatorConsistency()
{
    // Test that each time point is decoded independently
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator);
    sliding_est.fit(epochs, labels);
    
    // Get predictions for full data
    MatrixXi predictions_full = sliding_est.predict(epochs);
    
    // Manually predict at a specific time point and compare
    int test_time = 10;
    auto estimators = sliding_est.getEstimators();
    
    // Extract data at test_time
    MatrixXd X_t(epochs.size(), n_channels);
    for (size_t i = 0; i < epochs.size(); ++i) {
        X_t.row(i) = epochs[i].col(test_time).transpose();
    }
    
    VectorXi predictions_t = estimators[test_time]->predict(X_t);
    
    // Compare with sliding estimator predictions at that time
    for (int i = 0; i < predictions_t.size(); ++i) {
        QVERIFY(predictions_t(i) == predictions_full(i, test_time));
    }
    
    qDebug() << "SlidingEstimator consistency test passed: Independent time point decoding verified";
}

//=============================================================================================================
// Test GeneralizingEstimator Fit
//=============================================================================================================

void TestDecoding::testGeneralizingEstimatorFit()
{
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    // Create and fit GeneralizingEstimator
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    GeneralizingEstimator gen_est(base_estimator);
    
    bool success = gen_est.fit(epochs, labels);
    QVERIFY(success);
    
    // Verify estimators were created for each time point
    auto estimators = gen_est.getEstimators();
    QVERIFY(estimators.size() == n_times);
    
    qDebug() << "GeneralizingEstimator fit test passed: Created" << estimators.size() << "estimators";
}

//=============================================================================================================
// Test GeneralizingEstimator Score (Temporal Generalization Matrix)
//=============================================================================================================

void TestDecoding::testGeneralizingEstimatorScore()
{
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    GeneralizingEstimator gen_est(base_estimator);
    gen_est.fit(epochs, labels);
    
    // Compute temporal generalization matrix
    MatrixXd gen_matrix = gen_est.score(epochs, labels);
    
    // Verify dimensions: train_time x test_time
    QVERIFY(gen_matrix.rows() == n_times);
    QVERIFY(gen_matrix.cols() == n_times);
    
    // All scores should be between 0 and 1
    for (int i = 0; i < gen_matrix.rows(); ++i) {
        for (int j = 0; j < gen_matrix.cols(); ++j) {
            QVERIFY(gen_matrix(i, j) >= 0.0 && gen_matrix(i, j) <= 1.0);
        }
    }
    
    qDebug() << "GeneralizingEstimator score test passed: Generalization matrix shape" 
             << gen_matrix.rows() << "x" << gen_matrix.cols();
}

//=============================================================================================================
// Test GeneralizingEstimator Diagonal Property
//=============================================================================================================

void TestDecoding::testGeneralizingEstimatorDiagonal()
{
    // The diagonal of the generalization matrix should match SlidingEstimator scores
    int n_epochs_per_class = 30;
    int n_channels = 5;
    int n_times = 20;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 2.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 2.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    // Fit both estimators
    auto base_estimator1 = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator1);
    sliding_est.fit(epochs, labels);
    
    auto base_estimator2 = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    GeneralizingEstimator gen_est(base_estimator2);
    gen_est.fit(epochs, labels);
    
    // Get scores
    VectorXd sliding_scores = sliding_est.score(epochs, labels);
    MatrixXd gen_matrix = gen_est.score(epochs, labels);
    
    // Extract diagonal from generalization matrix
    VectorXd gen_diagonal = gen_matrix.diagonal();
    
    // Compare diagonal with sliding estimator scores
    double max_diff = (sliding_scores - gen_diagonal).cwiseAbs().maxCoeff();
    QVERIFY(max_diff < 1e-10); // Should be essentially identical
    
    qDebug() << "GeneralizingEstimator diagonal test passed: Max difference" << max_diff;
}

//=============================================================================================================
// Test Temporal Generalization Matrix Properties
//=============================================================================================================

void TestDecoding::testTemporalGeneralizationMatrix()
{
    // Test that generalization matrix shows expected temporal structure
    int n_epochs_per_class = 40;
    int n_channels = 5;
    int n_times = 15;
    
    std::vector<MatrixXd> epochs;
    VectorXi labels(n_epochs_per_class * 2);
    
    // Create data with strong temporal structure
    auto epochs_c0 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 0, 3.0);
    epochs.insert(epochs.end(), epochs_c0.begin(), epochs_c0.end());
    labels.head(n_epochs_per_class).setConstant(0);
    
    auto epochs_c1 = generateTemporalData(n_epochs_per_class, n_channels, n_times, 1, 3.0);
    epochs.insert(epochs.end(), epochs_c1.begin(), epochs_c1.end());
    labels.tail(n_epochs_per_class).setConstant(1);
    
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    GeneralizingEstimator gen_est(base_estimator);
    gen_est.fit(epochs, labels);
    
    MatrixXd gen_matrix = gen_est.score(epochs, labels);
    
    // Diagonal should have highest scores (train and test at same time)
    VectorXd diagonal = gen_matrix.diagonal();
    double mean_diagonal = diagonal.mean();
    
    // Off-diagonal elements should generally be lower
    double sum_off_diagonal = 0.0;
    int count_off_diagonal = 0;
    for (int i = 0; i < gen_matrix.rows(); ++i) {
        for (int j = 0; j < gen_matrix.cols(); ++j) {
            if (i != j) {
                sum_off_diagonal += gen_matrix(i, j);
                count_off_diagonal++;
            }
        }
    }
    double mean_off_diagonal = sum_off_diagonal / count_off_diagonal;
    
    // Diagonal should be at least as good as off-diagonal on average
    QVERIFY(mean_diagonal >= mean_off_diagonal);
    
    qDebug() << "Temporal generalization test passed:";
    qDebug() << "  Mean diagonal accuracy:" << mean_diagonal;
    qDebug() << "  Mean off-diagonal accuracy:" << mean_off_diagonal;
}

//=============================================================================================================
// Helper function to generate synthetic spatial data
//=============================================================================================================

std::vector<MatrixXd> TestDecoding::generateSpatialData(int n_epochs, int n_channels, 
                                                         int n_times, double noise_level)
{
    std::vector<MatrixXd> epochs;
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd epoch(n_channels, n_times);
        
        // Generate random data with specified noise level
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                epoch(ch, t) = noise_level * ((double)rand() / RAND_MAX - 0.5);
            }
        }
        
        epochs.push_back(epoch);
    }
    
    return epochs;
}

//=============================================================================================================
// Test PCA Filter Fit
//=============================================================================================================

void TestDecoding::testPCAFilterFit()
{
    // Generate synthetic data
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    // Create and fit PCA filter
    UnsupervisedSpatialFilter pca_filter(UnsupervisedSpatialFilter::Method::PCA, 10);
    bool success = pca_filter.fit(epochs);
    
    QVERIFY(success);
    
    // Verify filter dimensions
    MatrixXd filters = pca_filter.getFilters();
    QVERIFY(filters.rows() == n_channels);
    QVERIFY(filters.cols() == 10);
    
    qDebug() << "PCA filter fit test passed: Filter shape" << filters.rows() << "x" << filters.cols();
}

//=============================================================================================================
// Test PCA Filter Transform
//=============================================================================================================

void TestDecoding::testPCAFilterTransform()
{
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    int n_components = 10;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    UnsupervisedSpatialFilter pca_filter(UnsupervisedSpatialFilter::Method::PCA, n_components);
    pca_filter.fit(epochs);
    
    // Transform data
    std::vector<MatrixXd> filtered_epochs = pca_filter.transform(epochs);
    
    // Verify output dimensions
    QVERIFY(filtered_epochs.size() == n_epochs);
    QVERIFY(filtered_epochs[0].rows() == n_components);
    QVERIFY(filtered_epochs[0].cols() == n_times);
    
    qDebug() << "PCA filter transform test passed: Output shape" 
             << filtered_epochs[0].rows() << "x" << filtered_epochs[0].cols();
}

//=============================================================================================================
// Test PCA Filter Dimension Reduction
//=============================================================================================================

void TestDecoding::testPCAFilterDimensionReduction()
{
    // Test that PCA correctly reduces dimensions
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    int n_components = 5;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    UnsupervisedSpatialFilter pca_filter(UnsupervisedSpatialFilter::Method::PCA, n_components);
    pca_filter.fit(epochs);
    
    std::vector<MatrixXd> filtered_epochs = pca_filter.transform(epochs);
    
    // Verify dimension reduction
    QVERIFY(filtered_epochs[0].rows() < epochs[0].rows());
    QVERIFY(filtered_epochs[0].rows() == n_components);
    
    // Verify time dimension is preserved
    QVERIFY(filtered_epochs[0].cols() == epochs[0].cols());
    
    qDebug() << "PCA dimension reduction test passed: Reduced from" 
             << n_channels << "to" << n_components << "components";
}

//=============================================================================================================
// Test PCA Filter Explained Variance
//=============================================================================================================

void TestDecoding::testPCAFilterExplainedVariance()
{
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    int n_components = 10;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    UnsupervisedSpatialFilter pca_filter(UnsupervisedSpatialFilter::Method::PCA, n_components);
    pca_filter.fit(epochs);
    
    // Get explained variance
    VectorXd explained_var = pca_filter.getExplainedVarianceRatio();
    
    // Verify explained variance properties
    QVERIFY(explained_var.size() == n_components);
    
    // All values should be positive
    for (int i = 0; i < explained_var.size(); ++i) {
        QVERIFY(explained_var(i) > 0.0);
    }
    
    // Explained variance should be in descending order
    for (int i = 0; i < explained_var.size() - 1; ++i) {
        QVERIFY(explained_var(i) >= explained_var(i + 1));
    }
    
    // Sum should be less than or equal to 1.0
    double total_var = explained_var.sum();
    QVERIFY(total_var <= 1.0);
    
    qDebug() << "PCA explained variance test passed: Total variance explained" << total_var;
}

//=============================================================================================================
// Test Whitening Filter Fit
//=============================================================================================================

void TestDecoding::testWhiteningFilterFit()
{
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    // Create and fit whitening filter
    UnsupervisedSpatialFilter whitening_filter(UnsupervisedSpatialFilter::Method::Whitening, 
                                               n_channels, 0.01);
    bool success = whitening_filter.fit(epochs);
    
    QVERIFY(success);
    
    // Verify filter dimensions
    MatrixXd filters = whitening_filter.getFilters();
    QVERIFY(filters.rows() == n_channels);
    QVERIFY(filters.cols() == n_channels);
    
    qDebug() << "Whitening filter fit test passed: Filter shape" 
             << filters.rows() << "x" << filters.cols();
}

//=============================================================================================================
// Test Whitening Filter Covariance
//=============================================================================================================

void TestDecoding::testWhiteningFilterCovariance()
{
    // Test that whitening produces approximately identity covariance
    int n_epochs = 100;
    int n_channels = 10;
    int n_times = 200;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    UnsupervisedSpatialFilter whitening_filter(UnsupervisedSpatialFilter::Method::Whitening, 
                                               n_channels, 0.01);
    whitening_filter.fit(epochs);
    
    // Transform data
    std::vector<MatrixXd> filtered_epochs = whitening_filter.transform(epochs);
    
    // Compute covariance of filtered data
    MatrixXd cov = MatrixXd::Zero(n_channels, n_channels);
    int total_samples = 0;
    
    for (const auto& epoch : filtered_epochs) {
        VectorXd mean = epoch.rowwise().mean();
        MatrixXd centered = epoch.colwise() - mean;
        cov += centered * centered.transpose();
        total_samples += epoch.cols();
    }
    cov /= (total_samples - 1);
    
    // Check that covariance is approximately identity
    MatrixXd identity = MatrixXd::Identity(n_channels, n_channels);
    double max_diff = (cov - identity).cwiseAbs().maxCoeff();
    
    // Allow some tolerance due to regularization and numerical errors
    QVERIFY(max_diff < 0.5);
    
    qDebug() << "Whitening covariance test passed: Max deviation from identity" << max_diff;
}

//=============================================================================================================
// Test Laplacian Filter Fit
//=============================================================================================================

void TestDecoding::testLaplacianFilterFit()
{
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    // Create and fit Laplacian filter
    UnsupervisedSpatialFilter laplacian_filter(UnsupervisedSpatialFilter::Method::Laplacian);
    bool success = laplacian_filter.fit(epochs);
    
    QVERIFY(success);
    
    // Verify filter dimensions
    MatrixXd filters = laplacian_filter.getFilters();
    QVERIFY(filters.rows() == n_channels);
    QVERIFY(filters.cols() == n_channels);
    
    // Transform data
    std::vector<MatrixXd> filtered_epochs = laplacian_filter.transform(epochs);
    QVERIFY(filtered_epochs.size() == n_epochs);
    QVERIFY(filtered_epochs[0].rows() == n_channels);
    
    qDebug() << "Laplacian filter fit test passed";
}

//=============================================================================================================
// Test Xdawn Filter Fit
//=============================================================================================================

void TestDecoding::testXdawnFilterFit()
{
    int n_epochs = 60;
    int n_channels = 15;
    int n_times = 100;
    int n_components = 4;
    
    // Generate epochs with labels
    std::vector<MatrixXd> epochs;
    std::vector<int> labels;
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd epoch(n_channels, n_times);
        
        int label = i % 2;
        labels.push_back(label);
        
        // Generate data with class-specific patterns
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                double noise = ((double)rand() / RAND_MAX - 0.5);
                
                // Add ERP-like signal for target class
                if (label == 1 && t >= 30 && t <= 50) {
                    epoch(ch, t) = noise + 2.0 * std::sin(2.0 * M_PI * (t - 30) / 20.0);
                } else {
                    epoch(ch, t) = noise;
                }
            }
        }
        
        epochs.push_back(epoch);
    }
    
    // Create and fit Xdawn filter
    XdawnSpatialFilter xdawn_filter(n_components, 0.01);
    bool success = xdawn_filter.fit(epochs, labels);
    
    QVERIFY(success);
    
    // Verify filter dimensions
    MatrixXd filters = xdawn_filter.getFilters();
    QVERIFY(filters.rows() == n_channels);
    QVERIFY(filters.cols() == n_components);
    
    // Verify patterns dimensions
    MatrixXd patterns = xdawn_filter.getPatterns();
    QVERIFY(patterns.rows() == n_channels);
    QVERIFY(patterns.cols() == n_components);
    
    qDebug() << "Xdawn filter fit test passed: Filter shape" 
             << filters.rows() << "x" << filters.cols();
}

//=============================================================================================================
// Test Xdawn Filter Transform
//=============================================================================================================

void TestDecoding::testXdawnFilterTransform()
{
    int n_epochs = 60;
    int n_channels = 15;
    int n_times = 100;
    int n_components = 4;
    
    std::vector<MatrixXd> epochs;
    std::vector<int> labels;
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd epoch(n_channels, n_times);
        int label = i % 2;
        labels.push_back(label);
        
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                double noise = ((double)rand() / RAND_MAX - 0.5);
                if (label == 1 && t >= 30 && t <= 50) {
                    epoch(ch, t) = noise + 2.0 * std::sin(2.0 * M_PI * (t - 30) / 20.0);
                } else {
                    epoch(ch, t) = noise;
                }
            }
        }
        epochs.push_back(epoch);
    }
    
    XdawnSpatialFilter xdawn_filter(n_components, 0.01);
    xdawn_filter.fit(epochs, labels);
    
    // Transform data
    std::vector<MatrixXd> filtered_epochs = xdawn_filter.transform(epochs);
    
    // Verify output dimensions
    QVERIFY(filtered_epochs.size() == n_epochs);
    QVERIFY(filtered_epochs[0].rows() == n_components);
    QVERIFY(filtered_epochs[0].cols() == n_times);
    
    qDebug() << "Xdawn filter transform test passed: Output shape" 
             << filtered_epochs[0].rows() << "x" << filtered_epochs[0].cols();
}

//=============================================================================================================
// Test Xdawn Filter Evoked Response
//=============================================================================================================

void TestDecoding::testXdawnFilterEvokedResponse()
{
    int n_epochs = 60;
    int n_channels = 15;
    int n_times = 100;
    int n_components = 4;
    
    std::vector<MatrixXd> epochs;
    std::vector<int> labels;
    
    for (int i = 0; i < n_epochs; ++i) {
        MatrixXd epoch(n_channels, n_times);
        int label = i % 2;
        labels.push_back(label);
        
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                double noise = ((double)rand() / RAND_MAX - 0.5);
                if (label == 1 && t >= 30 && t <= 50) {
                    epoch(ch, t) = noise + 2.0 * std::sin(2.0 * M_PI * (t - 30) / 20.0);
                } else {
                    epoch(ch, t) = noise;
                }
            }
        }
        epochs.push_back(epoch);
    }
    
    XdawnSpatialFilter xdawn_filter(n_components, 0.01);
    xdawn_filter.fit(epochs, labels);
    
    // Get evoked response
    MatrixXd evoked = xdawn_filter.getEvokedResponse();
    
    // Verify evoked response dimensions
    QVERIFY(evoked.rows() == n_channels);
    QVERIFY(evoked.cols() == n_times);
    
    // Evoked response should have higher amplitude in the signal region (30-50)
    double signal_power = 0.0;
    double noise_power = 0.0;
    
    for (int ch = 0; ch < n_channels; ++ch) {
        for (int t = 30; t <= 50; ++t) {
            signal_power += evoked(ch, t) * evoked(ch, t);
        }
        for (int t = 0; t < 30; ++t) {
            noise_power += evoked(ch, t) * evoked(ch, t);
        }
    }
    
    signal_power /= (n_channels * 21);
    noise_power /= (n_channels * 30);
    
    // Signal region should have higher power than noise region
    QVERIFY(signal_power > noise_power);
    
    qDebug() << "Xdawn evoked response test passed: Signal power" << signal_power 
             << "vs noise power" << noise_power;
}

//=============================================================================================================
// Test Surface Laplacian Filter Fit
//=============================================================================================================

void TestDecoding::testSurfaceLaplacianFilterFit()
{
    int n_epochs = 50;
    int n_channels = 10;
    int n_times = 100;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    // Create surface Laplacian filter
    SurfaceLaplacianFilter surf_laplacian(4, 1e-5);
    
    // Set electrode positions (simple grid on unit sphere)
    MatrixXd positions(n_channels, 3);
    for (int i = 0; i < n_channels; ++i) {
        double theta = 2.0 * M_PI * i / n_channels;
        positions(i, 0) = std::cos(theta);
        positions(i, 1) = std::sin(theta);
        positions(i, 2) = 0.0;
    }
    surf_laplacian.setElectrodePositions(positions);
    
    // Fit filter
    bool success = surf_laplacian.fit(epochs);
    QVERIFY(success);
    
    // Verify filter dimensions
    MatrixXd filters = surf_laplacian.getFilters();
    QVERIFY(filters.rows() == n_channels);
    QVERIFY(filters.cols() == n_channels);
    
    // Transform data
    std::vector<MatrixXd> filtered_epochs = surf_laplacian.transform(epochs);
    QVERIFY(filtered_epochs.size() == n_epochs);
    QVERIFY(filtered_epochs[0].rows() == n_channels);
    
    qDebug() << "Surface Laplacian filter fit test passed";
}

//=============================================================================================================
// Test Spatial Filter Consistency
//=============================================================================================================

void TestDecoding::testSpatialFilterConsistency()
{
    // Test that fitTransform produces same result as fit + transform
    int n_epochs = 50;
    int n_channels = 20;
    int n_times = 100;
    int n_components = 10;
    
    auto epochs = generateSpatialData(n_epochs, n_channels, n_times, 1.0);
    
    // Method 1: fit then transform
    UnsupervisedSpatialFilter filter1(UnsupervisedSpatialFilter::Method::PCA, n_components);
    filter1.fit(epochs);
    std::vector<MatrixXd> result1 = filter1.transform(epochs);
    
    // Method 2: fitTransform
    UnsupervisedSpatialFilter filter2(UnsupervisedSpatialFilter::Method::PCA, n_components);
    std::vector<MatrixXd> result2 = filter2.fitTransform(epochs);
    
    // Compare results
    QVERIFY(result1.size() == result2.size());
    
    for (size_t i = 0; i < result1.size(); ++i) {
        double max_diff = (result1[i] - result2[i]).cwiseAbs().maxCoeff();
        QVERIFY(max_diff < 1e-10);
    }
    
    qDebug() << "Spatial filter consistency test passed: fitTransform matches fit + transform";
}

QTEST_GUILESS_MAIN(TestDecoding)
#include "test_decoding.moc"
