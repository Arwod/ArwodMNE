#include "temporal_decoding.h"
#include <Eigen/LU>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>

namespace DECODINGLIB
{

//=============================================================================================================
// BaseTemporalDecoder Implementation
//=============================================================================================================

double BaseTemporalDecoder::score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const
{
    Eigen::VectorXi predictions = predict(X);
    
    if (predictions.size() != y.size()) {
        return 0.0;
    }
    
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) {
        if (predictions(i) == y(i)) {
            correct++;
        }
    }
    
    return (double)correct / (double)y.size();
}

//=============================================================================================================
// SlidingEstimator Implementation
//=============================================================================================================

SlidingEstimator::SlidingEstimator(std::shared_ptr<BaseTemporalDecoder> base_estimator,
                                  int n_jobs,
                                  const std::string& scoring)
: m_baseEstimator(base_estimator)
, m_nJobs(n_jobs)
, m_scoring(scoring)
, m_nTimes(0)
{
}

//=============================================================================================================

bool SlidingEstimator::fit(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y)
{
    if (X.empty() || X.size() != y.size()) {
        std::cerr << "SlidingEstimator::fit: Empty data or size mismatch." << std::endl;
        return false;
    }
    
    m_nTimes = X[0].cols(); // Assuming X[i] is (n_channels x n_times)
    int n_epochs = X.size();
    int n_channels = X[0].rows();
    
    // Verify all epochs have same dimensions
    for (const auto& epoch : X) {
        if (epoch.rows() != n_channels || epoch.cols() != m_nTimes) {
            std::cerr << "SlidingEstimator::fit: Inconsistent epoch dimensions." << std::endl;
            return false;
        }
    }
    
    m_estimators.clear();
    m_estimators.reserve(m_nTimes);
    
    // Fit estimator for each time point
    for (int t = 0; t < m_nTimes; ++t) {
        // Extract data at time point t
        Eigen::MatrixXd X_t(n_epochs, n_channels);
        for (int i = 0; i < n_epochs; ++i) {
            X_t.row(i) = X[i].col(t).transpose();
        }
        
        // Create and fit estimator for this time point
        auto estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
        if (!estimator->fit(X_t, y)) {
            std::cerr << "SlidingEstimator::fit: Failed to fit estimator at time " << t << std::endl;
            return false;
        }
        
        m_estimators.push_back(estimator);
    }
    
    return true;
}

//=============================================================================================================

Eigen::MatrixXi SlidingEstimator::predict(const std::vector<Eigen::MatrixXd>& X) const
{
    if (X.empty() || m_estimators.empty()) {
        return Eigen::MatrixXi();
    }
    
    int n_epochs = X.size();
    int n_channels = X[0].rows();
    
    Eigen::MatrixXi predictions(n_epochs, m_nTimes);
    
    for (int t = 0; t < m_nTimes; ++t) {
        // Extract data at time point t
        Eigen::MatrixXd X_t(n_epochs, n_channels);
        for (int i = 0; i < n_epochs; ++i) {
            X_t.row(i) = X[i].col(t).transpose();
        }
        
        // Predict using estimator for this time point
        Eigen::VectorXi pred_t = m_estimators[t]->predict(X_t);
        predictions.col(t) = pred_t;
    }
    
    return predictions;
}

//=============================================================================================================

Eigen::VectorXd SlidingEstimator::score(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y) const
{
    if (X.empty() || m_estimators.empty()) {
        return Eigen::VectorXd();
    }
    
    int n_epochs = X.size();
    int n_channels = X[0].rows();
    
    Eigen::VectorXd scores(m_nTimes);
    
    for (int t = 0; t < m_nTimes; ++t) {
        // Extract data at time point t
        Eigen::MatrixXd X_t(n_epochs, n_channels);
        for (int i = 0; i < n_epochs; ++i) {
            X_t.row(i) = X[i].col(t).transpose();
        }
        
        // Score using estimator for this time point
        scores(t) = m_estimators[t]->score(X_t, y);
    }
    
    return scores;
}

//=============================================================================================================
// GeneralizingEstimator Implementation
//=============================================================================================================

GeneralizingEstimator::GeneralizingEstimator(std::shared_ptr<BaseTemporalDecoder> base_estimator,
                                            int n_jobs,
                                            const std::string& scoring)
: m_baseEstimator(base_estimator)
, m_nJobs(n_jobs)
, m_scoring(scoring)
, m_nTimes(0)
{
}

//=============================================================================================================

bool GeneralizingEstimator::fit(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y)
{
    if (X.empty() || X.size() != y.size()) {
        std::cerr << "GeneralizingEstimator::fit: Empty data or size mismatch." << std::endl;
        return false;
    }
    
    m_nTimes = X[0].cols();
    int n_epochs = X.size();
    int n_channels = X[0].rows();
    
    m_estimators.clear();
    m_estimators.reserve(m_nTimes);
    
    // Fit estimator for each time point
    for (int t = 0; t < m_nTimes; ++t) {
        // Extract data at time point t
        Eigen::MatrixXd X_t(n_epochs, n_channels);
        for (int i = 0; i < n_epochs; ++i) {
            X_t.row(i) = X[i].col(t).transpose();
        }
        
        // Create and fit estimator for this time point
        auto estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
        if (!estimator->fit(X_t, y)) {
            std::cerr << "GeneralizingEstimator::fit: Failed to fit estimator at time " << t << std::endl;
            return false;
        }
        
        m_estimators.push_back(estimator);
    }
    
    return true;
}

//=============================================================================================================

Eigen::MatrixXd GeneralizingEstimator::score(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y) const
{
    if (X.empty() || m_estimators.empty()) {
        return Eigen::MatrixXd();
    }
    
    int n_epochs = X.size();
    int n_channels = X[0].rows();
    int n_test_times = X[0].cols();
    
    // Generalization matrix: train_time x test_time
    Eigen::MatrixXd gen_matrix(m_nTimes, n_test_times);
    
    for (int train_t = 0; train_t < m_nTimes; ++train_t) {
        for (int test_t = 0; test_t < n_test_times; ++test_t) {
            // Extract test data at time point test_t
            Eigen::MatrixXd X_test(n_epochs, n_channels);
            for (int i = 0; i < n_epochs; ++i) {
                X_test.row(i) = X[i].col(test_t).transpose();
            }
            
            // Score using estimator trained at train_t, tested at test_t
            gen_matrix(train_t, test_t) = m_estimators[train_t]->score(X_test, y);
        }
    }
    
    return gen_matrix;
}

//=============================================================================================================
// LinearDiscriminantAnalysis Implementation
//=============================================================================================================

LinearDiscriminantAnalysis::LinearDiscriminantAnalysis(double reg_param)
: m_dRegParam(reg_param)
, m_bFitted(false)
{
}

//=============================================================================================================

bool LinearDiscriminantAnalysis::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y)
{
    if (X.rows() != y.size()) {
        std::cerr << "LDA::fit: Size mismatch between X and y." << std::endl;
        return false;
    }
    
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Find unique classes
    std::vector<int> classes_vec(y.data(), y.data() + y.size());
    std::sort(classes_vec.begin(), classes_vec.end());
    classes_vec.erase(std::unique(classes_vec.begin(), classes_vec.end()), classes_vec.end());
    
    int n_classes = classes_vec.size();
    m_vecClasses = Eigen::VectorXi::Map(classes_vec.data(), n_classes);
    
    // Compute class means and priors
    std::map<int, Eigen::VectorXd> class_means;
    std::map<int, int> class_counts;
    
    for (int c : classes_vec) {
        class_means[c] = Eigen::VectorXd::Zero(n_features);
        class_counts[c] = 0;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        int label = y(i);
        class_means[label] += X.row(i).transpose();
        class_counts[label]++;
    }
    
    m_vecPriors = Eigen::VectorXd(n_classes);
    for (int i = 0; i < n_classes; ++i) {
        int c = m_vecClasses(i);
        class_means[c] /= class_counts[c];
        m_vecPriors(i) = (double)class_counts[c] / (double)n_samples;
    }
    
    // Compute pooled covariance matrix
    Eigen::MatrixXd cov_pooled = Eigen::MatrixXd::Zero(n_features, n_features);
    
    for (int i = 0; i < n_samples; ++i) {
        int label = y(i);
        Eigen::VectorXd diff = X.row(i).transpose() - class_means[label];
        cov_pooled += diff * diff.transpose();
    }
    
    cov_pooled /= (n_samples - n_classes);
    
    // Add regularization
    if (m_dRegParam > 0.0) {
        double trace = cov_pooled.trace();
        cov_pooled += m_dRegParam * trace / n_features * Eigen::MatrixXd::Identity(n_features, n_features);
    }
    
    // Compute weights and bias for each class
    Eigen::MatrixXd cov_inv;
    try {
        cov_inv = cov_pooled.inverse();
    } catch (const std::exception& e) {
        std::cerr << "LDA::fit: Failed to invert covariance matrix: " << e.what() << std::endl;
        return false;
    }
    
    m_matW = Eigen::MatrixXd(n_features, n_classes);
    m_vecBias = Eigen::VectorXd(n_classes);
    
    for (int i = 0; i < n_classes; ++i) {
        int c = m_vecClasses(i);
        Eigen::VectorXd mean_c = class_means[c];
        
        m_matW.col(i) = cov_inv * mean_c;
        m_vecBias(i) = -0.5 * mean_c.transpose() * cov_inv * mean_c + std::log(m_vecPriors(i));
    }
    
    m_bFitted = true;
    return true;
}

//=============================================================================================================

Eigen::VectorXi LinearDiscriminantAnalysis::predict(const Eigen::MatrixXd& X) const
{
    if (!m_bFitted) {
        std::cerr << "LDA::predict: Model not fitted." << std::endl;
        return Eigen::VectorXi();
    }
    
    int n_samples = X.rows();
    Eigen::VectorXi predictions(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        Eigen::VectorXd scores = X.row(i) * m_matW + m_vecBias.transpose();
        
        int best_class_idx;
        scores.maxCoeff(&best_class_idx);
        predictions(i) = m_vecClasses(best_class_idx);
    }
    
    return predictions;
}

//=============================================================================================================

Eigen::MatrixXd LinearDiscriminantAnalysis::predictProba(const Eigen::MatrixXd& X) const
{
    if (!m_bFitted) {
        std::cerr << "LDA::predictProba: Model not fitted." << std::endl;
        return Eigen::MatrixXd();
    }
    
    int n_samples = X.rows();
    int n_classes = m_vecClasses.size();
    Eigen::MatrixXd probabilities(n_samples, n_classes);
    
    for (int i = 0; i < n_samples; ++i) {
        Eigen::VectorXd scores = X.row(i) * m_matW + m_vecBias.transpose();
        
        // Apply softmax
        double max_score = scores.maxCoeff();
        Eigen::VectorXd exp_scores = (scores.array() - max_score).exp();
        double sum_exp = exp_scores.sum();
        
        probabilities.row(i) = exp_scores.transpose() / sum_exp;
    }
    
    return probabilities;
}

} // namespace DECODINGLIB