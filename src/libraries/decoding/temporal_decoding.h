#ifndef TEMPORAL_DECODING_H
#define TEMPORAL_DECODING_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "decoding_global.h"
#include <Eigen/Core>
#include <vector>
#include <memory>
#include <functional>

//=============================================================================================================
// DEFINE NAMESPACE DECODINGLIB
//=============================================================================================================

namespace DECODINGLIB
{

//=============================================================================================================
/**
 * Base class for temporal decoding estimators
 */
class DECODINGSHARED_EXPORT BaseTemporalDecoder
{
public:
    typedef std::shared_ptr<BaseTemporalDecoder> SPtr;
    typedef std::shared_ptr<const BaseTemporalDecoder> ConstSPtr;

    virtual ~BaseTemporalDecoder() = default;

    /**
     * Fit the decoder on training data
     */
    virtual bool fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) = 0;

    /**
     * Predict labels for test data
     */
    virtual Eigen::VectorXi predict(const Eigen::MatrixXd& X) const = 0;

    /**
     * Predict probabilities for test data
     */
    virtual Eigen::MatrixXd predictProba(const Eigen::MatrixXd& X) const = 0;

    /**
     * Score the decoder (accuracy for classification)
     */
    virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const;
};

//=============================================================================================================
/**
 * Sliding Window Estimator
 * 
 * Applies a decoder to sliding windows of time series data.
 */
class DECODINGSHARED_EXPORT SlidingEstimator
{
public:
    typedef std::shared_ptr<SlidingEstimator> SPtr;
    typedef std::shared_ptr<const SlidingEstimator> ConstSPtr;

    //=========================================================================================================
    /**
     * Constructs a SlidingEstimator.
     *
     * @param[in] base_estimator  Base estimator to apply at each time point.
     * @param[in] n_jobs         Number of parallel jobs (not implemented yet).
     * @param[in] scoring        Scoring method ("accuracy", "roc_auc").
     */
    explicit SlidingEstimator(std::shared_ptr<BaseTemporalDecoder> base_estimator,
                             int n_jobs = 1,
                             const std::string& scoring = "accuracy");

    //=========================================================================================================
    /**
     * Destroys the SlidingEstimator.
     */
    ~SlidingEstimator() = default;

    //=========================================================================================================
    /**
     * Fit the sliding estimator.
     * 
     * @param[in] X  Training data (n_epochs x n_channels x n_times).
     * @param[in] y  Training labels (n_epochs).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y);

    //=========================================================================================================
    /**
     * Predict using the sliding estimator.
     * 
     * @param[in] X  Test data (n_epochs x n_channels x n_times).
     * @return Predictions (n_epochs x n_times).
     */
    Eigen::MatrixXi predict(const std::vector<Eigen::MatrixXd>& X) const;

    //=========================================================================================================
    /**
     * Score the sliding estimator.
     * 
     * @param[in] X  Test data (n_epochs x n_channels x n_times).
     * @param[in] y  True labels (n_epochs).
     * @return Scores (n_times).
     */
    Eigen::VectorXd score(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y) const;

    //=========================================================================================================
    /**
     * Get fitted estimators for each time point.
     */
    std::vector<std::shared_ptr<BaseTemporalDecoder>> getEstimators() const { return m_estimators; }

private:
    std::shared_ptr<BaseTemporalDecoder> m_baseEstimator;
    std::vector<std::shared_ptr<BaseTemporalDecoder>> m_estimators;
    int m_nJobs;
    std::string m_scoring;
    int m_nTimes;
};

//=============================================================================================================
/**
 * Generalizing Estimator
 * 
 * Fits a decoder at each time point and tests generalization across time.
 */
class DECODINGSHARED_EXPORT GeneralizingEstimator
{
public:
    typedef std::shared_ptr<GeneralizingEstimator> SPtr;
    typedef std::shared_ptr<const GeneralizingEstimator> ConstSPtr;

    //=========================================================================================================
    /**
     * Constructs a GeneralizingEstimator.
     *
     * @param[in] base_estimator  Base estimator to apply.
     * @param[in] n_jobs         Number of parallel jobs (not implemented yet).
     * @param[in] scoring        Scoring method ("accuracy", "roc_auc").
     */
    explicit GeneralizingEstimator(std::shared_ptr<BaseTemporalDecoder> base_estimator,
                                  int n_jobs = 1,
                                  const std::string& scoring = "accuracy");

    //=========================================================================================================
    /**
     * Destroys the GeneralizingEstimator.
     */
    ~GeneralizingEstimator() = default;

    //=========================================================================================================
    /**
     * Fit the generalizing estimator.
     * 
     * @param[in] X  Training data (n_epochs x n_channels x n_times).
     * @param[in] y  Training labels (n_epochs).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y);

    //=========================================================================================================
    /**
     * Score the generalizing estimator (temporal generalization matrix).
     * 
     * @param[in] X  Test data (n_epochs x n_channels x n_times).
     * @param[in] y  True labels (n_epochs).
     * @return Generalization matrix (n_train_times x n_test_times).
     */
    Eigen::MatrixXd score(const std::vector<Eigen::MatrixXd>& X, const Eigen::VectorXi& y) const;

    //=========================================================================================================
    /**
     * Get fitted estimators for each time point.
     */
    std::vector<std::shared_ptr<BaseTemporalDecoder>> getEstimators() const { return m_estimators; }

private:
    std::shared_ptr<BaseTemporalDecoder> m_baseEstimator;
    std::vector<std::shared_ptr<BaseTemporalDecoder>> m_estimators;
    int m_nJobs;
    std::string m_scoring;
    int m_nTimes;
};

//=============================================================================================================
/**
 * Simple Linear Discriminant Analysis (LDA) decoder
 * 
 * Basic implementation for use with temporal decoders.
 */
class DECODINGSHARED_EXPORT LinearDiscriminantAnalysis : public BaseTemporalDecoder
{
public:
    typedef std::shared_ptr<LinearDiscriminantAnalysis> SPtr;
    typedef std::shared_ptr<const LinearDiscriminantAnalysis> ConstSPtr;

    //=========================================================================================================
    /**
     * Constructs an LDA decoder.
     *
     * @param[in] reg_param  Regularization parameter for covariance matrix.
     */
    explicit LinearDiscriminantAnalysis(double reg_param = 0.01);

    //=========================================================================================================
    /**
     * Destroys the LDA decoder.
     */
    ~LinearDiscriminantAnalysis() = default;

    //=========================================================================================================
    /**
     * Fit the LDA decoder.
     * 
     * @param[in] X  Training data (n_samples x n_features).
     * @param[in] y  Training labels (n_samples).
     * @return true if successful.
     */
    bool fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) override;

    //=========================================================================================================
    /**
     * Predict labels.
     * 
     * @param[in] X  Test data (n_samples x n_features).
     * @return Predicted labels (n_samples).
     */
    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const override;

    //=========================================================================================================
    /**
     * Predict class probabilities.
     * 
     * @param[in] X  Test data (n_samples x n_features).
     * @return Class probabilities (n_samples x n_classes).
     */
    Eigen::MatrixXd predictProba(const Eigen::MatrixXd& X) const override;

private:
    double m_dRegParam;
    Eigen::MatrixXd m_matW;        // Weight matrix
    Eigen::VectorXd m_vecBias;     // Bias vector
    Eigen::VectorXi m_vecClasses;  // Unique class labels
    Eigen::VectorXd m_vecPriors;   // Class priors
    bool m_bFitted;
};

} // namespace DECODINGLIB

#endif // TEMPORAL_DECODING_H