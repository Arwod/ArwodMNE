#ifndef CSP_H
#define CSP_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "decoding_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>

//=============================================================================================================
// DEFINE NAMESPACE DECODINGLIB
//=============================================================================================================

namespace DECODINGLIB
{

//=============================================================================================================
/**
 * Common Spatial Patterns (CSP)
 * 
 * Implements the CSP algorithm for binary classification tasks.
 */
class DECODINGSHARED_EXPORT CSP
{
public:
    typedef std::shared_ptr<CSP> SPtr;            /**< Shared pointer type for CSP. */
    typedef std::shared_ptr<const CSP> ConstSPtr; /**< Const shared pointer type for CSP. */

    /**
     * @brief Regularization methods for covariance estimation
     */
    enum class RegularizationMethod {
        None,           /**< No regularization */
        LedoitWolf,     /**< Ledoit-Wolf shrinkage */
        Diagonal,       /**< Diagonal loading */
        Shrinkage       /**< Manual shrinkage parameter */
    };

    //=========================================================================================================
    /**
     * Constructs a CSP object.
     *
     * @param[in] n_components  Number of components to decompose M/EEG signals.
     * @param[in] norm_trace    If true, normalize covariance matrices by their trace.
     * @param[in] log           If true, apply log to the variance features.
     * @param[in] reg_method    Regularization method for covariance estimation.
     * @param[in] reg_param     Regularization parameter (for manual shrinkage).
     */
    explicit CSP(int n_components = 4, 
                 bool norm_trace = false, 
                 bool log = true, 
                 RegularizationMethod reg_method = RegularizationMethod::None,
                 double reg_param = 0.01);

    //=========================================================================================================
    /**
     * Destroys the CSP object.
     */
    ~CSP() = default;

    //=========================================================================================================
    /**
     * Fit CSP filters from epochs.
     * 
     * @param[in] epochs Vector of epochs data (Channels x Time).
     * @param[in] labels Vector of labels corresponding to each epoch (0 or 1).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels);

    //=========================================================================================================
    /**
     * Apply CSP filters to transform data.
     * 
     * @param[in] epochs Vector of epochs data.
     * @return Feature matrix (n_epochs x n_components).
     */
    Eigen::MatrixXd transform(const std::vector<Eigen::MatrixXd>& epochs) const;

    //=========================================================================================================
    /**
     * Fit and transform in one step.
     * 
     * @param[in] epochs Vector of epochs data.
     * @param[in] labels Vector of labels.
     * @return Feature matrix (n_epochs x n_components).
     */
    Eigen::MatrixXd fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                const std::vector<int>& labels);

    //=========================================================================================================
    /**
     * Get spatial filters (W).
     * Each column is a spatial filter.
     */
    Eigen::MatrixXd getFilters() const { return m_matFilters; }

    //=========================================================================================================
    /**
     * Get spatial patterns (A = W^-1).
     * Each column is a spatial pattern.
     */
    Eigen::MatrixXd getPatterns() const { return m_matPatterns; }

    //=========================================================================================================
    /**
     * Get eigen values.
     */
    Eigen::VectorXd getEigenValues() const { return m_vecEigenValues; }

    //=========================================================================================================
    /**
     * Get selected component indices.
     */
    std::vector<int> getSelectedIndices() const { return m_vecSelectedIndices; }

    //=========================================================================================================
    /**
     * Set component selection strategy.
     * 
     * @param[in] strategy "extremes" (default) or "top" or "bottom"
     */
    void setComponentSelection(const std::string& strategy) { m_strComponentSelection = strategy; }

private:
    //=========================================================================================================
    /**
     * Compute covariance matrix with regularization.
     */
    Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd& epoch) const;

    //=========================================================================================================
    /**
     * Apply regularization to covariance matrix.
     */
    Eigen::MatrixXd regularizeCovariance(const Eigen::MatrixXd& cov) const;

    //=========================================================================================================
    /**
     * Select component indices based on eigenvalues.
     */
    std::vector<int> selectComponents(const Eigen::VectorXd& eigenvalues) const;

    int m_iNComponents;
    bool m_bNormTrace;
    bool m_bLog;
    RegularizationMethod m_regMethod;
    double m_dRegParam;
    std::string m_strComponentSelection;

    Eigen::MatrixXd m_matFilters;  // Spatial filters (W)
    Eigen::MatrixXd m_matPatterns; // Spatial patterns (A = W^-1)
    Eigen::VectorXd m_vecEigenValues;
    std::vector<int> m_vecSelectedIndices;
    std::vector<std::string> m_vecClassNames;
};

//=============================================================================================================
/**
 * Source Power Comodulation (SPoC)
 * 
 * Implements the SPoC algorithm for continuous target variable prediction.
 */
class DECODINGSHARED_EXPORT SPoC
{
public:
    typedef std::shared_ptr<SPoC> SPtr;            /**< Shared pointer type for SPoC. */
    typedef std::shared_ptr<const SPoC> ConstSPtr; /**< Const shared pointer type for SPoC. */

    //=========================================================================================================
    /**
     * Constructs a SPoC object.
     *
     * @param[in] n_components  Number of components to extract.
     * @param[in] norm_trace    If true, normalize covariance matrices by their trace.
     * @param[in] log           If true, apply log to the variance features.
     * @param[in] reg_param     Regularization parameter for covariance estimation.
     */
    explicit SPoC(int n_components = 4, 
                  bool norm_trace = false, 
                  bool log = true, 
                  double reg_param = 0.01);

    //=========================================================================================================
    /**
     * Destroys the SPoC object.
     */
    ~SPoC() = default;

    //=========================================================================================================
    /**
     * Fit SPoC filters from epochs and continuous target.
     * 
     * @param[in] epochs Vector of epochs data (Channels x Time).
     * @param[in] target Vector of continuous target values for each epoch.
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs, const Eigen::VectorXd& target);

    //=========================================================================================================
    /**
     * Apply SPoC filters to transform data.
     * 
     * @param[in] epochs Vector of epochs data.
     * @return Feature matrix (n_epochs x n_components).
     */
    Eigen::MatrixXd transform(const std::vector<Eigen::MatrixXd>& epochs) const;

    //=========================================================================================================
    /**
     * Fit and transform in one step.
     * 
     * @param[in] epochs Vector of epochs data.
     * @param[in] target Vector of continuous target values.
     * @return Feature matrix (n_epochs x n_components).
     */
    Eigen::MatrixXd fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                const Eigen::VectorXd& target);

    //=========================================================================================================
    /**
     * Get spatial filters (W).
     */
    Eigen::MatrixXd getFilters() const { return m_matFilters; }

    //=========================================================================================================
    /**
     * Get spatial patterns (A).
     */
    Eigen::MatrixXd getPatterns() const { return m_matPatterns; }

    //=========================================================================================================
    /**
     * Get eigen values.
     */
    Eigen::VectorXd getEigenValues() const { return m_vecEigenValues; }

private:
    //=========================================================================================================
    /**
     * Compute covariance matrix.
     */
    Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd& epoch) const;

    //=========================================================================================================
    /**
     * Compute weighted covariance matrix based on target values.
     */
    Eigen::MatrixXd computeWeightedCovariance(const std::vector<Eigen::MatrixXd>& epochs,
                                             const Eigen::VectorXd& weights) const;

    int m_iNComponents;
    bool m_bNormTrace;
    bool m_bLog;
    double m_dRegParam;

    Eigen::MatrixXd m_matFilters;  // Spatial filters (W)
    Eigen::MatrixXd m_matPatterns; // Spatial patterns (A)
    Eigen::VectorXd m_vecEigenValues;
};

} // namespace DECODINGLIB

#endif // CSP_H
