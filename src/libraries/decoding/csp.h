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

    //=========================================================================================================
    /**
     * Constructs a CSP object.
     *
     * @param[in] n_components  Number of components to decompose M/EEG signals.
     * @param[in] norm_trace    If true, normalize covariance matrices by their trace.
     * @param[in] log           If true, apply log to the variance features.
     * @param[in] cov_est       If true, use covariance estimation (Ledoit-Wolf) - NOT IMPLEMENTED YET.
     */
    explicit CSP(int n_components = 4, bool norm_trace = false, bool log = true, bool cov_est = false);

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

private:
    //=========================================================================================================
    /**
     * Compute covariance matrix.
     */
    Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd& epoch) const;

    int m_iNComponents;
    bool m_bNormTrace;
    bool m_bLog;
    bool m_bCovEst;

    Eigen::MatrixXd m_matFilters;  // Spatial filters (W)
    Eigen::MatrixXd m_matPatterns; // Spatial patterns (A = W^-1)
    Eigen::VectorXd m_vecEigenValues;
    std::vector<std::string> m_vecClassNames;
};

} // namespace DECODINGLIB

#endif // CSP_H
