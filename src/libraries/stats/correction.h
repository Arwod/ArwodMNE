#ifndef CORRECTION_H
#define CORRECTION_H

#include "stats_global.h"
#include <Eigen/Core>
#include <vector>
#include <QString>

namespace STATSLIB {

class STATSSHARED_EXPORT Correction
{
public:
    /**
     * @brief Multiple comparison correction methods
     */
    enum class Method {
        Bonferroni,     /**< Bonferroni correction */
        Holm,           /**< Holm-Bonferroni correction */
        Sidak,          /**< Sidak correction */
        HolmSidak,      /**< Holm-Sidak correction */
        FDR_BH,         /**< False Discovery Rate (Benjamini-Hochberg) */
        FDR_BY,         /**< False Discovery Rate (Benjamini-Yekutieli) */
        FDR_TST,        /**< False Discovery Rate (Two-Stage) */
        None            /**< No correction */
    };

    /**
     * Bonferroni Correction.
     * p_corr = p * n_tests
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values (clamped to 1.0).
     */
    static Eigen::VectorXd bonferroni(const Eigen::VectorXd& p_values);
    
    /**
     * Holm-Bonferroni Correction (step-down method).
     * More powerful than standard Bonferroni.
     * 
     * @param[in] p_values Input P-values.
     * @param[in] alpha    Significance level (default: 0.05).
     * @return Corrected P-values.
     */
    static Eigen::VectorXd holm(const Eigen::VectorXd& p_values, double alpha = 0.05);
    
    /**
     * Sidak Correction.
     * p_corr = 1 - (1 - p)^n_tests
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values.
     */
    static Eigen::VectorXd sidak(const Eigen::VectorXd& p_values);
    
    /**
     * Holm-Sidak Correction (step-down Sidak method).
     * 
     * @param[in] p_values Input P-values.
     * @param[in] alpha    Significance level (default: 0.05).
     * @return Corrected P-values.
     */
    static Eigen::VectorXd holmSidak(const Eigen::VectorXd& p_values, double alpha = 0.05);
    
    /**
     * FDR Correction (Benjamini-Hochberg).
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values (q-values).
     */
    static Eigen::VectorXd fdr(const Eigen::VectorXd& p_values);
    
    /**
     * FDR Correction (Benjamini-Yekutieli).
     * More conservative than BH, controls FDR under arbitrary dependence.
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values (q-values).
     */
    static Eigen::VectorXd fdrBY(const Eigen::VectorXd& p_values);
    
    /**
     * FDR Two-Stage Correction (Benjamini, Krieger & Yekutieli).
     * Adaptive method that estimates the proportion of true null hypotheses.
     * 
     * @param[in] p_values Input P-values.
     * @param[in] alpha    Significance level (default: 0.05).
     * @return Corrected P-values (q-values).
     */
    static Eigen::VectorXd fdrTwoStage(const Eigen::VectorXd& p_values, double alpha = 0.05);
    
    /**
     * Generic multiple comparison correction.
     * 
     * @param[in] p_values Input P-values.
     * @param[in] method   Correction method.
     * @param[in] alpha    Significance level (default: 0.05).
     * @return Corrected P-values.
     */
    static Eigen::VectorXd multipleComparison(const Eigen::VectorXd& p_values, 
                                             Method method, 
                                             double alpha = 0.05);
    
    /**
     * Get rejection mask for corrected p-values.
     * 
     * @param[in] p_corrected Corrected P-values.
     * @param[in] alpha       Significance level (default: 0.05).
     * @return Boolean vector indicating rejected hypotheses.
     */
    static std::vector<bool> getRejectionMask(const Eigen::VectorXd& p_corrected, 
                                             double alpha = 0.05);
    
    /**
     * Count number of rejected hypotheses.
     * 
     * @param[in] p_corrected Corrected P-values.
     * @param[in] alpha       Significance level (default: 0.05).
     * @return Number of rejected hypotheses.
     */
    static int countRejected(const Eigen::VectorXd& p_corrected, double alpha = 0.05);
    
    /**
     * Estimate the proportion of true null hypotheses (pi0).
     * Used in adaptive FDR methods.
     * 
     * @param[in] p_values Input P-values.
     * @param[in] lambda   Threshold parameter (default: 0.5).
     * @return Estimated proportion of true nulls.
     */
    static double estimatePi0(const Eigen::VectorXd& p_values, double lambda = 0.5);

private:
    /**
     * Sort indices based on p-values.
     * 
     * @param[in] p_values Input P-values.
     * @return Sorted indices (ascending order of p-values).
     */
    static std::vector<int> sortIndices(const Eigen::VectorXd& p_values);
    
    /**
     * Restore original order from sorted results.
     * 
     * @param[in] sorted_values Sorted values.
     * @param[in] indices       Original indices.
     * @return Values in original order.
     */
    static Eigen::VectorXd restoreOrder(const Eigen::VectorXd& sorted_values, 
                                       const std::vector<int>& indices);
};

} // NAMESPACE

#endif // CORRECTION_H
