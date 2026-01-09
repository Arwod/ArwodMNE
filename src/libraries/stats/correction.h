#ifndef CORRECTION_H
#define CORRECTION_H

#include "stats_global.h"
#include <Eigen/Core>
#include <vector>

namespace STATSLIB {

class STATSSHARED_EXPORT Correction
{
public:
    /**
     * Bonferroni Correction.
     * p_corr = p * n_tests
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values (clamped to 1.0).
     */
    static Eigen::VectorXd bonferroni(const Eigen::VectorXd& p_values);
    
    /**
     * FDR Correction (Benjamini-Hochberg).
     * 
     * @param[in] p_values Input P-values.
     * @return Corrected P-values (q-values).
     */
    static Eigen::VectorXd fdr(const Eigen::VectorXd& p_values);
};

} // NAMESPACE

#endif // CORRECTION_H
