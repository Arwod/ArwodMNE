#ifndef TTEST_H
#define TTEST_H

#include "stats_global.h"
#include <Eigen/Core>
#include <utility>

namespace STATSLIB {

class STATSSHARED_EXPORT TTest
{
public:
    /**
     * Compute 1-sample T-test.
     * Tests if the mean of data is significantly different from 0.
     * 
     * @param[in] data Input data (n_samples x n_features).
     *                 Samples are observations (e.g., subjects/epochs).
     *                 Features are variables (e.g., channels/time-points).
     *                 T-test is computed for each feature across samples.
     * @return Pair of (T-values, P-values).
     *         T-values: (1 x n_features).
     *         P-values: (1 x n_features) - currently placeholder or approximated.
     */
    static std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> ttest_1samp(const Eigen::MatrixXd& data);

    /**
     * Compute 2-sample independent T-test.
     * Tests if the means of two independent samples are significantly different.
     * Assumes equal variance (for now).
     * 
     * @param[in] data1 Group 1 data (n_samples1 x n_features).
     * @param[in] data2 Group 2 data (n_samples2 x n_features).
     * @return Pair of (T-values, P-values).
     */
    static std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> ttest_ind(const Eigen::MatrixXd& data1, const Eigen::MatrixXd& data2);
};

} // NAMESPACE

#endif // TTEST_H
