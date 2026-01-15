#ifndef STATISTICS_H
#define STATISTICS_H

#include "stats_global.h"
#include <Eigen/Core>

namespace STATSLIB
{

    class STATSSHARED_EXPORT Statistics
    {
    public:
        /**
         * Computes the 1-sample t-test on the columns of data.
         *
         * @param[in] data   Input data (n_samples x n_features).
         * @param[in] sigma  Regularization parameter (default 0).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd ttest1Samp(const Eigen::MatrixXd &data, double sigma = 0.0);

        /**
         * Computes the independent samples t-test.
         *
         * @param[in] group1  Data for group 1 (n_samples1 x n_features).
         * @param[in] group2  Data for group 2 (n_samples2 x n_features).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd ttestIndep(const Eigen::MatrixXd &group1, const Eigen::MatrixXd &group2);
    };

} // NAMESPACE

#endif // STATISTICS_H
