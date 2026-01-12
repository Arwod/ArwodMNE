#include "statistics.h"
#include <cmath>
#include <iostream>

using namespace Eigen;

namespace STATSLIB
{

    Eigen::RowVectorXd Statistics::ttest1Samp(const Eigen::MatrixXd &data, double sigma)
    {
        int n_samples = data.rows();
        if (n_samples <= 1)
        {
            return Eigen::RowVectorXd::Zero(data.cols());
        }

        // Mean
        RowVectorXd mean = data.colwise().mean();

        // Variance
        // Eigen doesn't have colwise variance directly
        // var = (data - mean).squaredNorm() / (n-1) ? No, that's scalar.
        // var = (data - mean).cwiseAbs2().colwise().sum() / (n-1)

        MatrixXd centered = data.rowwise() - mean;
        RowVectorXd var = centered.cwiseAbs2().colwise().sum() / (n_samples - 1);

        // Standard deviation
        RowVectorXd std = var.cwiseSqrt();

        // Regularization? (hat adjustment)
        if (sigma > 0.0)
        {
            std.array() += sigma;
        }

        // Standard Error of Mean
        RowVectorXd sem = std / std::sqrt(static_cast<double>(n_samples));

        // T-values
        // Avoid division by zero
        RowVectorXd t_values = mean.array() / sem.array();

        // Handle NaNs/Infs if sem is 0
        for (int i = 0; i < t_values.size(); ++i)
        {
            if (!std::isfinite(t_values(i)))
            {
                t_values(i) = 0.0;
            }
        }

        return t_values;
    }

    Eigen::RowVectorXd Statistics::ttestIndep(const Eigen::MatrixXd &group1, const Eigen::MatrixXd &group2)
    {
        int n1 = group1.rows();
        int n2 = group2.rows();

        if (n1 <= 1 || n2 <= 1)
        {
            return Eigen::RowVectorXd::Zero(group1.cols());
        }

        // Means
        RowVectorXd mean1 = group1.colwise().mean();
        RowVectorXd mean2 = group2.colwise().mean();

        // Variances
        MatrixXd centered1 = group1.rowwise() - mean1;
        RowVectorXd var1 = centered1.cwiseAbs2().colwise().sum() / (n1 - 1);

        MatrixXd centered2 = group2.rowwise() - mean2;
        RowVectorXd var2 = centered2.cwiseAbs2().colwise().sum() / (n2 - 1);

        // Pooled Variance (Equal variance assumption)
        // s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)
        RowVectorXd var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);

        // SE = sqrt(var_pooled * (1/n1 + 1/n2))
        RowVectorXd se = (var_pooled * (1.0 / n1 + 1.0 / n2)).cwiseSqrt();

        RowVectorXd t_values = (mean1 - mean2).array() / se.array();

        for (int i = 0; i < t_values.size(); ++i)
        {
            if (!std::isfinite(t_values(i)))
            {
                t_values(i) = 0.0;
            }
        }

        return t_values;
    }

} // NAMESPACE
