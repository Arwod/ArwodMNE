#ifndef STATISTICS_H
#define STATISTICS_H

#include "stats_global.h"
#include <Eigen/Core>
#include <QList>
#include <utility>
#include <functional>

namespace STATSLIB
{

    class STATSSHARED_EXPORT Statistics
    {
    public:
        /**
         * @brief Bootstrap confidence interval methods
         */
        enum class BootstrapMethod {
            Percentile,     /**< Percentile method */
            BCa,            /**< Bias-corrected and accelerated */
            Basic           /**< Basic bootstrap */
        };

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

        /**
         * Computes one-way ANOVA F-statistic.
         *
         * @param[in] groups  List of group data matrices (each n_samples_i x n_features).
         *
         * @return F-values (1 x n_features).
         */
        static Eigen::RowVectorXd f_oneway(const QList<Eigen::MatrixXd> &groups);

        /**
         * Computes repeated measures ANOVA F-statistic.
         *
         * @param[in] data     Data matrix (n_subjects x n_conditions x n_features).
         * @param[in] factor   Factor levels for each condition.
         *
         * @return F-values (1 x n_features).
         */
        static Eigen::RowVectorXd f_mway_rm(const QList<Eigen::MatrixXd> &data, 
                                           const Eigen::VectorXi &factor);

        /**
         * Computes 1-sample t-test without p-values (faster).
         *
         * @param[in] data   Input data (n_samples x n_features).
         * @param[in] popmean Population mean to test against (default: 0).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd ttest_1samp_no_p(const Eigen::MatrixXd &data, 
                                                   double popmean = 0.0);

        /**
         * Computes independent samples t-test without p-values (faster).
         *
         * @param[in] group1  Data for group 1 (n_samples1 x n_features).
         * @param[in] group2  Data for group 2 (n_samples2 x n_features).
         * @param[in] equal_var Assume equal variances (default: true).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd ttest_ind_no_p(const Eigen::MatrixXd &group1, 
                                                 const Eigen::MatrixXd &group2,
                                                 bool equal_var = true);

        /**
         * Computes bootstrap confidence intervals.
         *
         * @param[in] data       Input data (n_samples x n_features).
         * @param[in] statistic  Function to compute statistic (mean, median, etc.).
         * @param[in] n_bootstrap Number of bootstrap samples (default: 1000).
         * @param[in] confidence  Confidence level (default: 0.95).
         * @param[in] method     Bootstrap method (default: Percentile).
         * @param[in] seed       Random seed (0 = random).
         *
         * @return Pair of (lower_bounds, upper_bounds) for each feature.
         */
        static std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> bootstrap_ci(
            const Eigen::MatrixXd &data,
            std::function<double(const Eigen::VectorXd&)> statistic,
            int n_bootstrap = 1000,
            double confidence = 0.95,
            BootstrapMethod method = BootstrapMethod::Percentile,
            unsigned int seed = 0);

        /**
         * Computes bootstrap confidence intervals for the mean.
         *
         * @param[in] data       Input data (n_samples x n_features).
         * @param[in] n_bootstrap Number of bootstrap samples (default: 1000).
         * @param[in] confidence  Confidence level (default: 0.95).
         * @param[in] method     Bootstrap method (default: Percentile).
         * @param[in] seed       Random seed (0 = random).
         *
         * @return Pair of (lower_bounds, upper_bounds) for each feature.
         */
        static std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> bootstrap_ci_mean(
            const Eigen::MatrixXd &data,
            int n_bootstrap = 1000,
            double confidence = 0.95,
            BootstrapMethod method = BootstrapMethod::Percentile,
            unsigned int seed = 0);

        /**
         * Computes effect size (Cohen's d) for two groups.
         *
         * @param[in] group1  Data for group 1 (n_samples1 x n_features).
         * @param[in] group2  Data for group 2 (n_samples2 x n_features).
         * @param[in] pooled  Use pooled standard deviation (default: true).
         *
         * @return Cohen's d values (1 x n_features).
         */
        static Eigen::RowVectorXd cohens_d(const Eigen::MatrixXd &group1, 
                                          const Eigen::MatrixXd &group2,
                                          bool pooled = true);

        /**
         * Computes effect size (eta-squared) for ANOVA.
         *
         * @param[in] groups  List of group data matrices.
         *
         * @return Eta-squared values (1 x n_features).
         */
        static Eigen::RowVectorXd eta_squared(const QList<Eigen::MatrixXd> &groups);

        /**
         * Computes Welch's t-test (unequal variances).
         *
         * @param[in] group1  Data for group 1 (n_samples1 x n_features).
         * @param[in] group2  Data for group 2 (n_samples2 x n_features).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd welch_ttest(const Eigen::MatrixXd &group1, 
                                             const Eigen::MatrixXd &group2);

        /**
         * Computes paired samples t-test.
         *
         * @param[in] data1   Data for condition 1 (n_samples x n_features).
         * @param[in] data2   Data for condition 2 (n_samples x n_features).
         *
         * @return t-values (1 x n_features).
         */
        static Eigen::RowVectorXd paired_ttest(const Eigen::MatrixXd &data1, 
                                              const Eigen::MatrixXd &data2);

    private:
        /**
         * Computes percentile from sorted data.
         *
         * @param[in] sorted_data Sorted data vector.
         * @param[in] percentile  Percentile (0-100).
         *
         * @return Percentile value.
         */
        static double computePercentile(const Eigen::VectorXd &sorted_data, double percentile);

        /**
         * Generates bootstrap sample indices.
         *
         * @param[in] n_samples   Original sample size.
         * @param[in] n_bootstrap Number of bootstrap samples.
         * @param[in] seed        Random seed.
         *
         * @return Bootstrap sample indices.
         */
        static Eigen::MatrixXi generateBootstrapIndices(int n_samples, 
                                                       int n_bootstrap, 
                                                       unsigned int seed);
    };

} // NAMESPACE

#endif // STATISTICS_H
