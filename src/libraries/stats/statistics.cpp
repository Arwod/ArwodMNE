#include "statistics.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <QRandomGenerator>

using namespace Eigen;

namespace STATSLIB
{

    Eigen::RowVectorXd STATSLIB::Statistics::ttest1Samp(const Eigen::MatrixXd &data, double sigma)
    {
        int n_samples = data.rows();
        if (n_samples <= 1)
        {
            return Eigen::RowVectorXd::Zero(data.cols());
        }

        // Mean
        RowVectorXd mean = data.colwise().mean();

        // Variance
        MatrixXd centered = data.rowwise() - mean;
        RowVectorXd var = centered.cwiseAbs2().colwise().sum() / (n_samples - 1);

        // Standard deviation
        RowVectorXd std = var.cwiseSqrt();

        // Regularization
        if (sigma > 0.0)
        {
            std.array() += sigma;
        }

        // Standard Error of Mean
        RowVectorXd sem = std / std::sqrt(static_cast<double>(n_samples));

        // T-values
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

    Eigen::RowVectorXd STATSLIB::Statistics::ttestIndep(const Eigen::MatrixXd &group1, const Eigen::MatrixXd &group2)
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

        // Pooled Variance
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

    Eigen::RowVectorXd STATSLIB::Statistics::f_oneway(const QList<Eigen::MatrixXd> &groups)
    {
        if (groups.size() < 2)
        {
            return Eigen::RowVectorXd::Zero(groups.isEmpty() ? 0 : groups[0].cols());
        }

        int n_features = groups[0].cols();
        int n_groups = groups.size();
        
        // Calculate total sample size and group sizes
        QList<int> group_sizes;
        int total_n = 0;
        for (const auto &group : groups)
        {
            group_sizes.append(group.rows());
            total_n += group.rows();
        }

        if (total_n <= n_groups)
        {
            return Eigen::RowVectorXd::Zero(n_features);
        }

        // Calculate group means and overall mean
        QList<RowVectorXd> group_means;
        RowVectorXd overall_mean = RowVectorXd::Zero(n_features);
        
        for (const auto &group : groups)
        {
            RowVectorXd mean = group.colwise().mean();
            group_means.append(mean);
            overall_mean += mean * group.rows();
        }
        overall_mean /= total_n;

        // Calculate between-group sum of squares (SSB)
        RowVectorXd ssb = RowVectorXd::Zero(n_features);
        for (int i = 0; i < n_groups; ++i)
        {
            RowVectorXd diff = group_means[i] - overall_mean;
            ssb += group_sizes[i] * diff.cwiseAbs2();
        }

        // Calculate within-group sum of squares (SSW)
        RowVectorXd ssw = RowVectorXd::Zero(n_features);
        for (int i = 0; i < n_groups; ++i)
        {
            MatrixXd centered = groups[i].rowwise() - group_means[i];
            ssw += centered.cwiseAbs2().colwise().sum();
        }

        // Calculate F-statistic
        RowVectorXd msb = ssb / (n_groups - 1);
        RowVectorXd msw = ssw / (total_n - n_groups);
        
        RowVectorXd f_values = msb.array() / msw.array();

        // Handle division by zero
        for (int i = 0; i < f_values.size(); ++i)
        {
            if (!std::isfinite(f_values(i)) || msw(i) == 0.0)
            {
                f_values(i) = 0.0;
            }
        }

        return f_values;
    }

    Eigen::RowVectorXd STATSLIB::Statistics::f_mway_rm(const QList<Eigen::MatrixXd> &data, 
                                           const Eigen::VectorXi &factor)
    {
        if (data.isEmpty() || factor.size() != data.size())
        {
            return Eigen::RowVectorXd::Zero(data.isEmpty() ? 0 : data[0].cols());
        }

        int n_features = data[0].cols();
        int n_subjects = data[0].rows();
        int n_conditions = data.size();

        // Verify all data matrices have same dimensions
        for (const auto &mat : data)
        {
            if (mat.rows() != n_subjects || mat.cols() != n_features)
            {
                return Eigen::RowVectorXd::Zero(n_features);
            }
        }

        // Calculate condition means
        QList<RowVectorXd> condition_means;
        RowVectorXd grand_mean = RowVectorXd::Zero(n_features);
        
        for (const auto &mat : data)
        {
            RowVectorXd mean = mat.colwise().mean();
            condition_means.append(mean);
            grand_mean += mean;
        }
        grand_mean /= n_conditions;

        // Calculate subject means
        MatrixXd subject_means = MatrixXd::Zero(n_subjects, n_features);
        for (int s = 0; s < n_subjects; ++s)
        {
            for (int c = 0; c < n_conditions; ++c)
            {
                subject_means.row(s) += data[c].row(s);
            }
            subject_means.row(s) /= n_conditions;
        }

        // Calculate sum of squares
        RowVectorXd ss_total = RowVectorXd::Zero(n_features);
        RowVectorXd ss_between = RowVectorXd::Zero(n_features);
        RowVectorXd ss_subjects = RowVectorXd::Zero(n_features);
        RowVectorXd ss_error = RowVectorXd::Zero(n_features);

        // Total sum of squares
        for (int c = 0; c < n_conditions; ++c)
        {
            MatrixXd centered = data[c].rowwise() - grand_mean;
            ss_total += centered.cwiseAbs2().colwise().sum();
        }

        // Between-conditions sum of squares
        for (int c = 0; c < n_conditions; ++c)
        {
            RowVectorXd diff = condition_means[c] - grand_mean;
            ss_between += n_subjects * diff.cwiseAbs2();
        }

        // Between-subjects sum of squares
        for (int s = 0; s < n_subjects; ++s)
        {
            RowVectorXd diff = subject_means.row(s) - grand_mean;
            ss_subjects += n_conditions * diff.cwiseAbs2();
        }

        // Error sum of squares
        ss_error = ss_total - ss_between - ss_subjects;

        // Calculate F-statistic
        int df_between = n_conditions - 1;
        int df_error = (n_subjects - 1) * (n_conditions - 1);

        RowVectorXd ms_between = ss_between / df_between;
        RowVectorXd ms_error = ss_error / df_error;
        
        RowVectorXd f_values = ms_between.array() / ms_error.array();

        // Handle division by zero
        for (int i = 0; i < f_values.size(); ++i)
        {
            if (!std::isfinite(f_values(i)) || ms_error(i) == 0.0)
            {
                f_values(i) = 0.0;
            }
        }

        return f_values;
    }

    Eigen::RowVectorXd STATSLIB::Statistics::ttest_1samp_no_p(const Eigen::MatrixXd &data, 
                                                   double popmean)
    {
        int n_samples = data.rows();
        if (n_samples <= 1)
        {
            return Eigen::RowVectorXd::Zero(data.cols());
        }

        // Mean
        RowVectorXd sample_mean = data.colwise().mean();

        // Standard deviation
        MatrixXd centered = data.rowwise() - sample_mean;
        RowVectorXd var = centered.cwiseAbs2().colwise().sum() / (n_samples - 1);
        RowVectorXd std = var.cwiseSqrt();

        // Standard Error of Mean
        RowVectorXd sem = std / std::sqrt(static_cast<double>(n_samples));

        // T-values
        RowVectorXd t_values = (sample_mean.array() - popmean) / sem.array();

        // Handle NaNs/Infs
        for (int i = 0; i < t_values.size(); ++i)
        {
            if (!std::isfinite(t_values(i)))
            {
                t_values(i) = 0.0;
            }
        }

        return t_values;
    }

    Eigen::RowVectorXd STATSLIB::Statistics::ttest_ind_no_p(const Eigen::MatrixXd &group1, 
                                                 const Eigen::MatrixXd &group2,
                                                 bool equal_var)
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

        RowVectorXd se;
        if (equal_var)
        {
            // Pooled variance
            RowVectorXd var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            se = (var_pooled * (1.0 / n1 + 1.0 / n2)).cwiseSqrt();
        }
        else
        {
            // Welch's t-test (unequal variances)
            se = (var1 / n1 + var2 / n2).cwiseSqrt();
        }

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
    std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> STATSLIB::Statistics::bootstrap_ci(
        const Eigen::MatrixXd &data,
        std::function<double(const Eigen::VectorXd&)> statistic,
        int n_bootstrap,
        double confidence,
        BootstrapMethod method,
        unsigned int seed)
    {
        int n_samples = data.rows();
        int n_features = data.cols();
        
        if (n_samples == 0 || n_bootstrap <= 0)
        {
            return std::make_pair(RowVectorXd::Zero(n_features), RowVectorXd::Zero(n_features));
        }

        // Generate bootstrap indices
        MatrixXi bootstrap_indices = generateBootstrapIndices(n_samples, n_bootstrap, seed);
        
        // Calculate bootstrap statistics for each feature
        MatrixXd bootstrap_stats(n_bootstrap, n_features);
        
        for (int f = 0; f < n_features; ++f)
        {
            VectorXd feature_data = data.col(f);
            
            for (int b = 0; b < n_bootstrap; ++b)
            {
                VectorXd bootstrap_sample(n_samples);
                for (int i = 0; i < n_samples; ++i)
                {
                    bootstrap_sample(i) = feature_data(bootstrap_indices(b, i));
                }
                bootstrap_stats(b, f) = statistic(bootstrap_sample);
            }
        }

        // Calculate confidence intervals
        double alpha = 1.0 - confidence;
        double lower_percentile = 100.0 * alpha / 2.0;
        double upper_percentile = 100.0 * (1.0 - alpha / 2.0);
        
        RowVectorXd lower_bounds(n_features);
        RowVectorXd upper_bounds(n_features);

        for (int f = 0; f < n_features; ++f)
        {
            VectorXd sorted_stats = bootstrap_stats.col(f);
            std::sort(sorted_stats.data(), sorted_stats.data() + sorted_stats.size());
            
            switch (method)
            {
                case BootstrapMethod::Percentile:
                    lower_bounds(f) = computePercentile(sorted_stats, lower_percentile);
                    upper_bounds(f) = computePercentile(sorted_stats, upper_percentile);
                    break;
                    
                case BootstrapMethod::Basic:
                {
                    double original_stat = statistic(data.col(f));
                    double lower_boot = computePercentile(sorted_stats, upper_percentile);
                    double upper_boot = computePercentile(sorted_stats, lower_percentile);
                    lower_bounds(f) = 2.0 * original_stat - lower_boot;
                    upper_bounds(f) = 2.0 * original_stat - upper_boot;
                    break;
                }
                
                case BootstrapMethod::BCa:
                    // Simplified BCa - for full implementation would need bias correction and acceleration
                    lower_bounds(f) = computePercentile(sorted_stats, lower_percentile);
                    upper_bounds(f) = computePercentile(sorted_stats, upper_percentile);
                    break;
            }
        }

        return std::make_pair(lower_bounds, upper_bounds);
    }

    std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> STATSLIB::Statistics::bootstrap_ci_mean(
        const Eigen::MatrixXd &data,
        int n_bootstrap,
        double confidence,
        BootstrapMethod method,
        unsigned int seed)
    {
        auto mean_function = [](const VectorXd& x) { return x.mean(); };
        return bootstrap_ci(data, mean_function, n_bootstrap, confidence, method, seed);
    }

    Eigen::RowVectorXd STATSLIB::Statistics::cohens_d(const Eigen::MatrixXd &group1, 
                                          const Eigen::MatrixXd &group2,
                                          bool pooled)
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

        // Standard deviations
        MatrixXd centered1 = group1.rowwise() - mean1;
        RowVectorXd var1 = centered1.cwiseAbs2().colwise().sum() / (n1 - 1);
        RowVectorXd std1 = var1.cwiseSqrt();

        MatrixXd centered2 = group2.rowwise() - mean2;
        RowVectorXd var2 = centered2.cwiseAbs2().colwise().sum() / (n2 - 1);
        RowVectorXd std2 = var2.cwiseSqrt();

        RowVectorXd denominator;
        if (pooled)
        {
            // Pooled standard deviation
            RowVectorXd var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            denominator = var_pooled.cwiseSqrt();
        }
        else
        {
            // Average of standard deviations
            denominator = (std1 + std2) / 2.0;
        }

        RowVectorXd cohens_d_values = (mean1 - mean2).array() / denominator.array();

        // Handle division by zero
        for (int i = 0; i < cohens_d_values.size(); ++i)
        {
            if (!std::isfinite(cohens_d_values(i)))
            {
                cohens_d_values(i) = 0.0;
            }
        }

        return cohens_d_values;
    }

    Eigen::RowVectorXd STATSLIB::Statistics::eta_squared(const QList<Eigen::MatrixXd> &groups)
    {
        if (groups.size() < 2)
        {
            return Eigen::RowVectorXd::Zero(groups.isEmpty() ? 0 : groups[0].cols());
        }

        int n_features = groups[0].cols();
        int n_groups = groups.size();
        
        // Calculate total sample size
        int total_n = 0;
        for (const auto &group : groups)
        {
            total_n += group.rows();
        }

        if (total_n <= n_groups)
        {
            return Eigen::RowVectorXd::Zero(n_features);
        }

        // Calculate group means and overall mean
        QList<RowVectorXd> group_means;
        RowVectorXd overall_mean = RowVectorXd::Zero(n_features);
        
        for (const auto &group : groups)
        {
            RowVectorXd mean = group.colwise().mean();
            group_means.append(mean);
            overall_mean += mean * group.rows();
        }
        overall_mean /= total_n;

        // Calculate between-group sum of squares (SSB)
        RowVectorXd ssb = RowVectorXd::Zero(n_features);
        for (int i = 0; i < n_groups; ++i)
        {
            RowVectorXd diff = group_means[i] - overall_mean;
            ssb += groups[i].rows() * diff.cwiseAbs2();
        }

        // Calculate total sum of squares (SST)
        RowVectorXd sst = RowVectorXd::Zero(n_features);
        for (const auto &group : groups)
        {
            MatrixXd centered = group.rowwise() - overall_mean;
            sst += centered.cwiseAbs2().colwise().sum();
        }

        // Calculate eta-squared
        RowVectorXd eta_sq = ssb.array() / sst.array();

        // Handle division by zero
        for (int i = 0; i < eta_sq.size(); ++i)
        {
            if (!std::isfinite(eta_sq(i)) || sst(i) == 0.0)
            {
                eta_sq(i) = 0.0;
            }
        }

        return eta_sq;
    }

    Eigen::RowVectorXd STATSLIB::Statistics::welch_ttest(const Eigen::MatrixXd &group1, 
                                             const Eigen::MatrixXd &group2)
    {
        return ttest_ind_no_p(group1, group2, false); // Use unequal variances
    }

    Eigen::RowVectorXd STATSLIB::Statistics::paired_ttest(const Eigen::MatrixXd &data1, 
                                              const Eigen::MatrixXd &data2)
    {
        if (data1.rows() != data2.rows() || data1.cols() != data2.cols())
        {
            return Eigen::RowVectorXd::Zero(std::min(data1.cols(), data2.cols()));
        }

        // Calculate differences
        MatrixXd differences = data1 - data2;
        
        // Perform one-sample t-test on differences (against mean = 0)
        return ttest_1samp_no_p(differences, 0.0);
    }

    double STATSLIB::Statistics::computePercentile(const Eigen::VectorXd &sorted_data, double percentile)
    {
        if (sorted_data.size() == 0)
        {
            return 0.0;
        }
        
        if (percentile <= 0.0)
        {
            return sorted_data(0);
        }
        
        if (percentile >= 100.0)
        {
            return sorted_data(sorted_data.size() - 1);
        }

        double index = (percentile / 100.0) * (sorted_data.size() - 1);
        int lower_index = static_cast<int>(std::floor(index));
        int upper_index = static_cast<int>(std::ceil(index));
        
        if (lower_index == upper_index)
        {
            return sorted_data(lower_index);
        }
        
        double weight = index - lower_index;
        return (1.0 - weight) * sorted_data(lower_index) + weight * sorted_data(upper_index);
    }

    Eigen::MatrixXi STATSLIB::Statistics::generateBootstrapIndices(int n_samples, 
                                                       int n_bootstrap, 
                                                       unsigned int seed)
    {
        MatrixXi indices(n_bootstrap, n_samples);
        
        QRandomGenerator generator;
        if (seed != 0)
        {
            generator.seed(seed);
        }
        
        for (int b = 0; b < n_bootstrap; ++b)
        {
            for (int s = 0; s < n_samples; ++s)
            {
                indices(b, s) = generator.bounded(n_samples);
            }
        }
        
        return indices;
    }