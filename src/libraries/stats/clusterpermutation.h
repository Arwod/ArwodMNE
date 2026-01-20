#ifndef CLUSTERPERMUTATION_H
#define CLUSTERPERMUTATION_H

#include "stats_global.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <QList>
#include <QSharedPointer>
#include <QString>
#include <QMap>

namespace STATSLIB
{

    class STATSSHARED_EXPORT ClusterPermutation
    {
    public:
        /**
         * @brief Cluster statistic methods
         */
        enum class ClusterStatistic {
            Sum,        /**< Sum of statistics in cluster */
            Max,        /**< Maximum statistic in cluster */
            Mean,       /**< Mean statistic in cluster */
            Size        /**< Cluster size (number of elements) */
        };

        /**
         * @brief Statistical test types
         */
        enum class TestType {
            OneSample,      /**< One-sample test */
            TwoSample,      /**< Two-sample independent test */
            Paired          /**< Paired samples test */
        };

        struct Cluster
        {
            QList<int> indices;     /**< Indices of features in the cluster */
            double clusterStat;     /**< Statistic (e.g. sum of t-values) */
            double pValue;          /**< P-value */
            int timeStart;          /**< Start time index (for spatio-temporal clusters) */
            int timeEnd;            /**< End time index (for spatio-temporal clusters) */
            QList<int> timeIndices; /**< Time indices for spatio-temporal clusters */

            Cluster() : clusterStat(0.0), pValue(1.0), timeStart(-1), timeEnd(-1) {}

            // Sorting for easier output reading
            bool operator<(const Cluster &other) const
            {
                return std::abs(clusterStat) > std::abs(other.clusterStat); // Sort by magnitude descending
            }
        };

        /**
         * @brief Performs 1-sample permutation cluster test.
         *
         * @param data           Input data (n_samples x n_features).
         * @param threshold      Threshold for t-values (e.g. 2.0 or -2.0).
         *                       Points with |t| > |threshold| are selected.
         * @param n_permutations Number of permutations.
         * @param adjacency      Adjacency matrix (n_features x n_features).
         *                       If empty, assumes 1D linear adjacency.
         * @param tail           0 for two-tailed, 1 for upper, -1 for lower.
         * @param cluster_stat   Method for computing cluster statistics.
         * @param seed           Random seed for reproducibility (0 = random).
         *
         * @return List of clusters found.
         */
        static QList<Cluster> permutationClusterOneSampleTest(
            const Eigen::MatrixXd &data,
            double threshold,
            int n_permutations = 1024,
            const Eigen::SparseMatrix<double> &adjacency = Eigen::SparseMatrix<double>(),
            int tail = 0,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum,
            unsigned int seed = 0);

        /**
         * @brief Performs 2-sample permutation cluster test.
         *
         * @param data1          Input data for group 1 (n_samples1 x n_features).
         * @param data2          Input data for group 2 (n_samples2 x n_features).
         * @param threshold      Threshold for t-values.
         * @param n_permutations Number of permutations.
         * @param adjacency      Adjacency matrix (n_features x n_features).
         * @param tail           0 for two-tailed, 1 for upper, -1 for lower.
         * @param cluster_stat   Method for computing cluster statistics.
         * @param seed           Random seed for reproducibility (0 = random).
         *
         * @return List of clusters found.
         */
        static QList<Cluster> permutationClusterTwoSampleTest(
            const Eigen::MatrixXd &data1,
            const Eigen::MatrixXd &data2,
            double threshold,
            int n_permutations = 1024,
            const Eigen::SparseMatrix<double> &adjacency = Eigen::SparseMatrix<double>(),
            int tail = 0,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum,
            unsigned int seed = 0);

        /**
         * @brief Performs spatio-temporal permutation cluster test.
         *
         * @param data           Input data (n_samples x n_channels x n_times).
         * @param threshold      Threshold for t-values.
         * @param n_permutations Number of permutations.
         * @param spatial_adjacency Spatial adjacency matrix (n_channels x n_channels).
         * @param temporal_adjacency Temporal adjacency matrix (n_times x n_times).
         *                          If empty, assumes linear temporal adjacency.
         * @param tail           0 for two-tailed, 1 for upper, -1 for lower.
         * @param cluster_stat   Method for computing cluster statistics.
         * @param test_type      Type of statistical test.
         * @param seed           Random seed for reproducibility (0 = random).
         *
         * @return List of spatio-temporal clusters found.
         */
        static QList<Cluster> spatioTemporalClusterTest(
            const QList<Eigen::MatrixXd> &data,
            double threshold,
            int n_permutations = 1024,
            const Eigen::SparseMatrix<double> &spatial_adjacency = Eigen::SparseMatrix<double>(),
            const Eigen::SparseMatrix<double> &temporal_adjacency = Eigen::SparseMatrix<double>(),
            int tail = 0,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum,
            TestType test_type = TestType::OneSample,
            unsigned int seed = 0);

        /**
         * @brief Performs spatio-temporal permutation cluster test for two groups.
         *
         * @param data1          Input data for group 1 (list of n_samples1 matrices, each n_channels x n_times).
         * @param data2          Input data for group 2 (list of n_samples2 matrices, each n_channels x n_times).
         * @param threshold      Threshold for t-values.
         * @param n_permutations Number of permutations.
         * @param spatial_adjacency Spatial adjacency matrix (n_channels x n_channels).
         * @param temporal_adjacency Temporal adjacency matrix (n_times x n_times).
         * @param tail           0 for two-tailed, 1 for upper, -1 for lower.
         * @param cluster_stat   Method for computing cluster statistics.
         * @param seed           Random seed for reproducibility (0 = random).
         *
         * @return List of spatio-temporal clusters found.
         */
        static QList<Cluster> spatioTemporalClusterTwoSampleTest(
            const QList<Eigen::MatrixXd> &data1,
            const QList<Eigen::MatrixXd> &data2,
            double threshold,
            int n_permutations = 1024,
            const Eigen::SparseMatrix<double> &spatial_adjacency = Eigen::SparseMatrix<double>(),
            const Eigen::SparseMatrix<double> &temporal_adjacency = Eigen::SparseMatrix<double>(),
            int tail = 0,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum,
            unsigned int seed = 0);

        /**
         * @brief Create linear adjacency matrix for 1D data.
         *
         * @param n_features Number of features.
         * @return Sparse adjacency matrix.
         */
        static Eigen::SparseMatrix<double> createLinearAdjacency(int n_features);

        /**
         * @brief Create grid adjacency matrix for 2D data.
         *
         * @param n_rows Number of rows.
         * @param n_cols Number of columns.
         * @param connect_diag Whether to connect diagonal neighbors.
         * @return Sparse adjacency matrix.
         */
        static Eigen::SparseMatrix<double> createGridAdjacency(int n_rows, int n_cols, bool connect_diag = false);

    private:
        /**
         * @brief Finds clusters in a thresholded map.
         *
         * @param stats      Statistical map (1 x n_features).
         * @param threshold  Threshold value.
         * @param adjacency  Adjacency matrix.
         * @param tail       Tail.
         * @param cluster_stat Method for computing cluster statistics.
         *
         * @return List of clusters.
         */
        static QList<Cluster> findClusters(
            const Eigen::RowVectorXd &stats,
            double threshold,
            const Eigen::SparseMatrix<double> &adjacency,
            int tail,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum);

        /**
         * @brief Finds spatio-temporal clusters in a thresholded map.
         *
         * @param stats_matrix Statistical map (n_channels x n_times).
         * @param threshold    Threshold value.
         * @param spatial_adjacency Spatial adjacency matrix.
         * @param temporal_adjacency Temporal adjacency matrix.
         * @param tail         Tail.
         * @param cluster_stat Method for computing cluster statistics.
         *
         * @return List of spatio-temporal clusters.
         */
        static QList<Cluster> findSpatioTemporalClusters(
            const Eigen::MatrixXd &stats_matrix,
            double threshold,
            const Eigen::SparseMatrix<double> &spatial_adjacency,
            const Eigen::SparseMatrix<double> &temporal_adjacency,
            int tail,
            ClusterStatistic cluster_stat = ClusterStatistic::Sum);

        /**
         * @brief Run one permutation step (sign flip).
         *
         * @param data      Original data.
         * @param threshold Threshold.
         * @param adjacency Adjacency.
         * @param tail      Tail.
         * @param cluster_stat Method for computing cluster statistics.
         * @param seed      Random seed.
         *
         * @return Max cluster statistic (magnitude).
         */
        static double runPermutationStep(
            const Eigen::MatrixXd &data,
            double threshold,
            const Eigen::SparseMatrix<double> &adjacency,
            int tail,
            ClusterStatistic cluster_stat,
            unsigned int seed);

        /**
         * @brief Run one permutation step for two-sample test.
         *
         * @param data1     Group 1 data.
         * @param data2     Group 2 data.
         * @param threshold Threshold.
         * @param adjacency Adjacency.
         * @param tail      Tail.
         * @param cluster_stat Method for computing cluster statistics.
         * @param seed      Random seed.
         *
         * @return Max cluster statistic (magnitude).
         */
        static double runTwoSamplePermutationStep(
            const Eigen::MatrixXd &data1,
            const Eigen::MatrixXd &data2,
            double threshold,
            const Eigen::SparseMatrix<double> &adjacency,
            int tail,
            ClusterStatistic cluster_stat,
            unsigned int seed);

        /**
         * @brief Run one spatio-temporal permutation step.
         *
         * @param data      Original data (list of matrices).
         * @param threshold Threshold.
         * @param spatial_adjacency Spatial adjacency.
         * @param temporal_adjacency Temporal adjacency.
         * @param tail      Tail.
         * @param cluster_stat Method for computing cluster statistics.
         * @param test_type Type of statistical test.
         * @param seed      Random seed.
         *
         * @return Max cluster statistic (magnitude).
         */
        static double runSpatioTemporalPermutationStep(
            const QList<Eigen::MatrixXd> &data,
            double threshold,
            const Eigen::SparseMatrix<double> &spatial_adjacency,
            const Eigen::SparseMatrix<double> &temporal_adjacency,
            int tail,
            ClusterStatistic cluster_stat,
            TestType test_type,
            unsigned int seed);

        /**
         * @brief Compute cluster statistic based on method.
         *
         * @param stats     Statistical values in cluster.
         * @param method    Method for computing cluster statistic.
         *
         * @return Cluster statistic value.
         */
        static double computeClusterStatistic(
            const QList<double> &stats,
            ClusterStatistic method);

        /**
         * @brief Convert 2D indices to linear index.
         *
         * @param row       Row index.
         * @param col       Column index.
         * @param n_cols    Number of columns.
         *
         * @return Linear index.
         */
        static int toLinearIndex(int row, int col, int n_cols);

        /**
         * @brief Convert linear index to 2D indices.
         *
         * @param linear_idx Linear index.
         * @param n_cols     Number of columns.
         *
         * @return Pair of (row, col) indices.
         */
        static std::pair<int, int> to2DIndex(int linear_idx, int n_cols);
    };

} // NAMESPACE

#endif // CLUSTERPERMUTATION_H
