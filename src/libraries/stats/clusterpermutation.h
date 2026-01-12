#ifndef CLUSTERPERMUTATION_H
#define CLUSTERPERMUTATION_H

#include "stats_global.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <QList>
#include <QSharedPointer>

namespace STATSLIB
{

    class STATSSHARED_EXPORT ClusterPermutation
    {
    public:
        struct Cluster
        {
            QList<int> indices; // Indices of features in the cluster
            double clusterStat; // Statistic (e.g. sum of t-values)
            double pValue;      // P-value

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
         *
         * @return List of clusters found.
         */
        static QList<Cluster> permutationClusterOneSampleTest(
            const Eigen::MatrixXd &data,
            double threshold,
            int n_permutations = 1024,
            const Eigen::SparseMatrix<double> &adjacency = Eigen::SparseMatrix<double>(),
            int tail = 0);

    private:
        /**
         * @brief Finds clusters in a thresholded map.
         *
         * @param stats      Statistical map (1 x n_features).
         * @param threshold  Threshold value.
         * @param adjacency  Adjacency matrix.
         * @param tail       Tail.
         *
         * @return List of clusters.
         */
        static QList<Cluster> findClusters(
            const Eigen::RowVectorXd &stats,
            double threshold,
            const Eigen::SparseMatrix<double> &adjacency,
            int tail);

        /**
         * @brief Run one permutation step (sign flip).
         *
         * @param data      Original data.
         * @param threshold Threshold.
         * @param adjacency Adjacency.
         * @param tail      Tail.
         * @param seed      Random seed.
         *
         * @return Max cluster statistic (magnitude).
         */
        static double runPermutationStep(
            const Eigen::MatrixXd &data,
            double threshold,
            const Eigen::SparseMatrix<double> &adjacency,
            int tail,
            unsigned int seed);
    };

} // NAMESPACE

#endif // CLUSTERPERMUTATION_H
