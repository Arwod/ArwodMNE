#include "clusterpermutation.h"
#include "statistics.h"
#include <QtConcurrent/QtConcurrent>
#include <QRandomGenerator>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace Eigen;

namespace STATSLIB
{

    QList<ClusterPermutation::Cluster> ClusterPermutation::permutationClusterOneSampleTest(
        const Eigen::MatrixXd &data,
        double threshold,
        int n_permutations,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail)
    {
        // 1. Compute original T-values
        RowVectorXd t_obs = Statistics::ttest1Samp(data);

        // 2. Find original clusters
        QList<Cluster> clusters = findClusters(t_obs, threshold, adjacency, tail);

        if (clusters.isEmpty())
        {
            return clusters;
        }

        // 3. Permutation Loop (Parallel)
        // Create a vector of seeds or indices
        QVector<int> seeds(n_permutations);
        for (int i = 0; i < n_permutations; ++i)
            seeds[i] = i; // Will use as offset for random

        // Use QThreadPool::globalInstance()->maxThreadCount() implicitly by QtConcurrent

        std::function<double(int)> permuteFunc = [&](int seedOffset) -> double
        {
            // Use a unique seed per task
            unsigned int seed = QRandomGenerator::global()->generate() + seedOffset;
            return runPermutationStep(data, threshold, adjacency, tail, seed);
        };

        QFuture<double> future = QtConcurrent::mapped(seeds, permuteFunc);
        future.waitForFinished();

        QList<double> h0_distribution = future.results();

        // 4. Compute P-values
        // Sort H0 for faster search or just iterate
        // For 2-tailed, we use absolute values in H0 (already done in runPermutationStep)
        // and compare with absolute cluster stats.

        std::sort(h0_distribution.begin(), h0_distribution.end());

        for (Cluster &cluster : clusters)
        {
            double absStat = std::abs(cluster.clusterStat);
            int count = 0;
            for (double val : h0_distribution)
            {
                if (val >= absStat)
                {
                    count++;
                }
            }
            // If H0 is sorted ascending, we can count from end.
            // But the loop above is fine for N=1024.

            cluster.pValue = (double)(count + 1) / (n_permutations + 1);
        }

        // Sort clusters by p-value (ascending) or stat (descending)
        std::sort(clusters.begin(), clusters.end(), [](const Cluster &a, const Cluster &b)
                  { return std::abs(a.clusterStat) > std::abs(b.clusterStat); });

        return clusters;
    }

    QList<ClusterPermutation::Cluster> ClusterPermutation::findClusters(
        const Eigen::RowVectorXd &stats,
        double threshold,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail)
    {
        QList<Cluster> foundClusters;
        int n_features = stats.size();

        // Determine candidates based on tail
        // We handle positive and negative clusters separately to ensure sign consistency

        auto search = [&](bool checkPositive)
        {
            std::vector<int> candidates;
            candidates.reserve(n_features);

            for (int i = 0; i < n_features; ++i)
            {
                if (checkPositive)
                {
                    if (stats(i) > threshold)
                        candidates.push_back(i);
                }
                else
                {
                    if (stats(i) < -threshold)
                        candidates.push_back(i);
                }
            }

            if (candidates.empty())
                return;

            // Clustering using BFS
            std::vector<bool> visited(n_features, false);
            // Mark non-candidates as visited so we don't process them
            // Actually, better to use a set or just check presence in candidates.
            // But since we iterate candidates, we just need to track visited among candidates.
            // To do efficient lookup, we can use a boolean mask of candidates.
            std::vector<bool> isCandidate(n_features, false);
            for (int idx : candidates)
                isCandidate[idx] = true;

            for (int idx : candidates)
            {
                if (visited[idx])
                    continue;

                // New Cluster
                Cluster currentCluster;
                currentCluster.clusterStat = 0.0;
                currentCluster.pValue = 1.0; // Default

                std::queue<int> q;
                q.push(idx);
                visited[idx] = true;

                while (!q.empty())
                {
                    int curr = q.front();
                    q.pop();

                    currentCluster.indices.append(curr);
                    currentCluster.clusterStat += stats(curr);

                    // Find neighbors
                    if (adjacency.rows() == 0)
                    {
                        // 1D Linear Adjacency
                        int neighbors[] = {curr - 1, curr + 1};
                        for (int neighbor : neighbors)
                        {
                            if (neighbor >= 0 && neighbor < n_features)
                            {
                                if (isCandidate[neighbor] && !visited[neighbor])
                                {
                                    visited[neighbor] = true;
                                    q.push(neighbor);
                                }
                            }
                        }
                    }
                    else
                    {
                        // Sparse Adjacency
                        // Iterate non-zero column elements
                        // Adjacency is assumed symmetric? Or we check both ways?
                        // Usually adjacency is symmetric.
                        // If row-major/col-major?
                        // Eigen SparseMatrix is ColMajor by default.
                        // Iterate over column `curr` gives outgoing edges?
                        // For symmetric, it's neighbors.

                        for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency, curr); it; ++it)
                        {
                            int neighbor = it.row(); // row index is the neighbor
                            // Check if neighbor is candidate and not visited
                            if (neighbor < n_features && isCandidate[neighbor] && !visited[neighbor])
                            {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }

                foundClusters.append(currentCluster);
            }
        };

        if (tail >= 0)
        {
            search(true); // Positive clusters
        }
        if (tail <= 0)
        {
            search(false); // Negative clusters
        }

        return foundClusters;
    }

    double ClusterPermutation::runPermutationStep(
        const Eigen::MatrixXd &data,
        double threshold,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail,
        unsigned int seed)
    {
        // Sign flipping
        int n_samples = data.rows();
        int n_features = data.cols();

        // We can modify data in place if we copy it, but that's expensive.
        // Instead, compute mean/var with sign flip on the fly?
        // Statistics::ttest1Samp takes Matrix.
        // Let's create a copy with flipped signs.
        // To save memory, we might want to optimize t-test to take signs vector.
        // But for now, let's copy. MatrixXd copy is contiguous, reasonably fast for moderate data.
        // For large data (e.g. 20 subjects x 20000 vertices), it's 3MB. Fast.

        MatrixXd permData = data;
        QRandomGenerator gen(seed);

        for (int i = 0; i < n_samples; ++i)
        {
            if (gen.generate() % 2 == 0)
            { // 50% chance
                permData.row(i) *= -1.0;
            }
        }

        RowVectorXd t_perm = Statistics::ttest1Samp(permData);

        QList<Cluster> clusters = findClusters(t_perm, threshold, adjacency, tail);

        double maxStat = 0.0;
        for (const Cluster &c : clusters)
        {
            double absStat = std::abs(c.clusterStat);
            if (absStat > maxStat)
            {
                maxStat = absStat;
            }
        }

        return maxStat;
    }

} // NAMESPACE
