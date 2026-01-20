#include "clusterpermutation.h"
#include "statistics.h"
#include <QtConcurrent/QtConcurrent>
#include <QRandomGenerator>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace Eigen;

namespace STATSLIB
{

    QList<ClusterPermutation::Cluster> ClusterPermutation::permutationClusterOneSampleTest(
        const Eigen::MatrixXd &data,
        double threshold,
        int n_permutations,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        unsigned int seed)
    {
        // Set random seed if provided
        if (seed != 0) {
            QRandomGenerator::global()->seed(seed);
        }

        // 1. Compute original T-values
        RowVectorXd t_obs = Statistics::ttest1Samp(data);

        // 2. Find original clusters
        QList<Cluster> clusters = findClusters(t_obs, threshold, adjacency, tail, cluster_stat);

        if (clusters.isEmpty())
        {
            return clusters;
        }

        // 3. Permutation Loop (Parallel)
        QVector<int> seeds(n_permutations);
        for (int i = 0; i < n_permutations; ++i)
            seeds[i] = i;

        std::function<double(int)> permuteFunc = [&](int seedOffset) -> double
        {
            unsigned int permSeed = (seed != 0) ? seed + seedOffset : QRandomGenerator::global()->generate() + seedOffset;
            return runPermutationStep(data, threshold, adjacency, tail, cluster_stat, permSeed);
        };

        QFuture<double> future = QtConcurrent::mapped(seeds, permuteFunc);
        future.waitForFinished();

        QList<double> h0_distribution = future.results();

        // 4. Compute P-values
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

            cluster.pValue = (double)(count + 1) / (n_permutations + 1);
        }

        // Sort clusters by statistic magnitude (descending)
        std::sort(clusters.begin(), clusters.end());

        return clusters;
    }

    QList<ClusterPermutation::Cluster> ClusterPermutation::permutationClusterTwoSampleTest(
        const Eigen::MatrixXd &data1,
        const Eigen::MatrixXd &data2,
        double threshold,
        int n_permutations,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        unsigned int seed)
    {
        // Set random seed if provided
        if (seed != 0) {
            QRandomGenerator::global()->seed(seed);
        }

        // 1. Compute original T-values for two-sample test
        RowVectorXd t_obs = Statistics::ttestIndep(data1, data2);

        // 2. Find original clusters
        QList<Cluster> clusters = findClusters(t_obs, threshold, adjacency, tail, cluster_stat);

        if (clusters.isEmpty())
        {
            return clusters;
        }

        // 3. Permutation Loop (Parallel)
        QVector<int> seeds(n_permutations);
        for (int i = 0; i < n_permutations; ++i)
            seeds[i] = i;

        std::function<double(int)> permuteFunc = [&](int seedOffset) -> double
        {
            unsigned int permSeed = (seed != 0) ? seed + seedOffset : QRandomGenerator::global()->generate() + seedOffset;
            return runTwoSamplePermutationStep(data1, data2, threshold, adjacency, tail, cluster_stat, permSeed);
        };

        QFuture<double> future = QtConcurrent::mapped(seeds, permuteFunc);
        future.waitForFinished();

        QList<double> h0_distribution = future.results();

        // 4. Compute P-values
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

            cluster.pValue = (double)(count + 1) / (n_permutations + 1);
        }

        // Sort clusters by statistic magnitude (descending)
        std::sort(clusters.begin(), clusters.end());

        return clusters;
    }
    QList<ClusterPermutation::Cluster> ClusterPermutation::spatioTemporalClusterTest(
        const QList<Eigen::MatrixXd> &data,
        double threshold,
        int n_permutations,
        const Eigen::SparseMatrix<double> &spatial_adjacency,
        const Eigen::SparseMatrix<double> &temporal_adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        TestType test_type,
        unsigned int seed)
    {
        // Set random seed if provided
        if (seed != 0) {
            QRandomGenerator::global()->seed(seed);
        }

        if (data.isEmpty()) {
            return QList<Cluster>();
        }

        int n_channels = data[0].rows();
        int n_times = data[0].cols();
        int n_samples = data.size();

        // 1. Compute original statistics for each channel-time point
        MatrixXd stats_matrix = MatrixXd::Zero(n_channels, n_times);

        if (test_type == TestType::OneSample) {
            // Compute one-sample t-test for each channel-time point
            for (int ch = 0; ch < n_channels; ++ch) {
                for (int t = 0; t < n_times; ++t) {
                    VectorXd values(n_samples);
                    for (int s = 0; s < n_samples; ++s) {
                        values(s) = data[s](ch, t);
                    }
                    
                    // Compute t-statistic
                    double mean = values.mean();
                    double std_dev = std::sqrt((values.array() - mean).square().sum() / (n_samples - 1));
                    if (std_dev > 1e-12) {
                        stats_matrix(ch, t) = mean * std::sqrt(n_samples) / std_dev;
                    }
                }
            }
        }

        // 2. Find original spatio-temporal clusters
        QList<Cluster> clusters = findSpatioTemporalClusters(stats_matrix, threshold, 
                                                            spatial_adjacency, temporal_adjacency, 
                                                            tail, cluster_stat);

        if (clusters.isEmpty()) {
            return clusters;
        }

        // 3. Permutation Loop (Parallel)
        QVector<int> seeds(n_permutations);
        for (int i = 0; i < n_permutations; ++i)
            seeds[i] = i;

        std::function<double(int)> permuteFunc = [&](int seedOffset) -> double
        {
            unsigned int permSeed = (seed != 0) ? seed + seedOffset : QRandomGenerator::global()->generate() + seedOffset;
            return runSpatioTemporalPermutationStep(data, threshold, spatial_adjacency, temporal_adjacency, 
                                                  tail, cluster_stat, test_type, permSeed);
        };

        QFuture<double> future = QtConcurrent::mapped(seeds, permuteFunc);
        future.waitForFinished();

        QList<double> h0_distribution = future.results();

        // 4. Compute P-values
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

            cluster.pValue = (double)(count + 1) / (n_permutations + 1);
        }

        // Sort clusters by statistic magnitude (descending)
        std::sort(clusters.begin(), clusters.end());

        return clusters;
    }

    QList<ClusterPermutation::Cluster> ClusterPermutation::spatioTemporalClusterTwoSampleTest(
        const QList<Eigen::MatrixXd> &data1,
        const QList<Eigen::MatrixXd> &data2,
        double threshold,
        int n_permutations,
        const Eigen::SparseMatrix<double> &spatial_adjacency,
        const Eigen::SparseMatrix<double> &temporal_adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        unsigned int seed)
    {
        // Set random seed if provided
        if (seed != 0) {
            QRandomGenerator::global()->seed(seed);
        }

        if (data1.isEmpty() || data2.isEmpty()) {
            return QList<Cluster>();
        }

        int n_channels = data1[0].rows();
        int n_times = data1[0].cols();
        int n_samples1 = data1.size();
        int n_samples2 = data2.size();

        // 1. Compute original two-sample t-statistics for each channel-time point
        MatrixXd stats_matrix = MatrixXd::Zero(n_channels, n_times);

        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                VectorXd values1(n_samples1);
                VectorXd values2(n_samples2);
                
                for (int s = 0; s < n_samples1; ++s) {
                    values1(s) = data1[s](ch, t);
                }
                for (int s = 0; s < n_samples2; ++s) {
                    values2(s) = data2[s](ch, t);
                }
                
                // Compute two-sample t-statistic
                double mean1 = values1.mean();
                double mean2 = values2.mean();
                double var1 = (values1.array() - mean1).square().sum() / (n_samples1 - 1);
                double var2 = (values2.array() - mean2).square().sum() / (n_samples2 - 1);
                double pooled_se = std::sqrt(var1 / n_samples1 + var2 / n_samples2);
                
                if (pooled_se > 1e-12) {
                    stats_matrix(ch, t) = (mean1 - mean2) / pooled_se;
                }
            }
        }

        // 2. Find original spatio-temporal clusters
        QList<Cluster> clusters = findSpatioTemporalClusters(stats_matrix, threshold, 
                                                            spatial_adjacency, temporal_adjacency, 
                                                            tail, cluster_stat);

        if (clusters.isEmpty()) {
            return clusters;
        }

        // 3. Permutation Loop (Parallel) - combine data for permutation
        QList<MatrixXd> combined_data = data1 + data2;
        
        QVector<int> seeds(n_permutations);
        for (int i = 0; i < n_permutations; ++i)
            seeds[i] = i;

        std::function<double(int)> permuteFunc = [&](int seedOffset) -> double
        {
            unsigned int permSeed = (seed != 0) ? seed + seedOffset : QRandomGenerator::global()->generate() + seedOffset;
            
            // Randomly permute group assignments
            QRandomGenerator gen(permSeed);
            QList<MatrixXd> shuffled_data = combined_data;
            
            for (int i = shuffled_data.size() - 1; i > 0; --i) {
                int j = gen.bounded(i + 1);
                shuffled_data.swapItemsAt(i, j);
            }
            
            // Split back into two groups
            QList<MatrixXd> perm_data1 = shuffled_data.mid(0, n_samples1);
            QList<MatrixXd> perm_data2 = shuffled_data.mid(n_samples1, n_samples2);
            
            // Compute permuted statistics
            MatrixXd perm_stats = MatrixXd::Zero(n_channels, n_times);
            for (int ch = 0; ch < n_channels; ++ch) {
                for (int t = 0; t < n_times; ++t) {
                    VectorXd perm_values1(n_samples1);
                    VectorXd perm_values2(n_samples2);
                    
                    for (int s = 0; s < n_samples1; ++s) {
                        perm_values1(s) = perm_data1[s](ch, t);
                    }
                    for (int s = 0; s < n_samples2; ++s) {
                        perm_values2(s) = perm_data2[s](ch, t);
                    }
                    
                    double mean1 = perm_values1.mean();
                    double mean2 = perm_values2.mean();
                    double var1 = (perm_values1.array() - mean1).square().sum() / (n_samples1 - 1);
                    double var2 = (perm_values2.array() - mean2).square().sum() / (n_samples2 - 1);
                    double pooled_se = std::sqrt(var1 / n_samples1 + var2 / n_samples2);
                    
                    if (pooled_se > 1e-12) {
                        perm_stats(ch, t) = (mean1 - mean2) / pooled_se;
                    }
                }
            }
            
            // Find clusters in permuted data
            QList<Cluster> perm_clusters = findSpatioTemporalClusters(perm_stats, threshold, 
                                                                     spatial_adjacency, temporal_adjacency, 
                                                                     tail, cluster_stat);
            
            double maxStat = 0.0;
            for (const Cluster &c : perm_clusters) {
                double absStat = std::abs(c.clusterStat);
                if (absStat > maxStat) {
                    maxStat = absStat;
                }
            }
            
            return maxStat;
        };

        QFuture<double> future = QtConcurrent::mapped(seeds, permuteFunc);
        future.waitForFinished();

        QList<double> h0_distribution = future.results();

        // 4. Compute P-values
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

            cluster.pValue = (double)(count + 1) / (n_permutations + 1);
        }

        // Sort clusters by statistic magnitude (descending)
        std::sort(clusters.begin(), clusters.end());

        return clusters;
    }
    Eigen::SparseMatrix<double> ClusterPermutation::createLinearAdjacency(int n_features)
    {
        SparseMatrix<double> adjacency(n_features, n_features);
        std::vector<Triplet<double>> triplets;
        
        for (int i = 0; i < n_features; ++i) {
            if (i > 0) {
                triplets.push_back(Triplet<double>(i, i-1, 1.0));
            }
            if (i < n_features - 1) {
                triplets.push_back(Triplet<double>(i, i+1, 1.0));
            }
        }
        
        adjacency.setFromTriplets(triplets.begin(), triplets.end());
        return adjacency;
    }

    Eigen::SparseMatrix<double> ClusterPermutation::createGridAdjacency(int n_rows, int n_cols, bool connect_diag)
    {
        int n_features = n_rows * n_cols;
        SparseMatrix<double> adjacency(n_features, n_features);
        std::vector<Triplet<double>> triplets;
        
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                int idx = toLinearIndex(r, c, n_cols);
                
                // 4-connectivity (up, down, left, right)
                if (r > 0) {
                    int neighbor = toLinearIndex(r-1, c, n_cols);
                    triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                }
                if (r < n_rows - 1) {
                    int neighbor = toLinearIndex(r+1, c, n_cols);
                    triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                }
                if (c > 0) {
                    int neighbor = toLinearIndex(r, c-1, n_cols);
                    triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                }
                if (c < n_cols - 1) {
                    int neighbor = toLinearIndex(r, c+1, n_cols);
                    triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                }
                
                // 8-connectivity (diagonal neighbors)
                if (connect_diag) {
                    if (r > 0 && c > 0) {
                        int neighbor = toLinearIndex(r-1, c-1, n_cols);
                        triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                    }
                    if (r > 0 && c < n_cols - 1) {
                        int neighbor = toLinearIndex(r-1, c+1, n_cols);
                        triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                    }
                    if (r < n_rows - 1 && c > 0) {
                        int neighbor = toLinearIndex(r+1, c-1, n_cols);
                        triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                    }
                    if (r < n_rows - 1 && c < n_cols - 1) {
                        int neighbor = toLinearIndex(r+1, c+1, n_cols);
                        triplets.push_back(Triplet<double>(idx, neighbor, 1.0));
                    }
                }
            }
        }
        
        adjacency.setFromTriplets(triplets.begin(), triplets.end());
        return adjacency;
    }
    QList<ClusterPermutation::Cluster> ClusterPermutation::findClusters(
        const Eigen::RowVectorXd &stats,
        double threshold,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail,
        ClusterStatistic cluster_stat)
    {
        QList<Cluster> foundClusters;
        int n_features = stats.size();

        // Determine candidates based on tail
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
            std::vector<bool> isCandidate(n_features, false);
            for (int idx : candidates)
                isCandidate[idx] = true;

            for (int idx : candidates)
            {
                if (visited[idx])
                    continue;

                // New Cluster
                Cluster currentCluster;
                QList<double> clusterStats;

                std::queue<int> q;
                q.push(idx);
                visited[idx] = true;

                while (!q.empty())
                {
                    int curr = q.front();
                    q.pop();

                    currentCluster.indices.append(curr);
                    clusterStats.append(stats(curr));

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
                        for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency, curr); it; ++it)
                        {
                            int neighbor = it.row();
                            if (neighbor < n_features && isCandidate[neighbor] && !visited[neighbor])
                            {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }

                // Compute cluster statistic
                currentCluster.clusterStat = computeClusterStatistic(clusterStats, cluster_stat);
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

    QList<ClusterPermutation::Cluster> ClusterPermutation::findSpatioTemporalClusters(
        const Eigen::MatrixXd &stats_matrix,
        double threshold,
        const Eigen::SparseMatrix<double> &spatial_adjacency,
        const Eigen::SparseMatrix<double> &temporal_adjacency,
        int tail,
        ClusterStatistic cluster_stat)
    {
        QList<Cluster> foundClusters;
        int n_channels = stats_matrix.rows();
        int n_times = stats_matrix.cols();

        // Create temporal adjacency if not provided
        SparseMatrix<double> temp_adj = temporal_adjacency;
        if (temp_adj.rows() == 0) {
            temp_adj = createLinearAdjacency(n_times);
        }

        // Determine candidates based on tail
        auto search = [&](bool checkPositive)
        {
            std::vector<std::pair<int, int>> candidates;
            candidates.reserve(n_channels * n_times);

            for (int ch = 0; ch < n_channels; ++ch)
            {
                for (int t = 0; t < n_times; ++t)
                {
                    if (checkPositive)
                    {
                        if (stats_matrix(ch, t) > threshold)
                            candidates.push_back({ch, t});
                    }
                    else
                    {
                        if (stats_matrix(ch, t) < -threshold)
                            candidates.push_back({ch, t});
                    }
                }
            }

            if (candidates.empty())
                return;

            // Clustering using BFS in spatio-temporal space
            std::unordered_set<int> visited;
            std::unordered_set<int> candidateSet;
            
            for (const auto& candidate : candidates) {
                int linear_idx = toLinearIndex(candidate.first, candidate.second, n_times);
                candidateSet.insert(linear_idx);
            }

            for (const auto& candidate : candidates)
            {
                int ch = candidate.first;
                int t = candidate.second;
                int linear_idx = toLinearIndex(ch, t, n_times);
                
                if (visited.find(linear_idx) != visited.end())
                    continue;

                // New Cluster
                Cluster currentCluster;
                QList<double> clusterStats;

                std::queue<std::pair<int, int>> q;
                q.push({ch, t});
                visited.insert(linear_idx);

                while (!q.empty())
                {
                    auto curr = q.front();
                    q.pop();
                    
                    int curr_ch = curr.first;
                    int curr_t = curr.second;
                    int curr_linear = toLinearIndex(curr_ch, curr_t, n_times);

                    currentCluster.indices.append(curr_linear);
                    currentCluster.timeIndices.append(curr_t);
                    clusterStats.append(stats_matrix(curr_ch, curr_t));

                    // Update time range
                    if (currentCluster.timeStart == -1 || curr_t < currentCluster.timeStart) {
                        currentCluster.timeStart = curr_t;
                    }
                    if (currentCluster.timeEnd == -1 || curr_t > currentCluster.timeEnd) {
                        currentCluster.timeEnd = curr_t;
                    }

                    // Find spatial neighbors
                    if (spatial_adjacency.rows() > 0) {
                        for (SparseMatrix<double>::InnerIterator it(spatial_adjacency, curr_ch); it; ++it) {
                            int neighbor_ch = it.row();
                            int neighbor_linear = toLinearIndex(neighbor_ch, curr_t, n_times);
                            
                            if (candidateSet.find(neighbor_linear) != candidateSet.end() && 
                                visited.find(neighbor_linear) == visited.end()) {
                                visited.insert(neighbor_linear);
                                q.push({neighbor_ch, curr_t});
                            }
                        }
                    }

                    // Find temporal neighbors
                    for (SparseMatrix<double>::InnerIterator it(temp_adj, curr_t); it; ++it) {
                        int neighbor_t = it.row();
                        int neighbor_linear = toLinearIndex(curr_ch, neighbor_t, n_times);
                        
                        if (candidateSet.find(neighbor_linear) != candidateSet.end() && 
                            visited.find(neighbor_linear) == visited.end()) {
                            visited.insert(neighbor_linear);
                            q.push({curr_ch, neighbor_t});
                        }
                    }
                }

                // Compute cluster statistic
                currentCluster.clusterStat = computeClusterStatistic(clusterStats, cluster_stat);
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
        ClusterStatistic cluster_stat,
        unsigned int seed)
    {
        // Sign flipping for one-sample test
        int n_samples = data.rows();
        MatrixXd permData = data;
        QRandomGenerator gen(seed);

        for (int i = 0; i < n_samples; ++i)
        {
            if (gen.generate() % 2 == 0)
            {
                permData.row(i) *= -1.0;
            }
        }

        RowVectorXd t_perm = Statistics::ttest1Samp(permData);
        QList<Cluster> clusters = findClusters(t_perm, threshold, adjacency, tail, cluster_stat);

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

    double ClusterPermutation::runTwoSamplePermutationStep(
        const Eigen::MatrixXd &data1,
        const Eigen::MatrixXd &data2,
        double threshold,
        const Eigen::SparseMatrix<double> &adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        unsigned int seed)
    {
        // Combine data and randomly reassign group labels
        int n_samples1 = data1.rows();
        int n_samples2 = data2.rows();
        int total_samples = n_samples1 + n_samples2;
        
        MatrixXd combined_data(total_samples, data1.cols());
        combined_data.topRows(n_samples1) = data1;
        combined_data.bottomRows(n_samples2) = data2;
        
        // Shuffle rows
        QRandomGenerator gen(seed);
        std::vector<int> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        for (int i = total_samples - 1; i > 0; --i) {
            int j = gen.bounded(i + 1);
            std::swap(indices[i], indices[j]);
        }
        
        // Create permuted groups
        MatrixXd perm_data1(n_samples1, data1.cols());
        MatrixXd perm_data2(n_samples2, data2.cols());
        
        for (int i = 0; i < n_samples1; ++i) {
            perm_data1.row(i) = combined_data.row(indices[i]);
        }
        for (int i = 0; i < n_samples2; ++i) {
            perm_data2.row(i) = combined_data.row(indices[n_samples1 + i]);
        }
        
        RowVectorXd t_perm = Statistics::ttestIndep(perm_data1, perm_data2);
        QList<Cluster> clusters = findClusters(t_perm, threshold, adjacency, tail, cluster_stat);

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

    double ClusterPermutation::runSpatioTemporalPermutationStep(
        const QList<Eigen::MatrixXd> &data,
        double threshold,
        const Eigen::SparseMatrix<double> &spatial_adjacency,
        const Eigen::SparseMatrix<double> &temporal_adjacency,
        int tail,
        ClusterStatistic cluster_stat,
        TestType test_type,
        unsigned int seed)
    {
        if (data.isEmpty()) {
            return 0.0;
        }

        int n_channels = data[0].rows();
        int n_times = data[0].cols();
        int n_samples = data.size();

        // Create permuted data by sign flipping for one-sample test
        QList<MatrixXd> permData = data;
        QRandomGenerator gen(seed);

        if (test_type == TestType::OneSample) {
            for (int s = 0; s < n_samples; ++s) {
                if (gen.generate() % 2 == 0) {
                    permData[s] *= -1.0;
                }
            }
        }

        // Compute permuted statistics
        MatrixXd perm_stats = MatrixXd::Zero(n_channels, n_times);
        
        for (int ch = 0; ch < n_channels; ++ch) {
            for (int t = 0; t < n_times; ++t) {
                VectorXd values(n_samples);
                for (int s = 0; s < n_samples; ++s) {
                    values(s) = permData[s](ch, t);
                }
                
                // Compute t-statistic
                double mean = values.mean();
                double std_dev = std::sqrt((values.array() - mean).square().sum() / (n_samples - 1));
                if (std_dev > 1e-12) {
                    perm_stats(ch, t) = mean * std::sqrt(n_samples) / std_dev;
                }
            }
        }

        QList<Cluster> clusters = findSpatioTemporalClusters(perm_stats, threshold, 
                                                           spatial_adjacency, temporal_adjacency, 
                                                           tail, cluster_stat);

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
    double ClusterPermutation::computeClusterStatistic(
        const QList<double> &stats,
        ClusterStatistic method)
    {
        if (stats.isEmpty()) {
            return 0.0;
        }

        switch (method) {
            case ClusterStatistic::Sum:
                return std::accumulate(stats.begin(), stats.end(), 0.0);
                
            case ClusterStatistic::Max:
                return *std::max_element(stats.begin(), stats.end(), 
                    [](double a, double b) { return std::abs(a) < std::abs(b); });
                
            case ClusterStatistic::Mean:
                return std::accumulate(stats.begin(), stats.end(), 0.0) / stats.size();
                
            case ClusterStatistic::Size:
                return static_cast<double>(stats.size());
                
            default:
                return std::accumulate(stats.begin(), stats.end(), 0.0);
        }
    }

    int ClusterPermutation::toLinearIndex(int row, int col, int n_cols)
    {
        return row * n_cols + col;
    }

    std::pair<int, int> ClusterPermutation::to2DIndex(int linear_idx, int n_cols)
    {
        int row = linear_idx / n_cols;
        int col = linear_idx % n_cols;
        return {row, col};
    }

} // NAMESPACE