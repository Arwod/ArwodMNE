#include <QtTest/QtTest>
#include <stats/statistics.h>
#include <stats/clusterpermutation.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>

using namespace STATSLIB;
using namespace Eigen;

class TestStats : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testTTest1Samp();
    void testTTestIndep();
    void testCluster1D();
    void testCluster2D();
};

void TestStats::initTestCase()
{
    srand(42);
}

void TestStats::cleanupTestCase()
{
}

void TestStats::testTTest1Samp()
{
    // 5 samples, 1 feature
    MatrixXd data(5, 1);
    data << 1, 2, 3, 4, 5; // Mean = 3, Var = 2.5, Std = 1.58, SEM = 1.58/sqrt(5) = 0.707
    // t = 3 / 0.707 = 4.24

    RowVectorXd t = Statistics::ttest1Samp(data);
    QCOMPARE(t.size(), 1);
    QVERIFY(std::abs(t(0) - 4.2426) < 0.01);

    // Constant data (Var=0)
    data.setConstant(3.0);
    t = Statistics::ttest1Samp(data);
    // t should be inf? No, we set to 0.0 if not finite.
    // Wait, mean=3, sem=0. t = inf. Statistics.cpp handles this.
    // Let's check implementation.
    // RowVectorXd t_values = mean.array() / sem.array();
    // Then check isfinite.
    // 3/0 is inf. isfinite(inf) is false. returns 0.0.
    QCOMPARE(t(0), 0.0);
}

void TestStats::testTTestIndep()
{
    // Group 1: 1, 2, 3 (Mean=2, Var=1, n=3)
    MatrixXd g1(3, 1);
    g1 << 1, 2, 3;

    // Group 2: 4, 5, 6 (Mean=5, Var=1, n=3)
    MatrixXd g2(3, 1);
    g2 << 4, 5, 6;

    // Pooled Var = ((2*1) + (2*1)) / 4 = 1.
    // SE = sqrt(1 * (1/3 + 1/3)) = sqrt(2/3) = 0.816
    // t = (2 - 5) / 0.816 = -3 / 0.816 = -3.67

    RowVectorXd t = Statistics::ttestIndep(g1, g2);
    QVERIFY(std::abs(t(0) + 3.6742) < 0.01);
}

void TestStats::testCluster1D()
{
    // Simulate 1D signal with a bump
    int n_samples = 10;
    int n_times = 20;
    MatrixXd data(n_samples, n_times);
    data.setZero();

    // Add noise
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < n_times; ++j)
        {
            data(i, j) = ((double)rand() / RAND_MAX) - 0.5; // [-0.5, 0.5]
        }
    }

    // Add signal at 5-10
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 5; j <= 10; ++j)
        {
            data(i, j) += 2.0; // Strong signal
        }
    }

    // Threshold = 2.0 (approx p < 0.05 for df=9)
    // Permutation test
    QList<ClusterPermutation::Cluster> clusters = ClusterPermutation::permutationClusterOneSampleTest(
        data, 2.0, 100);

    // Should find one significant cluster
    QVERIFY(!clusters.isEmpty());
    QVERIFY(clusters[0].pValue < 0.05);

    // Check indices
    bool hasSignalIndex = false;
    for (int idx : clusters[0].indices)
    {
        if (idx >= 5 && idx <= 10)
            hasSignalIndex = true;
    }
    QVERIFY(hasSignalIndex);
}

void TestStats::testCluster2D()
{
    // 2D Spatial Grid 3x3 (9 features)
    // Adjacency: grid connections
    /*
       0 - 1 - 2
       |   |   |
       3 - 4 - 5
       |   |   |
       6 - 7 - 8
    */
    int n_features = 9;
    SparseMatrix<double> adj(n_features, n_features);
    std::vector<Triplet<double>> trips;

    auto add_edge = [&](int u, int v)
    {
        trips.push_back(Triplet<double>(u, v, 1.0));
        trips.push_back(Triplet<double>(v, u, 1.0));
    };

    // Horizontal
    add_edge(0, 1);
    add_edge(1, 2);
    add_edge(3, 4);
    add_edge(4, 5);
    add_edge(6, 7);
    add_edge(7, 8);
    // Vertical
    add_edge(0, 3);
    add_edge(3, 6);
    add_edge(1, 4);
    add_edge(4, 7);
    add_edge(2, 5);
    add_edge(5, 8);

    adj.setFromTriplets(trips.begin(), trips.end());

    // Data: 10 subjects
    int n_samples = 10;
    MatrixXd data(n_samples, n_features);
    data.setZero();

    // Add signal to top-left corner (0, 1, 3, 4)
    for (int i = 0; i < n_samples; ++i)
    {
        data(i, 0) += 3.0;
        data(i, 1) += 3.0;
        data(i, 3) += 3.0;
        data(i, 4) += 3.0;
        // Noise elsewhere
        for (int j = 0; j < n_features; ++j)
        {
            data(i, j) += ((double)rand() / RAND_MAX - 0.5);
        }
    }

    QList<ClusterPermutation::Cluster> clusters = ClusterPermutation::permutationClusterOneSampleTest(
        data, 2.0, 100, adj);

    QVERIFY(!clusters.isEmpty());
    ClusterPermutation::Cluster top = clusters[0];

    // Top cluster should contain 0, 1, 3, 4
    QVERIFY(top.indices.contains(0));
    QVERIFY(top.indices.contains(1));
    QVERIFY(top.indices.contains(3));
    QVERIFY(top.indices.contains(4));
    QVERIFY(top.pValue < 0.05);
}

QTEST_GUILESS_MAIN(TestStats)
#include "test_stats.moc"
