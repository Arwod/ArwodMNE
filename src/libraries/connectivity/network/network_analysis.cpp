//=============================================================================================================
/**
 * @file     network_analysis.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Massachusetts General Hospital nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MASSACHUSETTS GENERAL HOSPITAL BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    NetworkAnalysis class definition.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "network_analysis.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QtMath>

//=============================================================================================================
// STD INCLUDES
//=============================================================================================================

#include <algorithm>
#include <queue>
#include <stack>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace CONNECTIVITYLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

NetworkAnalysis::NetworkAnalysis()
{
}

//=============================================================================================================

VectorXd NetworkAnalysis::computeDegree(const MatrixXd& adjacencyMatrix,
                                         double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    
    // Remove self-connections
    binaryMatrix.diagonal().setZero();
    
    // Compute degree as sum of connections
    return binaryMatrix.rowwise().sum();
}

//=============================================================================================================

VectorXd NetworkAnalysis::computeStrength(const MatrixXd& adjacencyMatrix)
{
    MatrixXd weightedMatrix = adjacencyMatrix;
    
    // Remove self-connections
    weightedMatrix.diagonal().setZero();
    
    // Compute strength as sum of edge weights
    return weightedMatrix.rowwise().sum();
}

//=============================================================================================================

VectorXd NetworkAnalysis::computeClusteringCoefficient(const MatrixXd& adjacencyMatrix,
                                                        double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    binaryMatrix.diagonal().setZero();
    
    int nNodes = binaryMatrix.rows();
    VectorXd clustering(nNodes);
    
    for (int i = 0; i < nNodes; ++i) {
        // Find neighbors of node i
        std::vector<int> neighbors;
        for (int j = 0; j < nNodes; ++j) {
            if (binaryMatrix(i, j) > 0) {
                neighbors.push_back(j);
            }
        }
        
        int degree = neighbors.size();
        if (degree < 2) {
            clustering[i] = 0.0;
            continue;
        }
        
        // Count triangles
        int triangles = 0;
        for (size_t j = 0; j < neighbors.size(); ++j) {
            for (size_t k = j + 1; k < neighbors.size(); ++k) {
                if (binaryMatrix(neighbors[j], neighbors[k]) > 0) {
                    triangles++;
                }
            }
        }
        
        // Clustering coefficient = 2 * triangles / (degree * (degree - 1))
        clustering[i] = 2.0 * triangles / (degree * (degree - 1));
    }
    
    return clustering;
}
//=============================================================================================================

VectorXd NetworkAnalysis::computeBetweennessCentrality(const MatrixXd& adjacencyMatrix,
                                                        double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    binaryMatrix.diagonal().setZero();
    
    int nNodes = binaryMatrix.rows();
    VectorXd betweenness = VectorXd::Zero(nNodes);
    
    // Brandes algorithm for betweenness centrality
    for (int s = 0; s < nNodes; ++s) {
        std::stack<int> S;
        std::vector<std::vector<int>> P(nNodes);
        VectorXd sigma = VectorXd::Zero(nNodes);
        VectorXd d = VectorXd::Constant(nNodes, -1);
        VectorXd delta = VectorXd::Zero(nNodes);
        
        sigma[s] = 1.0;
        d[s] = 0.0;
        
        std::queue<int> Q;
        Q.push(s);
        
        while (!Q.empty()) {
            int v = Q.front();
            Q.pop();
            S.push(v);
            
            for (int w = 0; w < nNodes; ++w) {
                if (binaryMatrix(v, w) > 0) {
                    // First time we found shortest path to w?
                    if (d[w] < 0) {
                        Q.push(w);
                        d[w] = d[v] + 1;
                    }
                    // Shortest path to w via v?
                    if (d[w] == d[v] + 1) {
                        sigma[w] += sigma[v];
                        P[w].push_back(v);
                    }
                }
            }
        }
        
        // Accumulation
        while (!S.empty()) {
            int w = S.top();
            S.pop();
            
            for (int v : P[w]) {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            
            if (w != s) {
                betweenness[w] += delta[w];
            }
        }
    }
    
    // Normalize
    double normFactor = 2.0 / ((nNodes - 1) * (nNodes - 2));
    return betweenness * normFactor;
}

//=============================================================================================================

VectorXd NetworkAnalysis::computeEigenvectorCentrality(const MatrixXd& adjacencyMatrix,
                                                        int maxIterations,
                                                        double tolerance)
{
    MatrixXd A = adjacencyMatrix;
    A.diagonal().setZero();
    
    int nNodes = A.rows();
    VectorXd centrality = VectorXd::Random(nNodes).cwiseAbs();
    centrality /= centrality.norm();
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        VectorXd newCentrality = A * centrality;
        
        if (newCentrality.norm() > 0) {
            newCentrality /= newCentrality.norm();
        }
        
        if ((newCentrality - centrality).norm() < tolerance) {
            break;
        }
        
        centrality = newCentrality;
    }
    
    return centrality;
}

//=============================================================================================================

MatrixXd NetworkAnalysis::computeShortestPaths(const MatrixXd& adjacencyMatrix,
                                                double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    return floydWarshall(binaryMatrix);
}

//=============================================================================================================

double NetworkAnalysis::computeGlobalEfficiency(const MatrixXd& adjacencyMatrix,
                                                 double threshold)
{
    MatrixXd distances = computeShortestPaths(adjacencyMatrix, threshold);
    int nNodes = distances.rows();
    
    double efficiency = 0.0;
    int count = 0;
    
    for (int i = 0; i < nNodes; ++i) {
        for (int j = i + 1; j < nNodes; ++j) {
            if (distances(i, j) > 0 && std::isfinite(distances(i, j))) {
                efficiency += 1.0 / distances(i, j);
                count++;
            }
        }
    }
    
    return count > 0 ? efficiency / count : 0.0;
}

//=============================================================================================================

VectorXd NetworkAnalysis::computeLocalEfficiency(const MatrixXd& adjacencyMatrix,
                                                  double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    binaryMatrix.diagonal().setZero();
    
    int nNodes = binaryMatrix.rows();
    VectorXd localEfficiency(nNodes);
    
    for (int i = 0; i < nNodes; ++i) {
        // Find neighbors of node i
        std::vector<int> neighbors;
        for (int j = 0; j < nNodes; ++j) {
            if (binaryMatrix(i, j) > 0) {
                neighbors.push_back(j);
            }
        }
        
        if (neighbors.size() < 2) {
            localEfficiency[i] = 0.0;
            continue;
        }
        
        // Create subgraph of neighbors
        int nNeighbors = neighbors.size();
        MatrixXd subgraph(nNeighbors, nNeighbors);
        
        for (int j = 0; j < nNeighbors; ++j) {
            for (int k = 0; k < nNeighbors; ++k) {
                subgraph(j, k) = binaryMatrix(neighbors[j], neighbors[k]);
            }
        }
        
        // Compute efficiency of subgraph
        MatrixXd subDistances = floydWarshall(subgraph);
        double efficiency = 0.0;
        int count = 0;
        
        for (int j = 0; j < nNeighbors; ++j) {
            for (int k = j + 1; k < nNeighbors; ++k) {
                if (subDistances(j, k) > 0 && std::isfinite(subDistances(j, k))) {
                    efficiency += 1.0 / subDistances(j, k);
                    count++;
                }
            }
        }
        
        localEfficiency[i] = count > 0 ? efficiency / count : 0.0;
    }
    
    return localEfficiency;
}
//=============================================================================================================

QMap<QString, double> NetworkAnalysis::computeSmallWorldProperties(const MatrixXd& adjacencyMatrix,
                                                                    double threshold)
{
    QMap<QString, double> properties;
    
    // Compute clustering coefficient
    VectorXd clustering = computeClusteringCoefficient(adjacencyMatrix, threshold);
    double avgClustering = clustering.mean();
    
    // Compute characteristic path length
    MatrixXd distances = computeShortestPaths(adjacencyMatrix, threshold);
    int nNodes = distances.rows();
    
    double totalDistance = 0.0;
    int count = 0;
    
    for (int i = 0; i < nNodes; ++i) {
        for (int j = i + 1; j < nNodes; ++j) {
            if (distances(i, j) > 0 && std::isfinite(distances(i, j))) {
                totalDistance += distances(i, j);
                count++;
            }
        }
    }
    
    double avgPathLength = count > 0 ? totalDistance / count : 0.0;
    
    properties["clustering"] = avgClustering;
    properties["path_length"] = avgPathLength;
    
    return properties;
}

//=============================================================================================================

VectorXi NetworkAnalysis::detectCommunities(const MatrixXd& adjacencyMatrix,
                                             double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    int nNodes = binaryMatrix.rows();
    
    // Simple greedy modularity optimization
    VectorXi communities = VectorXi::LinSpaced(nNodes, 0, nNodes - 1);
    double bestModularity = computeModularity(binaryMatrix, communities);
    
    bool improved = true;
    while (improved) {
        improved = false;
        
        for (int i = 0; i < nNodes; ++i) {
            int originalCommunity = communities[i];
            double bestLocalModularity = bestModularity;
            int bestCommunity = originalCommunity;
            
            // Try moving node i to each neighbor's community
            for (int j = 0; j < nNodes; ++j) {
                if (i != j && binaryMatrix(i, j) > 0) {
                    communities[i] = communities[j];
                    double newModularity = computeModularity(binaryMatrix, communities);
                    
                    if (newModularity > bestLocalModularity) {
                        bestLocalModularity = newModularity;
                        bestCommunity = communities[j];
                        improved = true;
                    }
                }
            }
            
            communities[i] = bestCommunity;
            bestModularity = bestLocalModularity;
        }
    }
    
    return communities;
}

//=============================================================================================================

double NetworkAnalysis::computeModularity(const MatrixXd& adjacencyMatrix,
                                           const VectorXi& communities)
{
    int nNodes = adjacencyMatrix.rows();
    double totalEdges = adjacencyMatrix.sum() / 2.0; // Undirected graph
    
    if (totalEdges == 0) {
        return 0.0;
    }
    
    double modularity = 0.0;
    
    for (int i = 0; i < nNodes; ++i) {
        for (int j = i + 1; j < nNodes; ++j) {
            if (communities[i] == communities[j]) {
                double aij = adjacencyMatrix(i, j);
                double ki = adjacencyMatrix.row(i).sum();
                double kj = adjacencyMatrix.row(j).sum();
                double expected = (ki * kj) / (2.0 * totalEdges);
                
                modularity += aij - expected;
            }
        }
    }
    
    return modularity / totalEdges;
}

//=============================================================================================================

double NetworkAnalysis::computeAssortativity(const MatrixXd& adjacencyMatrix,
                                              double threshold)
{
    MatrixXd binaryMatrix = binarizeMatrix(adjacencyMatrix, threshold);
    VectorXd degrees = computeDegree(binaryMatrix, 0.0);
    
    int nNodes = binaryMatrix.rows();
    double sumDegreeProducts = 0.0;
    double sumDegrees = 0.0;
    double sumSquaredDegrees = 0.0;
    int nEdges = 0;
    
    for (int i = 0; i < nNodes; ++i) {
        for (int j = i + 1; j < nNodes; ++j) {
            if (binaryMatrix(i, j) > 0) {
                sumDegreeProducts += degrees[i] * degrees[j];
                sumDegrees += degrees[i] + degrees[j];
                sumSquaredDegrees += degrees[i] * degrees[i] + degrees[j] * degrees[j];
                nEdges++;
            }
        }
    }
    
    if (nEdges == 0) {
        return 0.0;
    }
    
    double meanDegreeProduct = sumDegreeProducts / nEdges;
    double meanDegree = sumDegrees / (2.0 * nEdges);
    double meanSquaredDegree = sumSquaredDegrees / (2.0 * nEdges);
    
    double numerator = meanDegreeProduct - meanDegree * meanDegree;
    double denominator = meanSquaredDegree - meanDegree * meanDegree;
    
    return denominator > 0 ? numerator / denominator : 0.0;
}

//=============================================================================================================

MatrixXd NetworkAnalysis::binarizeMatrix(const MatrixXd& matrix, double threshold)
{
    if (threshold <= 0.0) {
        return matrix;
    }
    
    return (matrix.array() > threshold).cast<double>();
}

//=============================================================================================================

MatrixXd NetworkAnalysis::floydWarshall(const MatrixXd& adjacencyMatrix)
{
    int nNodes = adjacencyMatrix.rows();
    MatrixXd distances = MatrixXd::Constant(nNodes, nNodes, std::numeric_limits<double>::infinity());
    
    // Initialize distances
    for (int i = 0; i < nNodes; ++i) {
        distances(i, i) = 0.0;
        for (int j = 0; j < nNodes; ++j) {
            if (adjacencyMatrix(i, j) > 0) {
                distances(i, j) = 1.0; // Unweighted graph
            }
        }
    }
    
    // Floyd-Warshall algorithm
    for (int k = 0; k < nNodes; ++k) {
        for (int i = 0; i < nNodes; ++i) {
            for (int j = 0; j < nNodes; ++j) {
                if (distances(i, k) + distances(k, j) < distances(i, j)) {
                    distances(i, j) = distances(i, k) + distances(k, j);
                }
            }
        }
    }
    
    return distances;
}