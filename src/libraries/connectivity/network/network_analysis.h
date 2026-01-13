//=============================================================================================================
/**
 * @file     network_analysis.h
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
 * @brief    NetworkAnalysis class declaration for graph theory metrics
 *
 */

#ifndef NETWORK_ANALYSIS_H
#define NETWORK_ANALYSIS_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "../connectivity_global.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QSharedPointer>
#include <QMap>
#include <QString>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>
#include <Eigen/Sparse>

//=============================================================================================================
// DEFINE NAMESPACE CONNECTIVITYLIB
//=============================================================================================================

namespace CONNECTIVITYLIB {

//=============================================================================================================
/**
 * This class provides graph theory metrics and network analysis tools
 * for connectivity matrices.
 *
 * @brief This class provides network analysis and graph theory metrics.
 */
class CONNECTIVITYSHARED_EXPORT NetworkAnalysis
{

public:
    typedef QSharedPointer<NetworkAnalysis> SPtr;            /**< Shared pointer type for NetworkAnalysis. */
    typedef QSharedPointer<const NetworkAnalysis> ConstSPtr; /**< Const shared pointer type for NetworkAnalysis. */

    //=========================================================================================================
    /**
     * Constructs a NetworkAnalysis object.
     */
    explicit NetworkAnalysis();

    //=========================================================================================================
    /**
     * Computes node degree (number of connections per node).
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix (0 = no threshold).
     *
     * @return Vector of node degrees.
     */
    static Eigen::VectorXd computeDegree(const Eigen::MatrixXd& adjacencyMatrix,
                                          double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes node strength (sum of edge weights per node).
     *
     * @param[in] adjacencyMatrix    The weighted adjacency matrix.
     *
     * @return Vector of node strengths.
     */
    static Eigen::VectorXd computeStrength(const Eigen::MatrixXd& adjacencyMatrix);

    //=========================================================================================================
    /**
     * Computes clustering coefficient for each node.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Vector of clustering coefficients.
     */
    static Eigen::VectorXd computeClusteringCoefficient(const Eigen::MatrixXd& adjacencyMatrix,
                                                         double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes betweenness centrality for each node.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Vector of betweenness centrality values.
     */
    static Eigen::VectorXd computeBetweennessCentrality(const Eigen::MatrixXd& adjacencyMatrix,
                                                         double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes eigenvector centrality for each node.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] maxIterations      Maximum number of iterations.
     * @param[in] tolerance          Convergence tolerance.
     *
     * @return Vector of eigenvector centrality values.
     */
    static Eigen::VectorXd computeEigenvectorCentrality(const Eigen::MatrixXd& adjacencyMatrix,
                                                         int maxIterations = 1000,
                                                         double tolerance = 1e-6);

    //=========================================================================================================
    /**
     * Computes shortest path lengths between all pairs of nodes.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Matrix of shortest path lengths.
     */
    static Eigen::MatrixXd computeShortestPaths(const Eigen::MatrixXd& adjacencyMatrix,
                                                 double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes global efficiency of the network.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Global efficiency value.
     */
    static double computeGlobalEfficiency(const Eigen::MatrixXd& adjacencyMatrix,
                                           double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes local efficiency for each node.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Vector of local efficiency values.
     */
    static Eigen::VectorXd computeLocalEfficiency(const Eigen::MatrixXd& adjacencyMatrix,
                                                   double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes small-world properties (clustering coefficient and path length).
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Map containing "clustering" and "path_length" values.
     */
    static QMap<QString, double> computeSmallWorldProperties(const Eigen::MatrixXd& adjacencyMatrix,
                                                             double threshold = 0.0);

    //=========================================================================================================
    /**
     * Detects communities using modularity optimization.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Vector of community assignments for each node.
     */
    static Eigen::VectorXi detectCommunities(const Eigen::MatrixXd& adjacencyMatrix,
                                              double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes modularity of a given community structure.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] communities        Community assignments for each node.
     *
     * @return Modularity value.
     */
    static double computeModularity(const Eigen::MatrixXd& adjacencyMatrix,
                                    const Eigen::VectorXi& communities);

    //=========================================================================================================
    /**
     * Computes rich club coefficient.
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Map of degree levels to rich club coefficients.
     */
    static QMap<int, double> computeRichClubCoefficient(const Eigen::MatrixXd& adjacencyMatrix,
                                                         double threshold = 0.0);

    //=========================================================================================================
    /**
     * Computes assortativity coefficient (degree correlation).
     *
     * @param[in] adjacencyMatrix    The adjacency matrix.
     * @param[in] threshold          Threshold for binarizing the matrix.
     *
     * @return Assortativity coefficient.
     */
    static double computeAssortativity(const Eigen::MatrixXd& adjacencyMatrix,
                                       double threshold = 0.0);

private:
    //=========================================================================================================
    /**
     * Binarizes adjacency matrix using threshold.
     *
     * @param[in] matrix             Input matrix.
     * @param[in] threshold          Threshold value.
     *
     * @return Binarized matrix.
     */
    static Eigen::MatrixXd binarizeMatrix(const Eigen::MatrixXd& matrix, double threshold);

    //=========================================================================================================
    /**
     * Computes Floyd-Warshall shortest paths.
     *
     * @param[in] adjacencyMatrix    Binary adjacency matrix.
     *
     * @return Distance matrix.
     */
    static Eigen::MatrixXd floydWarshall(const Eigen::MatrixXd& adjacencyMatrix);

    //=========================================================================================================
    /**
     * Computes number of triangles for each node.
     *
     * @param[in] adjacencyMatrix    Binary adjacency matrix.
     *
     * @return Vector of triangle counts.
     */
    static Eigen::VectorXd computeTriangles(const Eigen::MatrixXd& adjacencyMatrix);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // namespace CONNECTIVITYLIB

#endif // NETWORK_ANALYSIS_H