//=============================================================================================================
/**
 * @file     fwd_leadfield_enhanced.cpp
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
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    Enhanced leadfield matrix computation implementation.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "fwd_leadfield_enhanced.h"

#include <utils/mnemath.h>
#include <fiff/fiff_constants.h>

#include <QDebug>
#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <cmath>

#include <Eigen/Geometry>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace FWDLIB;
using namespace MNELIB;
using namespace FIFFLIB;
using namespace UTILSLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

FwdLeadfieldEnhanced::FwdLeadfieldEnhanced()
{
}

//=============================================================================================================

FwdLeadfieldEnhanced::~FwdLeadfieldEnhanced()
{
}

//=============================================================================================================

MatrixXd FwdLeadfieldEnhanced::compute_leadfield_enhanced(const MNESourceSpace& src,
                                                        const FiffInfo& info,
                                                        const FiffCoordTrans& trans,
                                                        const FwdBemModel* bemModel,
                                                        const FwdEegSphereModel* eegModel,
                                                        const FwdCoilSet* megCoils,
                                                        const FwdCoilSet* eegElectrodes,
                                                        DipoleModelType dipoleType,
                                                        HeadModelType headModelType,
                                                        OptimizationMethod optimization,
                                                        int nJobs)
{
    Q_UNUSED(trans)
    Q_UNUSED(eegModel)
    Q_UNUSED(megCoils)
    Q_UNUSED(eegElectrodes)
    Q_UNUSED(dipoleType)
    
    // Collect source positions and orientations
    MatrixX3f sourcePositions;
    MatrixX3f sourceOrientations;
    int totalSources = 0;
    
    // Count total sources
    for (int hemi = 0; hemi < src.size(); ++hemi) {
        totalSources += src[hemi].nuse;
    }
    
    sourcePositions.resize(totalSources, 3);
    sourceOrientations.resize(totalSources, 3);
    
    int sourceIdx = 0;
    for (int hemi = 0; hemi < src.size(); ++hemi) {
        const MNEHemisphere& hemisphere = src[hemi];
        
        for (int i = 0; i < hemisphere.nuse; ++i) {
            sourcePositions.row(sourceIdx) = hemisphere.rr.row(hemisphere.vertno[i]);
            sourceOrientations.row(sourceIdx) = hemisphere.nn.row(hemisphere.vertno[i]);
            sourceIdx++;
        }
    }
    
    // Collect sensor positions and orientations
    MatrixX3f sensorPositions(info.nchan, 3);
    MatrixX3f sensorOrientations(info.nchan, 3);
    
    for (int i = 0; i < info.nchan; ++i) {
        const FiffChInfo& chInfo = info.chs[i];
        // Extract position and orientation from chpos matrix
        sensorPositions(i, 0) = chInfo.chpos.r0[0];
        sensorPositions(i, 1) = chInfo.chpos.r0[1];
        sensorPositions(i, 2) = chInfo.chpos.r0[2];
        sensorOrientations(i, 0) = chInfo.chpos.ex[0];
        sensorOrientations(i, 1) = chInfo.chpos.ex[1];
        sensorOrientations(i, 2) = chInfo.chpos.ex[2];
    }
    
    // Determine optimal computation strategy
    if (optimization == ADAPTIVE) {
        optimization = determine_optimal_strategy(sourcePositions, sensorPositions, bemModel);
    }
    
    // Compute leadfield matrix based on optimization method
    MatrixXd leadfield;
    
    switch (optimization) {
        case PARALLEL:
        case BLOCK_WISE:
            leadfield = compute_leadfield_parallel(sourcePositions, sourceOrientations,
                                                 sensorPositions, sensorOrientations,
                                                 bemModel, 1000, nJobs);
            break;
            
        case STANDARD:
        default:
            leadfield = compute_leadfield_parallel(sourcePositions, sourceOrientations,
                                                 sensorPositions, sensorOrientations,
                                                 bemModel, totalSources, 1);
            break;
    }
    
    // Validate leadfield quality
    QMap<QString, double> quality = validate_leadfield_quality(leadfield, sourcePositions, sensorPositions);
    
    qDebug() << "[FwdLeadfieldEnhanced::compute_leadfield_enhanced] Leadfield quality metrics:";
    for (auto it = quality.begin(); it != quality.end(); ++it) {
        qDebug() << "  " << it.key() << ":" << it.value();
    }
    
    return leadfield;
}

//=============================================================================================================

VectorXd FwdLeadfieldEnhanced::compute_single_dipole_leadfield(const Vector3f& dipolePos,
                                                             const Vector3f& dipoleOri,
                                                             const MatrixX3f& sensorPos,
                                                             const MatrixX3f& sensorOri,
                                                             const FwdBemModel* bemModel,
                                                             HeadModelType headModelType)
{
    VectorXd leadfield(sensorPos.rows());
    
    for (int i = 0; i < sensorPos.rows(); ++i) {
        Vector3f sPos = sensorPos.row(i);
        Vector3f sOri = sensorOri.row(i);
        
        double value = 0.0;
        
        switch (headModelType) {
            case BEM_MODEL: {
                if (bemModel) {
                    value = compute_bem_contribution(dipolePos, dipoleOri, sPos, sOri, bemModel);
                }
                break;
            }
                
            case SPHERE_MODEL: {
                // Analytical sphere model computation
                Vector3f r = sPos - dipolePos;
                double r_norm = r.norm();
                if (r_norm > 1e-6) {
                    // Simplified sphere model (Sarvas formula)
                    Vector3f r_hat = r / r_norm;
                    double dot_product = dipoleOri.dot(r_hat);
                    value = dot_product / (4.0 * M_PI * r_norm * r_norm);
                    value *= sOri.dot(r_hat);
                }
                break;
            }
                
            default: {
                // Default to simple dipole model
                Vector3f r = sPos - dipolePos;
                double r_norm = r.norm();
                if (r_norm > 1e-6) {
                    Vector3f r_hat = r / r_norm;
                    value = dipoleOri.dot(sOri) / (4.0 * M_PI * r_norm * r_norm);
                }
                break;
            }
        }
        
        leadfield(i) = value;
    }
    
    return leadfield;
}

//=============================================================================================================

MatrixXd FwdLeadfieldEnhanced::compute_leadfield_parallel(const MatrixX3f& sourcePositions,
                                                        const MatrixX3f& sourceOrientations,
                                                        const MatrixX3f& sensorPositions,
                                                        const MatrixX3f& sensorOrientations,
                                                        const FwdBemModel* bemModel,
                                                        int blockSize,
                                                        int nJobs)
{
    int nSources = sourcePositions.rows();
    int nSensors = sensorPositions.rows();
    
    MatrixXd leadfield = MatrixXd::Zero(nSensors, nSources);
    
    // Determine number of blocks
    int nBlocks = (nSources + blockSize - 1) / blockSize;
    
    if (nJobs > 1) {
        // Parallel computation
        QList<QFuture<MatrixXd>> futures;
        
        for (int block = 0; block < nBlocks; ++block) {
            int blockStart = block * blockSize;
            int blockEnd = qMin(blockStart + blockSize, nSources);
            
            QFuture<MatrixXd> future = QtConcurrent::run([=]() {
                return compute_leadfield_block(blockStart, blockEnd,
                                             sourcePositions, sourceOrientations,
                                             sensorPositions, sensorOrientations,
                                             bemModel);
            });
            
            futures.append(future);
        }
        
        // Collect results
        for (int block = 0; block < nBlocks; ++block) {
            int blockStart = block * blockSize;
            int blockEnd = qMin(blockStart + blockSize, nSources);
            
            MatrixXd blockResult = futures[block].result();
            leadfield.block(0, blockStart, nSensors, blockEnd - blockStart) = blockResult;
        }
    } else {
        // Sequential computation
        for (int block = 0; block < nBlocks; ++block) {
            int blockStart = block * blockSize;
            int blockEnd = qMin(blockStart + blockSize, nSources);
            
            MatrixXd blockResult = compute_leadfield_block(blockStart, blockEnd,
                                                         sourcePositions, sourceOrientations,
                                                         sensorPositions, sensorOrientations,
                                                         bemModel);
            
            leadfield.block(0, blockStart, nSensors, blockEnd - blockStart) = blockResult;
        }
    }
    
    return leadfield;
}

//=============================================================================================================

FwdLeadfieldEnhanced::OptimizationMethod FwdLeadfieldEnhanced::determine_optimal_strategy(const MatrixX3f& sourcePositions,
                                                                                         const MatrixX3f& sensorPositions,
                                                                                         const FwdBemModel* bemModel)
{
    Q_UNUSED(bemModel)
    
    int nSources = sourcePositions.rows();
    int nSensors = sensorPositions.rows();
    
    // Simple heuristics for optimization strategy selection
    if (nSources > 10000 && nSensors > 100) {
        return PARALLEL;
    } else if (nSources > 5000) {
        return BLOCK_WISE;
    } else {
        return STANDARD;
    }
}

//=============================================================================================================

VectorXd FwdLeadfieldEnhanced::compute_multipole_expansion(const Vector3f& dipolePos,
                                                         const Vector3f& sensorPos,
                                                         int order,
                                                         const FwdBemModel* bemModel)
{
    Q_UNUSED(bemModel)
    
    // Simplified multipole expansion
    VectorXd coefficients(order + 1);
    
    Vector3f r = sensorPos - dipolePos;
    double r_norm = r.norm();
    
    if (r_norm > 1e-6) {
        for (int l = 0; l <= order; ++l) {
            coefficients(l) = std::pow(r_norm, -l - 1) / (4.0 * M_PI);
        }
    } else {
        coefficients.setZero();
    }
    
    return coefficients;
}

//=============================================================================================================

FwdBemModel FwdLeadfieldEnhanced::apply_adaptive_refinement(const FwdBemModel& bemModel,
                                                          const MatrixX3f& sourcePositions,
                                                          double tolerance)
{
    Q_UNUSED(sourcePositions)
    Q_UNUSED(tolerance)
    
    // For now, return the original model
    // TODO: Implement adaptive mesh refinement based on source positions and tolerance
    return bemModel;
}

//=============================================================================================================

QMap<QString, double> FwdLeadfieldEnhanced::validate_leadfield_quality(const MatrixXd& leadfield,
                                                                      const MatrixX3f& sourcePositions,
                                                                      const MatrixX3f& sensorPositions)
{
    Q_UNUSED(sourcePositions)
    Q_UNUSED(sensorPositions)
    
    QMap<QString, double> quality;
    
    if (leadfield.size() == 0) {
        quality["condition_number"] = std::numeric_limits<double>::infinity();
        quality["rank"] = 0;
        quality["frobenius_norm"] = 0;
        return quality;
    }
    
    // Compute condition number
    JacobiSVD<MatrixXd> svd(leadfield, ComputeThinU | ComputeThinV);
    VectorXd singularValues = svd.singularValues();
    
    if (singularValues.size() > 0 && singularValues(singularValues.size() - 1) > 1e-12) {
        quality["condition_number"] = singularValues(0) / singularValues(singularValues.size() - 1);
    } else {
        quality["condition_number"] = std::numeric_limits<double>::infinity();
    }
    
    // Compute rank
    int rank = 0;
    for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > 1e-12) {
            rank++;
        }
    }
    quality["rank"] = rank;
    
    // Compute Frobenius norm
    quality["frobenius_norm"] = leadfield.norm();
    
    // Compute mean and std of leadfield values
    double mean = leadfield.mean();
    double variance = (leadfield.array() - mean).square().mean();
    quality["mean"] = mean;
    quality["std"] = std::sqrt(variance);
    
    return quality;
}

//=============================================================================================================

double FwdLeadfieldEnhanced::compute_bem_contribution(const Vector3f& dipolePos,
                                                     const Vector3f& dipoleOri,
                                                     const Vector3f& sensorPos,
                                                     const Vector3f& sensorOri,
                                                     const FwdBemModel* bemModel)
{
    Q_UNUSED(bemModel)
    
    // Simplified BEM computation
    // In a full implementation, this would involve solving the BEM equations
    
    Vector3f r = sensorPos - dipolePos;
    double r_norm = r.norm();
    
    if (r_norm < 1e-6) {
        return 0.0;
    }
    
    Vector3f r_hat = r / r_norm;
    
    // Simplified BEM formula (this is a placeholder)
    double infinite_medium = dipoleOri.dot(sensorOri) / (4.0 * M_PI * r_norm * r_norm);
    
    // BEM correction factor (simplified)
    double bem_correction = 1.0 + 0.1 * std::exp(-r_norm / 0.1); // Placeholder
    
    return infinite_medium * bem_correction;
}

//=============================================================================================================

double FwdLeadfieldEnhanced::compute_sphere_leadfield(const Vector3f& dipolePos,
                                                     const Vector3f& dipoleOri,
                                                     const Vector3f& sensorPos,
                                                     const Vector3f& sensorOri,
                                                     const FwdEegSphereModel* sphereModel)
{
    Q_UNUSED(sphereModel)
    
    // Analytical sphere model (Sarvas formula for MEG)
    Vector3f r = sensorPos - dipolePos;
    double r_norm = r.norm();
    
    if (r_norm < 1e-6) {
        return 0.0;
    }
    
    Vector3f r_hat = r / r_norm;
    
    // Simplified Sarvas formula
    Vector3f cross_product = dipoleOri.cross(r_hat);
    double leadfield_value = cross_product.dot(sensorOri) / (4.0 * M_PI * r_norm * r_norm);
    
    return leadfield_value;
}

//=============================================================================================================

MatrixXd FwdLeadfieldEnhanced::compute_leadfield_block(int blockStart,
                                                      int blockEnd,
                                                      const MatrixX3f& sourcePositions,
                                                      const MatrixX3f& sourceOrientations,
                                                      const MatrixX3f& sensorPositions,
                                                      const MatrixX3f& sensorOrientations,
                                                      const FwdBemModel* bemModel)
{
    int nSensors = sensorPositions.rows();
    int blockSize = blockEnd - blockStart;
    
    MatrixXd blockLeadfield(nSensors, blockSize);
    
    for (int sourceIdx = 0; sourceIdx < blockSize; ++sourceIdx) {
        int globalSourceIdx = blockStart + sourceIdx;
        
        Vector3f dipolePos = sourcePositions.row(globalSourceIdx);
        Vector3f dipoleOri = sourceOrientations.row(globalSourceIdx);
        
        VectorXd leadfieldColumn = compute_single_dipole_leadfield(dipolePos, dipoleOri,
                                                                 sensorPositions, sensorOrientations,
                                                                 bemModel, BEM_MODEL);
        
        blockLeadfield.col(sourceIdx) = leadfieldColumn;
    }
    
    return blockLeadfield;
}