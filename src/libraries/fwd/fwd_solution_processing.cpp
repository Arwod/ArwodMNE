//=============================================================================================================
/**
 * @file     fwd_solution_processing.cpp
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
 * @brief    Forward solution processing functions implementation.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "fwd_solution_processing.h"

#include <fiff/fiff_info.h>
#include <fiff/fiff_constants.h>
#include <utils/mnemath.h>

#include <QDebug>
#include <QtConcurrent>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace FWDLIB;
using namespace MNELIB;
using namespace FIFFLIB;
using namespace FSLIB;
using namespace UTILSLIB;
using namespace Eigen;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

FiffEvoked FwdSolutionProcessing::apply_forward(const MNEForwardSolution& fwd,
                                               const MNESourceEstimate& stc,
                                               const FiffInfo& info,
                                               bool use_cps)
{
    // Validate input compatibility
    if (!validate_fwd_stc_compatibility(fwd, stc)) {
        qWarning() << "[FwdSolutionProcessing::apply_forward] Forward solution and source estimate are not compatible";
        return FiffEvoked();
    }

    // Use provided info or create simple info from forward solution
    FiffInfo useInfo;
    if (info.isEmpty()) {
        // Create minimal info from forward solution
        useInfo.nchan = fwd.info.nchan;
        useInfo.chs = fwd.info.chs;
        useInfo.ch_names = fwd.info.ch_names;
        useInfo.bads = fwd.info.bads;
        useInfo.ctf_head_t = fwd.info.ctf_head_t;
        useInfo.dev_head_t = fwd.info.dev_head_t;
        useInfo.meas_id = fwd.info.meas_id;
        useInfo.filename = fwd.info.filename;
    } else {
        useInfo = info;
    }
    
    // Get source data
    MatrixXd sourceData = stc.data;
    
    // Apply cortical patch statistics if requested
    if (use_cps) {
        sourceData = apply_cortical_patch_statistics(fwd, sourceData, use_cps);
    }
    
    // Apply forward solution: sensor_data = G * source_data
    MatrixXd sensorData = fwd.sol->data * sourceData;
    
    // Create evoked object
    FiffEvoked evoked;
    evoked.info = useInfo;
    evoked.data = sensorData;
    evoked.times = stc.times;
    evoked.first = 0;
    evoked.last = stc.times.size() - 1;
    evoked.nave = 1;
    evoked.aspect_kind = FIFFV_ASPECT_AVERAGE;
    evoked.comment = QString("Simulated from forward solution");
    
    return evoked;
}

//=============================================================================================================

FiffRawData FwdSolutionProcessing::apply_forward_raw(const MNEForwardSolution& fwd,
                                                   const MNESourceEstimate& stc,
                                                   const FiffInfo& info,
                                                   qint32 start,
                                                   qint32 stop)
{
    // Validate input compatibility
    if (!validate_fwd_stc_compatibility(fwd, stc)) {
        qWarning() << "[FwdSolutionProcessing::apply_forward_raw] Forward solution and source estimate are not compatible";
        return FiffRawData();
    }

    // Use provided info or create simple info from forward solution
    FiffInfo useInfo;
    if (info.isEmpty()) {
        // Create minimal info from forward solution
        useInfo.nchan = fwd.info.nchan;
        useInfo.chs = fwd.info.chs;
        useInfo.ch_names = fwd.info.ch_names;
        useInfo.bads = fwd.info.bads;
        useInfo.ctf_head_t = fwd.info.ctf_head_t;
        useInfo.dev_head_t = fwd.info.dev_head_t;
        useInfo.meas_id = fwd.info.meas_id;
        useInfo.filename = fwd.info.filename;
    } else {
        useInfo = info;
    }
    
    // Handle start/stop indices
    qint32 nSamples = stc.data.cols();
    if (stop < 0) stop = nSamples - 1;
    start = qMax(0, start);
    stop = qMin(nSamples - 1, stop);
    
    if (start >= stop) {
        qWarning() << "[FwdSolutionProcessing::apply_forward_raw] Invalid start/stop indices";
        return FiffRawData();
    }
    
    // Extract time range
    MatrixXd sourceData = stc.data.block(0, start, stc.data.rows(), stop - start + 1);
    
    // Apply forward solution: sensor_data = G * source_data
    MatrixXd sensorData = fwd.sol->data * sourceData;
    
    // Create raw data object
    FiffRawData rawData;
    rawData.info = useInfo;
    rawData.first_samp = start;
    rawData.last_samp = stop;
    
    return rawData;
}

//=============================================================================================================

MNEForwardSolution FwdSolutionProcessing::convert_forward_solution(const MNEForwardSolution& fwd,
                                                                  bool surf_ori,
                                                                  bool force_fixed,
                                                                  bool copy,
                                                                  bool use_cps,
                                                                  bool verbose)
{
    Q_UNUSED(use_cps)
    Q_UNUSED(verbose)
    
    // Create copy if requested
    MNEForwardSolution convertedFwd = copy ? fwd : fwd;
    
    // Convert source orientations if needed
    if (surf_ori || force_fixed) {
        convertedFwd = convert_source_orientations(convertedFwd, surf_ori, force_fixed);
    }
    
    return convertedFwd;
}

//=============================================================================================================

MNEForwardSolution FwdSolutionProcessing::restrict_forward_to_label(const MNEForwardSolution& fwd,
                                                                   const Label& label,
                                                                   bool copy)
{
    QList<Label> labels;
    labels.append(label);
    return restrict_forward_to_labels(fwd, labels, copy);
}

//=============================================================================================================

MNEForwardSolution FwdSolutionProcessing::restrict_forward_to_labels(const MNEForwardSolution& fwd,
                                                                    const QList<Label>& labels,
                                                                    bool copy)
{
    // Create copy if requested
    MNEForwardSolution restrictedFwd = copy ? fwd : fwd;
    
    // Collect all vertices from labels
    QSet<qint32> selectedVertices;
    for (const Label& label : labels) {
        for (qint32 vertex : label.vertices) {
            selectedVertices.insert(vertex);
        }
    }
    
    // Find corresponding source space indices
    QList<qint32> sourceIndices;
    qint32 currentIdx = 0;
    
    for (qint32 hemi = 0; hemi < restrictedFwd.src.size(); ++hemi) {
        const MNEHemisphere& hemisphere = restrictedFwd.src[hemi];
        
        for (qint32 i = 0; i < hemisphere.vertno.size(); ++i) {
            if (selectedVertices.contains(hemisphere.vertno[i])) {
                // For free orientation, add all three components
                if (restrictedFwd.source_ori == FIFFV_MNE_FREE_ORI) {
                    sourceIndices.append(currentIdx * 3);
                    sourceIndices.append(currentIdx * 3 + 1);
                    sourceIndices.append(currentIdx * 3 + 2);
                } else {
                    sourceIndices.append(currentIdx);
                }
            }
            currentIdx++;
        }
    }
    
    // Restrict gain matrix
    if (!sourceIndices.isEmpty()) {
        MatrixXd restrictedGain(restrictedFwd.sol->data.rows(), sourceIndices.size());
        for (qint32 i = 0; i < sourceIndices.size(); ++i) {
            restrictedGain.col(i) = restrictedFwd.sol->data.col(sourceIndices[i]);
        }
        restrictedFwd.sol->data = restrictedGain;
        restrictedFwd.nsource = sourceIndices.size() / (restrictedFwd.source_ori == FIFFV_MNE_FREE_ORI ? 3 : 1);
    }
    
    return restrictedFwd;
}

//=============================================================================================================

MNEForwardSolution FwdSolutionProcessing::restrict_forward_to_stc(const MNEForwardSolution& fwd,
                                                                 const MNESourceEstimate& stc,
                                                                 bool copy)
{
    // Create copy if requested
    MNEForwardSolution restrictedFwd = copy ? fwd : fwd;
    
    // Get active vertices from source estimate
    QSet<qint32> activeVertices;
    
    // Add all vertices from the source estimate
    for (qint32 i = 0; i < stc.vertices.size(); ++i) {
        activeVertices.insert(stc.vertices[i]);
    }
    
    // Find corresponding source space indices
    QList<qint32> sourceIndices;
    qint32 currentIdx = 0;
    
    for (qint32 hemi = 0; hemi < restrictedFwd.src.size(); ++hemi) {
        const MNEHemisphere& hemisphere = restrictedFwd.src[hemi];
        
        for (qint32 i = 0; i < hemisphere.vertno.size(); ++i) {
            if (activeVertices.contains(hemisphere.vertno[i])) {
                // For free orientation, add all three components
                if (restrictedFwd.source_ori == FIFFV_MNE_FREE_ORI) {
                    sourceIndices.append(currentIdx * 3);
                    sourceIndices.append(currentIdx * 3 + 1);
                    sourceIndices.append(currentIdx * 3 + 2);
                } else {
                    sourceIndices.append(currentIdx);
                }
            }
            currentIdx++;
        }
    }
    
    // Restrict gain matrix
    if (!sourceIndices.isEmpty()) {
        MatrixXd restrictedGain(restrictedFwd.sol->data.rows(), sourceIndices.size());
        for (qint32 i = 0; i < sourceIndices.size(); ++i) {
            restrictedGain.col(i) = restrictedFwd.sol->data.col(sourceIndices[i]);
        }
        restrictedFwd.sol->data = restrictedGain;
        restrictedFwd.nsource = sourceIndices.size() / (restrictedFwd.source_ori == FIFFV_MNE_FREE_ORI ? 3 : 1);
    }
    
    return restrictedFwd;
}

//=============================================================================================================

bool FwdSolutionProcessing::validate_fwd_stc_compatibility(const MNEForwardSolution& fwd,
                                                          const MNESourceEstimate& stc)
{
    // Check if forward solution is valid
    if (fwd.isEmpty()) {
        qWarning() << "[FwdSolutionProcessing::validate_fwd_stc_compatibility] Forward solution is empty";
        return false;
    }
    
    // Check if source estimate is valid
    if (stc.isEmpty()) {
        qWarning() << "[FwdSolutionProcessing::validate_fwd_stc_compatibility] Source estimate is empty";
        return false;
    }
    
    // Check source space compatibility
    qint32 expectedSources = 0;
    for (qint32 hemi = 0; hemi < fwd.src.size(); ++hemi) {
        expectedSources += fwd.src[hemi].nuse;
    }
    
    // Adjust for orientation
    if (fwd.source_ori == FIFFV_MNE_FREE_ORI) {
        expectedSources *= 3;
    }
    
    if (stc.data.rows() != expectedSources) {
        qWarning() << "[FwdSolutionProcessing::validate_fwd_stc_compatibility] Source count mismatch:"
                   << "Expected" << expectedSources << "got" << stc.data.rows();
        return false;
    }
    
    return true;
}

//=============================================================================================================

MatrixXd FwdSolutionProcessing::apply_cortical_patch_statistics(const MNEForwardSolution& fwd,
                                                               const MatrixXd& data,
                                                               bool use_cps)
{
    Q_UNUSED(fwd)
    
    if (!use_cps) {
        return data;
    }
    
    // For now, return data as-is
    // TODO: Implement cortical patch statistics processing
    // This would involve averaging over cortical patches based on the source space geometry
    
    return data;
}

//=============================================================================================================

MNEForwardSolution FwdSolutionProcessing::convert_source_orientations(const MNEForwardSolution& fwd,
                                                                     bool surf_ori,
                                                                     bool force_fixed)
{
    MNEForwardSolution convertedFwd = fwd;
    
    if (force_fixed && fwd.source_ori == FIFFV_MNE_FREE_ORI) {
        // Convert from free to fixed orientation
        convertedFwd.to_fixed_ori();
    }
    
    if (surf_ori && !fwd.surf_ori) {
        // Convert to surface-based orientations
        convertedFwd.surf_ori = true;
        // TODO: Implement surface orientation conversion
        // This would involve rotating the source orientations to align with surface normals
    }
    
    return convertedFwd;
}