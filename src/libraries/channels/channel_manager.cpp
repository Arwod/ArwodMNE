//=============================================================================================================
/**
 * @file     channel_manager.cpp
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
 * @brief    Implementation of the ChannelManager class.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "channel_manager.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_constants.h>
#include <fiff/fiff_ch_info.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QRegularExpression>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace CHANNELSLIB;
using namespace FIFFLIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

ChannelManager::ChannelManager(QObject *parent)
: QObject(parent)
{
}

//=============================================================================================================

ChannelManager::~ChannelManager()
{
}
//=============================================================================================================

bool ChannelManager::combineChannels(FiffInfo& info,
                                      const QMap<QString, QStringList>& groups)
{
    // Create new channel list
    QList<FiffChInfo> newChannels;
    QStringList newChannelNames;
    
    // Keep track of channels that are being combined
    QSet<QString> combinedChannels;
    
    // Add combined channels
    for (auto it = groups.begin(); it != groups.end(); ++it) {
        const QString& newName = it.key();
        const QStringList& channelGroup = it.value();
        
        if (channelGroup.isEmpty()) {
            continue;
        }
        
        // Find the first valid channel in the group to use as template
        FiffChInfo templateChannel;
        bool foundTemplate = false;
        
        for (const QString& chName : channelGroup) {
            int idx = info.ch_names.indexOf(chName);
            if (idx >= 0) {
                templateChannel = info.chs[idx];
                foundTemplate = true;
                break;
            }
        }
        
        if (!foundTemplate) {
            qWarning() << "No valid channels found in group for" << newName;
            continue;
        }
        
        // Create new combined channel
        FiffChInfo combinedChannel = templateChannel;
        combinedChannel.ch_name = newName;
        
        // Average positions if available
        Eigen::Vector3d avgPos = Eigen::Vector3d::Zero();
        int validPositions = 0;
        
        for (const QString& chName : channelGroup) {
            int idx = info.ch_names.indexOf(chName);
            if (idx >= 0) {
                Eigen::Vector3d pos = getChannelPosition(info.chs[idx]);
                if (pos.norm() > 0) {
                    avgPos += pos;
                    validPositions++;
                }
                combinedChannels.insert(chName);
            }
        }
        
        if (validPositions > 0) {
            avgPos /= validPositions;
            combinedChannel.eeg_loc(0, 0) = avgPos.x();
            combinedChannel.eeg_loc(1, 0) = avgPos.y();
            combinedChannel.eeg_loc(2, 0) = avgPos.z();
        }
        
        newChannels.append(combinedChannel);
        newChannelNames.append(newName);
    }
    
    // Add remaining channels that weren't combined
    for (int i = 0; i < info.chs.size(); ++i) {
        const QString& chName = info.ch_names[i];
        if (!combinedChannels.contains(chName)) {
            newChannels.append(info.chs[i]);
            newChannelNames.append(chName);
        }
    }
    
    // Update info
    info.chs = newChannels;
    info.ch_names = newChannelNames;
    info.nchan = newChannels.size();
    
    return true;
}

//=============================================================================================================

bool ChannelManager::equalizeChannels(QList<FiffInfo>& infos)
{
    if (infos.isEmpty()) {
        return false;
    }
    
    // Find common channels across all infos
    QSet<QString> commonChannels;
    
    // Start with channels from first info
    for (const QString& chName : infos[0].ch_names) {
        commonChannels.insert(chName);
    }
    
    // Intersect with channels from other infos
    for (int i = 1; i < infos.size(); ++i) {
        QSet<QString> currentChannels;
        for (const QString& chName : infos[i].ch_names) {
            currentChannels.insert(chName);
        }
        commonChannels.intersect(currentChannels);
    }
    
    if (commonChannels.isEmpty()) {
        qWarning() << "No common channels found across all infos";
        return false;
    }
    
    // Update each info to contain only common channels
    for (FiffInfo& info : infos) {
        QList<FiffChInfo> newChannels;
        QStringList newChannelNames;
        
        for (const QString& chName : commonChannels) {
            int idx = info.ch_names.indexOf(chName);
            if (idx >= 0) {
                newChannels.append(info.chs[idx]);
                newChannelNames.append(chName);
            }
        }
        
        info.chs = newChannels;
        info.ch_names = newChannelNames;
        info.nchan = newChannels.size();
    }
    
    return true;
}
//=============================================================================================================

bool ChannelManager::renameChannels(FiffInfo& info,
                                     const QMap<QString, QString>& mapping)
{
    bool changed = false;
    
    for (int i = 0; i < info.ch_names.size(); ++i) {
        const QString& oldName = info.ch_names[i];
        if (mapping.contains(oldName)) {
            const QString& newName = mapping[oldName];
            info.ch_names[i] = newName;
            info.chs[i].ch_name = newName;
            changed = true;
        }
    }
    
    return changed;
}

//=============================================================================================================

Eigen::SparseMatrix<double> ChannelManager::computeChannelAdjacency(const FiffInfo& info,
                                                                     double maxDistance)
{
    int nChannels = info.nchan;
    Eigen::SparseMatrix<double> adjacency(nChannels, nChannels);
    
    QList<Eigen::Triplet<double>> triplets;
    
    for (int i = 0; i < nChannels; ++i) {
        Eigen::Vector3d pos1 = getChannelPosition(info.chs[i]);
        
        for (int j = i + 1; j < nChannels; ++j) {
            Eigen::Vector3d pos2 = getChannelPosition(info.chs[j]);
            
            double distance = computeDistance(pos1, pos2);
            
            if (distance <= maxDistance && distance > 0) {
                double weight = 1.0 / (1.0 + distance); // Distance-based weighting
                triplets.push_back(Eigen::Triplet<double>(i, j, weight));
                triplets.push_back(Eigen::Triplet<double>(j, i, weight));
            }
        }
        
        // Self-connection
        triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
    }
    
    adjacency.setFromTriplets(triplets.begin(), triplets.end());
    return adjacency;
}

//=============================================================================================================

QStringList ChannelManager::findNearbyChannels(const FiffInfo& info,
                                                const QString& centerChannel,
                                                double maxDistance)
{
    QStringList nearbyChannels;
    
    int centerIdx = info.ch_names.indexOf(centerChannel);
    if (centerIdx < 0) {
        qWarning() << "Center channel not found:" << centerChannel;
        return nearbyChannels;
    }
    
    Eigen::Vector3d centerPos = getChannelPosition(info.chs[centerIdx]);
    
    for (int i = 0; i < info.nchan; ++i) {
        if (i == centerIdx) {
            continue; // Skip center channel itself
        }
        
        Eigen::Vector3d pos = getChannelPosition(info.chs[i]);
        double distance = computeDistance(centerPos, pos);
        
        if (distance <= maxDistance) {
            nearbyChannels.append(info.ch_names[i]);
        }
    }
    
    return nearbyChannels;
}

//=============================================================================================================

Eigen::VectorXi ChannelManager::selectChannelsByType(const FiffInfo& info,
                                                      const QList<int>& channelTypes)
{
    QList<int> selectedIndices;
    
    for (int i = 0; i < info.nchan; ++i) {
        if (channelTypes.contains(info.chs[i].kind)) {
            selectedIndices.append(i);
        }
    }
    
    Eigen::VectorXi indices(selectedIndices.size());
    for (int i = 0; i < selectedIndices.size(); ++i) {
        indices[i] = selectedIndices[i];
    }
    
    return indices;
}
//=============================================================================================================

Eigen::VectorXi ChannelManager::selectChannelsByPattern(const FiffInfo& info,
                                                         const QString& pattern)
{
    QList<int> selectedIndices;
    QRegularExpression regex(pattern);
    
    for (int i = 0; i < info.nchan; ++i) {
        if (regex.match(info.ch_names[i]).hasMatch()) {
            selectedIndices.append(i);
        }
    }
    
    Eigen::VectorXi indices(selectedIndices.size());
    for (int i = 0; i < selectedIndices.size(); ++i) {
        indices[i] = selectedIndices[i];
    }
    
    return indices;
}

//=============================================================================================================

bool ChannelManager::dropBadChannels(FiffInfo& info,
                                      const QStringList& badChannels)
{
    QList<FiffChInfo> goodChannels;
    QStringList goodChannelNames;
    
    for (int i = 0; i < info.nchan; ++i) {
        const QString& chName = info.ch_names[i];
        if (!badChannels.contains(chName)) {
            goodChannels.append(info.chs[i]);
            goodChannelNames.append(chName);
        }
    }
    
    if (goodChannels.size() == info.nchan) {
        return false; // No channels were dropped
    }
    
    info.chs = goodChannels;
    info.ch_names = goodChannelNames;
    info.nchan = goodChannels.size();
    
    return true;
}

//=============================================================================================================

bool ChannelManager::interpolateBadChannels(Eigen::MatrixXd& data,
                                             const FiffInfo& info,
                                             const QStringList& badChannels)
{
    if (badChannels.isEmpty()) {
        return true;
    }
    
    // Compute adjacency matrix for interpolation weights
    Eigen::SparseMatrix<double> adjacency = computeChannelAdjacency(info, 0.1); // 10cm max distance
    
    for (const QString& badChannel : badChannels) {
        int badIdx = info.ch_names.indexOf(badChannel);
        if (badIdx < 0) {
            continue;
        }
        
        // Find neighboring channels
        QList<int> neighbors;
        QList<double> weights;
        double totalWeight = 0.0;
        
        for (int i = 0; i < info.nchan; ++i) {
            if (i != badIdx && adjacency.coeff(badIdx, i) > 0) {
                neighbors.append(i);
                double weight = adjacency.coeff(badIdx, i);
                weights.append(weight);
                totalWeight += weight;
            }
        }
        
        if (neighbors.isEmpty()) {
            qWarning() << "No neighbors found for bad channel:" << badChannel;
            continue;
        }
        
        // Interpolate using weighted average
        for (int sample = 0; sample < data.cols(); ++sample) {
            double interpolatedValue = 0.0;
            
            for (int i = 0; i < neighbors.size(); ++i) {
                int neighborIdx = neighbors[i];
                double weight = weights[i] / totalWeight;
                interpolatedValue += weight * data(neighborIdx, sample);
            }
            
            data(badIdx, sample) = interpolatedValue;
        }
    }
    
    return true;
}
//=============================================================================================================

Eigen::MatrixXd ChannelManager::computeChannelDistances(const FiffInfo& info)
{
    int nChannels = info.nchan;
    Eigen::MatrixXd distances(nChannels, nChannels);
    
    for (int i = 0; i < nChannels; ++i) {
        Eigen::Vector3d pos1 = getChannelPosition(info.chs[i]);
        
        for (int j = 0; j < nChannels; ++j) {
            if (i == j) {
                distances(i, j) = 0.0;
            } else {
                Eigen::Vector3d pos2 = getChannelPosition(info.chs[j]);
                distances(i, j) = computeDistance(pos1, pos2);
            }
        }
    }
    
    return distances;
}

//=============================================================================================================

FiffInfo ChannelManager::createBipolarMontage(const FiffInfo& info,
                                               const QList<QPair<QString, QString>>& pairs)
{
    FiffInfo bipolarInfo = info;
    bipolarInfo.chs.clear();
    bipolarInfo.ch_names.clear();
    
    for (const auto& pair : pairs) {
        const QString& ch1Name = pair.first;
        const QString& ch2Name = pair.second;
        
        int idx1 = info.ch_names.indexOf(ch1Name);
        int idx2 = info.ch_names.indexOf(ch2Name);
        
        if (idx1 < 0 || idx2 < 0) {
            qWarning() << "Channel pair not found:" << ch1Name << "-" << ch2Name;
            continue;
        }
        
        // Create bipolar channel
        FiffChInfo bipolarChannel = info.chs[idx1]; // Use first channel as template
        bipolarChannel.ch_name = ch1Name + "-" + ch2Name;
        
        // Average positions
        Eigen::Vector3d pos1 = getChannelPosition(info.chs[idx1]);
        Eigen::Vector3d pos2 = getChannelPosition(info.chs[idx2]);
        Eigen::Vector3d avgPos = (pos1 + pos2) / 2.0;
        
        bipolarChannel.eeg_loc(0, 0) = avgPos.x();
        bipolarChannel.eeg_loc(1, 0) = avgPos.y();
        bipolarChannel.eeg_loc(2, 0) = avgPos.z();
        
        bipolarInfo.chs.append(bipolarChannel);
        bipolarInfo.ch_names.append(bipolarChannel.ch_name);
    }
    
    bipolarInfo.nchan = bipolarInfo.chs.size();
    return bipolarInfo;
}

//=============================================================================================================

Eigen::MatrixXd ChannelManager::createAverageReference(const FiffInfo& info,
                                                        const QStringList& refChannels)
{
    int nChannels = info.nchan;
    Eigen::MatrixXd refMatrix = Eigen::MatrixXd::Identity(nChannels, nChannels);
    
    // Determine which channels to use for reference
    QList<int> refIndices;
    
    if (refChannels.isEmpty()) {
        // Use all EEG channels
        for (int i = 0; i < nChannels; ++i) {
            if (info.chs[i].kind == FIFFV_EEG_CH) {
                refIndices.append(i);
            }
        }
    } else {
        // Use specified channels
        for (const QString& chName : refChannels) {
            int idx = info.ch_names.indexOf(chName);
            if (idx >= 0) {
                refIndices.append(idx);
            }
        }
    }
    
    if (refIndices.isEmpty()) {
        qWarning() << "No reference channels found";
        return refMatrix;
    }
    
    // Create average reference matrix
    double refWeight = 1.0 / refIndices.size();
    
    for (int i = 0; i < nChannels; ++i) {
        for (int refIdx : refIndices) {
            refMatrix(i, refIdx) -= refWeight;
        }
    }
    
    return refMatrix;
}
//=============================================================================================================

Eigen::Vector3d ChannelManager::getChannelPosition(const FiffChInfo& chInfo)
{
    // Try to get position from eeg_loc first
    if (chInfo.eeg_loc.rows() >= 3 && chInfo.eeg_loc.cols() >= 1) {
        return Eigen::Vector3d(chInfo.eeg_loc(0, 0), chInfo.eeg_loc(1, 0), chInfo.eeg_loc(2, 0));
    }
    
    // Fallback to chpos if available
    // Note: This would require accessing chpos.r which might not be directly available
    // This is a simplified implementation
    return Eigen::Vector3d::Zero();
}

//=============================================================================================================

double ChannelManager::computeDistance(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2)
{
    return (pos1 - pos2).norm();
}