//=============================================================================================================
/**
 * @file     channel_manager.h
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
 * @brief    ChannelManager class for channel operations and management
 *
 */

#ifndef CHANNEL_MANAGER_H
#define CHANNEL_MANAGER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "channels_global.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_info.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QObject>
#include <QString>
#include <QStringList>
#include <QSharedPointer>
#include <QMap>
#include <QList>
#include <QPair>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>
#include <Eigen/Sparse>

//=============================================================================================================
// DEFINE NAMESPACE CHANNELSLIB
//=============================================================================================================

namespace CHANNELSLIB
{
//=============================================================================================================
/**
 * ChannelManager class for managing channel operations and adjacency
 *
 * @brief ChannelManager class for managing channel operations and adjacency
 */
class CHANNELSSHARED_EXPORT ChannelManager : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<ChannelManager> SPtr;            /**< Shared pointer type for ChannelManager. */
    typedef QSharedPointer<const ChannelManager> ConstSPtr; /**< Const shared pointer type for ChannelManager. */

    //=========================================================================================================
    /**
     * Constructs a ChannelManager
     */
    explicit ChannelManager(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destroys the ChannelManager
     */
    ~ChannelManager();

    //=========================================================================================================
    /**
     * Combine channels by averaging
     *
     * @param[in,out] info      FiffInfo to modify
     * @param[in] groups        Channel groups to combine (map of new name to list of channels)
     *
     * @return true if successful, false otherwise
     */
    static bool combineChannels(FIFFLIB::FiffInfo& info,
                                const QMap<QString, QStringList>& groups);

    //=========================================================================================================
    /**
     * Equalize channel counts across different channel types
     *
     * @param[in,out] infos     List of FiffInfo objects to equalize
     *
     * @return true if successful, false otherwise
     */
    static bool equalizeChannels(QList<FIFFLIB::FiffInfo>& infos);

    //=========================================================================================================
    /**
     * Rename channels
     *
     * @param[in,out] info      FiffInfo to modify
     * @param[in] mapping       Channel name mapping (old name -> new name)
     *
     * @return true if successful, false otherwise
     */
    static bool renameChannels(FIFFLIB::FiffInfo& info,
                               const QMap<QString, QString>& mapping);

    //=========================================================================================================
    /**
     * Compute channel adjacency matrix
     *
     * @param[in] info          FiffInfo containing channel positions
     * @param[in] maxDistance   Maximum distance for adjacency (in meters)
     *
     * @return Sparse adjacency matrix
     */
    static Eigen::SparseMatrix<double> computeChannelAdjacency(const FIFFLIB::FiffInfo& info,
                                                               double maxDistance = 0.05);

    //=========================================================================================================
    /**
     * Find channels within a certain distance
     *
     * @param[in] info          FiffInfo containing channel positions
     * @param[in] centerChannel Center channel name
     * @param[in] maxDistance   Maximum distance (in meters)
     *
     * @return List of nearby channel names
     */
    static QStringList findNearbyChannels(const FIFFLIB::FiffInfo& info,
                                           const QString& centerChannel,
                                           double maxDistance = 0.05);
    //=========================================================================================================
    /**
     * Select channels by type
     *
     * @param[in] info          FiffInfo to select from
     * @param[in] channelTypes  Channel types to include (FIFFV_EEG_CH, FIFFV_MEG_CH, etc.)
     *
     * @return Indices of selected channels
     */
    static Eigen::VectorXi selectChannelsByType(const FIFFLIB::FiffInfo& info,
                                                 const QList<int>& channelTypes);

    //=========================================================================================================
    /**
     * Select channels by name pattern
     *
     * @param[in] info          FiffInfo to select from
     * @param[in] pattern       Regular expression pattern
     *
     * @return Indices of matching channels
     */
    static Eigen::VectorXi selectChannelsByPattern(const FIFFLIB::FiffInfo& info,
                                                    const QString& pattern);

    //=========================================================================================================
    /**
     * Drop bad channels from info
     *
     * @param[in,out] info      FiffInfo to modify
     * @param[in] badChannels   List of bad channel names
     *
     * @return true if successful, false otherwise
     */
    static bool dropBadChannels(FIFFLIB::FiffInfo& info,
                                const QStringList& badChannels);

    //=========================================================================================================
    /**
     * Interpolate bad channels using nearby good channels
     *
     * @param[in,out] data      Data matrix (channels x samples)
     * @param[in] info          FiffInfo containing channel information
     * @param[in] badChannels   List of bad channel names
     *
     * @return true if successful, false otherwise
     */
    static bool interpolateBadChannels(Eigen::MatrixXd& data,
                                       const FIFFLIB::FiffInfo& info,
                                       const QStringList& badChannels);

    //=========================================================================================================
    /**
     * Compute channel distances matrix
     *
     * @param[in] info          FiffInfo containing channel positions
     *
     * @return Distance matrix (channels x channels)
     */
    static Eigen::MatrixXd computeChannelDistances(const FIFFLIB::FiffInfo& info);

    //=========================================================================================================
    /**
     * Create bipolar montage
     *
     * @param[in] info          Original FiffInfo
     * @param[in] pairs         Channel pairs for bipolar derivation
     *
     * @return New FiffInfo with bipolar channels
     */
    static FIFFLIB::FiffInfo createBipolarMontage(const FIFFLIB::FiffInfo& info,
                                                   const QList<QPair<QString, QString>>& pairs);

    //=========================================================================================================
    /**
     * Create average reference montage
     *
     * @param[in] info          Original FiffInfo
     * @param[in] refChannels   Channels to use for reference (empty = all EEG)
     *
     * @return Reference projection matrix
     */
    static Eigen::MatrixXd createAverageReference(const FIFFLIB::FiffInfo& info,
                                                   const QStringList& refChannels = QStringList());

private:
    //=========================================================================================================
    /**
     * Get channel position from FiffChInfo
     *
     * @param[in] chInfo        Channel info
     *
     * @return 3D position vector
     */
    static Eigen::Vector3d getChannelPosition(const FIFFLIB::FiffChInfo& chInfo);

    //=========================================================================================================
    /**
     * Compute distance between two 3D points
     *
     * @param[in] pos1          First position
     * @param[in] pos2          Second position
     *
     * @return Euclidean distance
     */
    static double computeDistance(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE CHANNELSLIB

#endif // CHANNEL_MANAGER_H