//=============================================================================================================
/**
 * @file     montage_utils.h
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
 * @brief    Utility functions for montage creation and management
 *
 */

#ifndef MONTAGE_UTILS_H
#define MONTAGE_UTILS_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "channels_global.h"
#include "dig_montage.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QString>
#include <QStringList>
#include <QMap>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>

//=============================================================================================================
// DEFINE NAMESPACE CHANNELSLIB
//=============================================================================================================

namespace CHANNELSLIB
{

//=============================================================================================================
/**
 * Create a DigMontage from channel positions (equivalent to MNE-Python's make_dig_montage)
 *
 * @param[in] chPos         Channel positions as 3xN matrix or map of channel names to positions
 * @param[in] chNames       Channel names (if using matrix)
 * @param[in] nasion        Nasion position (optional)
 * @param[in] lpa           Left preauricular position (optional)
 * @param[in] rpa           Right preauricular position (optional)
 * @param[in] coordFrame    Coordinate frame
 *
 * @return Shared pointer to DigMontage
 */
CHANNELSSHARED_EXPORT DigMontage::SPtr makeDigMontage(const Eigen::MatrixXd& chPos,
                                                      const QStringList& chNames,
                                                      const Eigen::Vector3d& nasion = Eigen::Vector3d::Zero(),
                                                      const Eigen::Vector3d& lpa = Eigen::Vector3d::Zero(),
                                                      const Eigen::Vector3d& rpa = Eigen::Vector3d::Zero(),
                                                      int coordFrame = FIFFV_COORD_HEAD);

//=============================================================================================================
/**
 * Create a DigMontage from channel position map
 *
 * @param[in] chPosMap      Map of channel names to 3D positions
 * @param[in] nasion        Nasion position (optional)
 * @param[in] lpa           Left preauricular position (optional)
 * @param[in] rpa           Right preauricular position (optional)
 * @param[in] coordFrame    Coordinate frame
 *
 * @return Shared pointer to DigMontage
 */
CHANNELSSHARED_EXPORT DigMontage::SPtr makeDigMontage(const QMap<QString, Eigen::Vector3d>& chPosMap,
                                                      const Eigen::Vector3d& nasion = Eigen::Vector3d::Zero(),
                                                      const Eigen::Vector3d& lpa = Eigen::Vector3d::Zero(),
                                                      const Eigen::Vector3d& rpa = Eigen::Vector3d::Zero(),
                                                      int coordFrame = FIFFV_COORD_HEAD);

//=============================================================================================================
/**
 * Get standard electrode positions for common montages
 *
 * @param[in] montageType   Type of montage ("standard_1020", "standard_1005", "biosemi64", etc.)
 *
 * @return Map of electrode names to positions
 */
CHANNELSSHARED_EXPORT QMap<QString, Eigen::Vector3d> getStandardElectrodePositions(const QString& montageType);

//=============================================================================================================
/**
 * Transform montage between coordinate systems
 *
 * @param[in,out] montage   Montage to transform
 * @param[in] fromFrame     Source coordinate frame
 * @param[in] toFrame       Target coordinate frame
 * @param[in] trans         Transformation matrix (4x4)
 *
 * @return true if successful, false otherwise
 */
CHANNELSSHARED_EXPORT bool transformMontage(DigMontage::SPtr montage,
                                            int fromFrame,
                                            int toFrame,
                                            const Eigen::Matrix4d& trans = Eigen::Matrix4d::Identity());

} // NAMESPACE CHANNELSLIB

#endif // MONTAGE_UTILS_H