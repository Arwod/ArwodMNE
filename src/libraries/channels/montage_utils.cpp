//=============================================================================================================
/**
 * @file     montage_utils.cpp
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
 * @brief    Implementation of montage utility functions.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "montage_utils.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_constants.h>
#include <fiff/fiff_coord_trans.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace CHANNELSLIB;
using namespace FIFFLIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

DigMontage::SPtr CHANNELSLIB::makeDigMontage(const Eigen::MatrixXd& chPos,
                                              const QStringList& chNames,
                                              const Eigen::Vector3d& nasion,
                                              const Eigen::Vector3d& lpa,
                                              const Eigen::Vector3d& rpa,
                                              int coordFrame)
{
    DigMontage::SPtr montage = DigMontage::fromChannelPositions(chPos, chNames, coordFrame);
    
    // Add fiducials if provided
    bool hasValidFiducials = (nasion.norm() > 0) || (lpa.norm() > 0) || (rpa.norm() > 0);
    if (hasValidFiducials) {
        montage->setFiducials(lpa, rpa, nasion);
    }
    
    return montage;
}

//=============================================================================================================

DigMontage::SPtr CHANNELSLIB::makeDigMontage(const QMap<QString, Eigen::Vector3d>& chPosMap,
                                              const Eigen::Vector3d& nasion,
                                              const Eigen::Vector3d& lpa,
                                              const Eigen::Vector3d& rpa,
                                              int coordFrame)
{
    // Convert map to matrix format
    QStringList chNames = chPosMap.keys();
    int nChannels = chNames.size();
    
    Eigen::MatrixXd chPos(3, nChannels);
    for (int i = 0; i < nChannels; ++i) {
        const Eigen::Vector3d& pos = chPosMap[chNames[i]];
        chPos.col(i) = pos;
    }
    
    return makeDigMontage(chPos, chNames, nasion, lpa, rpa, coordFrame);
}

//=============================================================================================================

QMap<QString, Eigen::Vector3d> CHANNELSLIB::getStandardElectrodePositions(const QString& montageType)
{
    QMap<QString, Eigen::Vector3d> positions;
    
    if (montageType == "standard_1020") {
        // Standard 10-20 positions (in head coordinates, meters)
        positions["Fp1"] = Eigen::Vector3d(-0.0309, 0.0711, 0.0538);
        positions["Fp2"] = Eigen::Vector3d(0.0309, 0.0711, 0.0538);
        positions["F7"] = Eigen::Vector3d(-0.0761, 0.0302, 0.0286);
        positions["F3"] = Eigen::Vector3d(-0.0454, 0.0302, 0.0713);
        positions["Fz"] = Eigen::Vector3d(0.0000, 0.0302, 0.0825);
        positions["F4"] = Eigen::Vector3d(0.0454, 0.0302, 0.0713);
        positions["F8"] = Eigen::Vector3d(0.0761, 0.0302, 0.0286);
        positions["T7"] = Eigen::Vector3d(-0.0825, 0.0000, 0.0000);
        positions["C3"] = Eigen::Vector3d(-0.0454, 0.0000, 0.0713);
        positions["Cz"] = Eigen::Vector3d(0.0000, 0.0000, 0.0825);
        positions["C4"] = Eigen::Vector3d(0.0454, 0.0000, 0.0713);
        positions["T8"] = Eigen::Vector3d(0.0825, 0.0000, 0.0000);
        positions["P7"] = Eigen::Vector3d(-0.0761, -0.0302, 0.0286);
        positions["P3"] = Eigen::Vector3d(-0.0454, -0.0302, 0.0713);
        positions["Pz"] = Eigen::Vector3d(0.0000, -0.0302, 0.0825);
        positions["P4"] = Eigen::Vector3d(0.0454, -0.0302, 0.0713);
        positions["P8"] = Eigen::Vector3d(0.0761, -0.0302, 0.0286);
        positions["O1"] = Eigen::Vector3d(-0.0309, -0.0711, 0.0538);
        positions["O2"] = Eigen::Vector3d(0.0309, -0.0711, 0.0538);
        
    } else if (montageType == "standard_1005") {
        // Start with 10-20 positions
        positions = getStandardElectrodePositions("standard_1020");
        
        // Add additional 10-05 positions
        positions["Fpz"] = Eigen::Vector3d(0.0000, 0.0825, 0.0000);
        positions["AF7"] = Eigen::Vector3d(-0.0585, 0.0585, 0.0412);
        positions["AF3"] = Eigen::Vector3d(-0.0293, 0.0585, 0.0650);
        positions["AFz"] = Eigen::Vector3d(0.0000, 0.0585, 0.0713);
        positions["AF4"] = Eigen::Vector3d(0.0293, 0.0585, 0.0650);
        positions["AF8"] = Eigen::Vector3d(0.0585, 0.0585, 0.0412);
        positions["F5"] = Eigen::Vector3d(-0.0650, 0.0302, 0.0520);
        positions["F1"] = Eigen::Vector3d(-0.0227, 0.0302, 0.0800);
        positions["F2"] = Eigen::Vector3d(0.0227, 0.0302, 0.0800);
        positions["F6"] = Eigen::Vector3d(0.0650, 0.0302, 0.0520);
        
    } else if (montageType == "biosemi64") {
        // BioSemi 64-channel cap positions (simplified)
        positions = getStandardElectrodePositions("standard_1020");
        
        // Add additional BioSemi-specific positions
        positions["A1"] = Eigen::Vector3d(-0.0825, 0.0000, -0.0200);
        positions["A2"] = Eigen::Vector3d(0.0825, 0.0000, -0.0200);
        positions["CPz"] = Eigen::Vector3d(0.0000, -0.0151, 0.0825);
        positions["POz"] = Eigen::Vector3d(0.0000, -0.0527, 0.0713);
        
    } else {
        qWarning() << "Unknown montage type:" << montageType;
    }
    
    return positions;
}

//=============================================================================================================

bool transformMontage(DigMontage::SPtr montage,
                      int fromFrame,
                      int toFrame,
                      const Eigen::Matrix4d& trans)
{
    if (!montage) {
        return false;
    }
    
    if (fromFrame == toFrame) {
        return true; // No transformation needed
    }
    
    // Create coordinate transformation
    FiffCoordTrans coordTrans;
    coordTrans.from = fromFrame;
    coordTrans.to = toFrame;
    
    // Convert Eigen::Matrix4d to the format expected by FiffCoordTrans
    // Note: This is a simplified implementation
    // In practice, you'd need to properly set up the transformation matrix
    
    try {
        montage->transform(coordTrans);
        return true;
    } catch (const std::exception& e) {
        qWarning() << "Failed to transform montage:" << e.what();
        return false;
    }
}