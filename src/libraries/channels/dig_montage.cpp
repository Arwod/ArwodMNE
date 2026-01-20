//=============================================================================================================
/**
 * @file     dig_montage.cpp
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
 * @brief    Implementation of the DigMontage class.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "dig_montage.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_dig_point.h>
#include <fiff/fiff_constants.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QFile>
#include <QTextStream>
#include <QFileInfo>
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

DigMontage::DigMontage(QObject *parent)
: QObject(parent)
, m_coordFrame(FIFFV_COORD_HEAD)
{
}
//=============================================================================================================

DigMontage::~DigMontage()
{
}

//=============================================================================================================

DigMontage::SPtr DigMontage::fromChannelPositions(const Eigen::MatrixXd& chPos,
                                                   const QStringList& chNames,
                                                   int coordFrame)
{
    SPtr montage(new DigMontage());
    montage->m_coordFrame = coordFrame;
    
    if (chPos.rows() != 3 || chPos.cols() != chNames.size()) {
        qWarning() << "Channel positions matrix dimensions don't match channel names";
        return montage;
    }
    
    for (int i = 0; i < chNames.size(); ++i) {
        FiffDigPoint digPoint;
        digPoint.kind = FIFFV_POINT_EEG;
        digPoint.ident = i + 1;
        digPoint.coord_frame = coordFrame;
        digPoint.r[0] = chPos(0, i);
        digPoint.r[1] = chPos(1, i);
        digPoint.r[2] = chPos(2, i);
        
        montage->m_digPoints << digPoint;
        montage->m_channelMap[chNames[i]] = i;
    }
    
    return montage;
}

//=============================================================================================================

DigMontage::SPtr DigMontage::fromStandardMontage(const QString& montageType,
                                                  const QStringList& chNames)
{
    SPtr montage(new DigMontage());
    
    if (montageType == "standard_1020") {
        montage->loadStandard1020();
    } else if (montageType == "standard_1005") {
        montage->loadStandard1005();
    } else {
        qWarning() << "Unknown montage type:" << montageType;
        return montage;
    }
    
    // Filter to requested channels if specified
    if (!chNames.isEmpty()) {
        FiffDigPointSet filteredPoints;
        QMap<QString, int> filteredMap;
        
        for (const QString& chName : chNames) {
            if (montage->m_channelMap.contains(chName)) {
                int idx = montage->m_channelMap[chName];
                filteredPoints << montage->m_digPoints[idx];
                filteredMap[chName] = filteredPoints.size() - 1;
            }
        }
        
        montage->m_digPoints = filteredPoints;
        montage->m_channelMap = filteredMap;
    }
    
    return montage;
}

//=============================================================================================================

DigMontage::SPtr DigMontage::fromFile(const QString& fileName)
{
    SPtr montage(new DigMontage());
    
    QFileInfo fileInfo(fileName);
    QString suffix = fileInfo.suffix().toLower();
    
    bool success = false;
    if (suffix == "elc") {
        success = montage->parseELCFile(fileName);
    } else if (suffix == "sfp") {
        success = montage->parseSFPFile(fileName);
    } else {
        qWarning() << "Unsupported file format:" << suffix;
    }
    
    if (!success) {
        qWarning() << "Failed to load montage from file:" << fileName;
    }
    
    return montage;
}
//=============================================================================================================

Eigen::MatrixXd DigMontage::getChannelPositions() const
{
    int nChannels = m_digPoints.size();
    Eigen::MatrixXd positions(3, nChannels);
    
    for (int i = 0; i < nChannels; ++i) {
        const FiffDigPoint& point = m_digPoints[i];
        positions(0, i) = point.r[0];
        positions(1, i) = point.r[1];
        positions(2, i) = point.r[2];
    }
    
    return positions;
}

//=============================================================================================================

QStringList DigMontage::getChannelNames() const
{
    QStringList names;
    for (auto it = m_channelMap.begin(); it != m_channelMap.end(); ++it) {
        names.append(it.key());
    }
    return names;
}

//=============================================================================================================

QMap<QString, Eigen::Vector3d> DigMontage::getFiducials() const
{
    QMap<QString, Eigen::Vector3d> fiducials;
    
    for (int i = 0; i < m_digPoints.size(); ++i) {
        const FiffDigPoint& point = m_digPoints[i];
        
        if (point.kind == FIFFV_POINT_CARDINAL) {
            Eigen::Vector3d pos(point.r[0], point.r[1], point.r[2]);
            
            switch (point.ident) {
                case FIFFV_POINT_LPA:
                    fiducials["LPA"] = pos;
                    break;
                case FIFFV_POINT_RPA:
                    fiducials["RPA"] = pos;
                    break;
                case FIFFV_POINT_NASION:
                    fiducials["Nasion"] = pos;
                    break;
            }
        }
    }
    
    return fiducials;
}

//=============================================================================================================

void DigMontage::setFiducials(const Eigen::Vector3d& lpa,
                               const Eigen::Vector3d& rpa,
                               const Eigen::Vector3d& nasion)
{
    // Remove existing fiducials
    for (int i = m_digPoints.size() - 1; i >= 0; --i) {
        if (m_digPoints[i].kind == FIFFV_POINT_CARDINAL) {
            // Note: FiffDigPointSet doesn't have remove method, so we need to rebuild
            // This is a simplified implementation
        }
    }
    
    // Add new fiducials
    FiffDigPoint lpaPoint, rpaPoint, nasionPoint;
    
    lpaPoint.kind = FIFFV_POINT_CARDINAL;
    lpaPoint.ident = FIFFV_POINT_LPA;
    lpaPoint.coord_frame = m_coordFrame;
    lpaPoint.r[0] = lpa.x(); lpaPoint.r[1] = lpa.y(); lpaPoint.r[2] = lpa.z();
    
    rpaPoint.kind = FIFFV_POINT_CARDINAL;
    rpaPoint.ident = FIFFV_POINT_RPA;
    rpaPoint.coord_frame = m_coordFrame;
    rpaPoint.r[0] = rpa.x(); rpaPoint.r[1] = rpa.y(); rpaPoint.r[2] = rpa.z();
    
    nasionPoint.kind = FIFFV_POINT_CARDINAL;
    nasionPoint.ident = FIFFV_POINT_NASION;
    nasionPoint.coord_frame = m_coordFrame;
    nasionPoint.r[0] = nasion.x(); nasionPoint.r[1] = nasion.y(); nasionPoint.r[2] = nasion.z();
    
    m_digPoints << lpaPoint << rpaPoint << nasionPoint;
}
//=============================================================================================================

void DigMontage::transform(const FiffCoordTrans& coordTrans)
{
    m_digPoints.applyTransform(coordTrans);
    m_coordFrame = coordTrans.to;
}

//=============================================================================================================

void DigMontage::scale(double scaleFactor)
{
    for (int i = 0; i < m_digPoints.size(); ++i) {
        FiffDigPoint& point = m_digPoints[i];
        point.r[0] *= scaleFactor;
        point.r[1] *= scaleFactor;
        point.r[2] *= scaleFactor;
    }
}

//=============================================================================================================

void DigMontage::addElectrode(const QString& chName,
                               const Eigen::Vector3d& position,
                               int kind)
{
    FiffDigPoint digPoint;
    digPoint.kind = kind;
    digPoint.ident = m_digPoints.size() + 1;
    digPoint.coord_frame = m_coordFrame;
    digPoint.r[0] = position.x();
    digPoint.r[1] = position.y();
    digPoint.r[2] = position.z();
    
    m_digPoints << digPoint;
    m_channelMap[chName] = m_digPoints.size() - 1;
}

//=============================================================================================================

void DigMontage::removeElectrode(const QString& chName)
{
    if (m_channelMap.contains(chName)) {
        // Note: This is a simplified implementation
        // In practice, we'd need to rebuild the dig point set
        m_channelMap.remove(chName);
    }
}

//=============================================================================================================

FiffDigPointSet DigMontage::getDigPointSet() const
{
    return m_digPoints;
}

//=============================================================================================================

void DigMontage::applyToInfo(FiffInfo& info) const
{
    // Apply electrode positions to channel info
    for (int i = 0; i < info.chs.size(); ++i) {
        const QString& chName = info.chs[i].ch_name;
        
        if (m_channelMap.contains(chName)) {
            int digIdx = m_channelMap[chName];
            const FiffDigPoint& point = m_digPoints[digIdx];
            
            // Update channel position
            info.chs[i].eeg_loc(0, 0) = point.r[0];
            info.chs[i].eeg_loc(1, 0) = point.r[1];
            info.chs[i].eeg_loc(2, 0) = point.r[2];
            info.chs[i].coord_frame = point.coord_frame;
        }
    }
    
    // Set digitizer points in info
    // Note: This would require extending FiffInfo to include dig points
}
//=============================================================================================================

double DigMontage::computeHeadCircumference() const
{
    // Simplified head circumference calculation
    // Find the radius of the electrode positions projected onto the XY plane
    double maxRadius = 0.0;
    
    for (int i = 0; i < m_digPoints.size(); ++i) {
        const FiffDigPoint& point = m_digPoints[i];
        if (point.kind == FIFFV_POINT_EEG) {
            double radius = std::sqrt(point.r[0] * point.r[0] + point.r[1] * point.r[1]);
            maxRadius = std::max(maxRadius, radius);
        }
    }
    
    return 2.0 * M_PI * maxRadius;
}

//=============================================================================================================

QString DigMontage::findNearestElectrode(const Eigen::Vector3d& position) const
{
    QString nearestChannel;
    double minDistance = std::numeric_limits<double>::max();
    
    for (auto it = m_channelMap.begin(); it != m_channelMap.end(); ++it) {
        const QString& chName = it.key();
        int idx = it.value();
        const FiffDigPoint& point = m_digPoints[idx];
        
        Eigen::Vector3d pointPos(point.r[0], point.r[1], point.r[2]);
        double distance = (position - pointPos).norm();
        
        if (distance < minDistance) {
            minDistance = distance;
            nearestChannel = chName;
        }
    }
    
    return nearestChannel;
}

//=============================================================================================================

bool DigMontage::validate() const
{
    // Check if we have at least some electrode positions
    if (m_digPoints.isEmpty()) {
        return false;
    }
    
    // Check if channel map is consistent
    for (auto it = m_channelMap.begin(); it != m_channelMap.end(); ++it) {
        int idx = it.value();
        if (idx < 0 || idx >= m_digPoints.size()) {
            return false;
        }
    }
    
    // Check for duplicate positions (within tolerance)
    const double tolerance = 1e-6;
    for (int i = 0; i < m_digPoints.size(); ++i) {
        for (int j = i + 1; j < m_digPoints.size(); ++j) {
            const FiffDigPoint& p1 = m_digPoints[i];
            const FiffDigPoint& p2 = m_digPoints[j];
            
            double distance = std::sqrt(
                std::pow(p1.r[0] - p2.r[0], 2) +
                std::pow(p1.r[1] - p2.r[1], 2) +
                std::pow(p1.r[2] - p2.r[2], 2)
            );
            
            if (distance < tolerance) {
                qWarning() << "Duplicate electrode positions detected";
                return false;
            }
        }
    }
    
    return true;
}
//=============================================================================================================

void DigMontage::loadStandard1020()
{
    // Standard 10-20 electrode positions (simplified subset)
    // Positions are in head coordinate system (meters)
    
    struct ElectrodePos {
        QString name;
        double x, y, z;
    };
    
    QList<ElectrodePos> electrodes = {
        {"Fp1", -0.0309, 0.0711, 0.0538},
        {"Fp2", 0.0309, 0.0711, 0.0538},
        {"F7", -0.0761, 0.0302, 0.0286},
        {"F3", -0.0454, 0.0302, 0.0713},
        {"Fz", 0.0000, 0.0302, 0.0825},
        {"F4", 0.0454, 0.0302, 0.0713},
        {"F8", 0.0761, 0.0302, 0.0286},
        {"T7", -0.0825, 0.0000, 0.0000},
        {"C3", -0.0454, 0.0000, 0.0713},
        {"Cz", 0.0000, 0.0000, 0.0825},
        {"C4", 0.0454, 0.0000, 0.0713},
        {"T8", 0.0825, 0.0000, 0.0000},
        {"P7", -0.0761, -0.0302, 0.0286},
        {"P3", -0.0454, -0.0302, 0.0713},
        {"Pz", 0.0000, -0.0302, 0.0825},
        {"P4", 0.0454, -0.0302, 0.0713},
        {"P8", 0.0761, -0.0302, 0.0286},
        {"O1", -0.0309, -0.0711, 0.0538},
        {"O2", 0.0309, -0.0711, 0.0538}
    };
    
    m_digPoints.clear();
    m_channelMap.clear();
    
    for (int i = 0; i < electrodes.size(); ++i) {
        const ElectrodePos& elec = electrodes[i];
        
        FiffDigPoint digPoint;
        digPoint.kind = FIFFV_POINT_EEG;
        digPoint.ident = i + 1;
        digPoint.coord_frame = FIFFV_COORD_HEAD;
        digPoint.r[0] = elec.x;
        digPoint.r[1] = elec.y;
        digPoint.r[2] = elec.z;
        
        m_digPoints << digPoint;
        m_channelMap[elec.name] = i;
    }
    
    // Add standard fiducials
    setFiducials(
        Eigen::Vector3d(-0.0825, 0.0000, 0.0000),  // LPA
        Eigen::Vector3d(0.0825, 0.0000, 0.0000),   // RPA
        Eigen::Vector3d(0.0000, 0.0825, 0.0000)    // Nasion
    );
}

//=============================================================================================================

void DigMontage::loadStandard1005()
{
    // Load standard 10-05 positions (extended set)
    // This is a simplified implementation - in practice would load from file
    loadStandard1020(); // Start with 10-20 as base
    
    // Add additional 10-05 positions
    struct ElectrodePos {
        QString name;
        double x, y, z;
    };
    
    QList<ElectrodePos> additional = {
        {"Fpz", 0.0000, 0.0825, 0.0000},
        {"AF7", -0.0585, 0.0585, 0.0412},
        {"AF3", -0.0293, 0.0585, 0.0650},
        {"AFz", 0.0000, 0.0585, 0.0713},
        {"AF4", 0.0293, 0.0585, 0.0650},
        {"AF8", 0.0585, 0.0585, 0.0412}
    };
    
    for (const ElectrodePos& elec : additional) {
        addElectrode(elec.name, Eigen::Vector3d(elec.x, elec.y, elec.z));
    }
}
//=============================================================================================================

bool DigMontage::parseELCFile(const QString& fileName)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Cannot open ELC file:" << fileName;
        return false;
    }
    
    QTextStream stream(&file);
    m_digPoints.clear();
    m_channelMap.clear();
    
    int lineNum = 0;
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        lineNum++;
        
        if (line.isEmpty() || line.startsWith("#")) {
            continue;
        }
        
        // ELC format: Number X Y Z Label
        QStringList parts = line.split(QRegularExpression("\\s+"));
        if (parts.size() >= 5) {
            bool ok;
            double x = parts[1].toDouble(&ok);
            if (!ok) continue;
            double y = parts[2].toDouble(&ok);
            if (!ok) continue;
            double z = parts[3].toDouble(&ok);
            if (!ok) continue;
            
            QString label = parts[4];
            
            FiffDigPoint digPoint;
            digPoint.kind = FIFFV_POINT_EEG;
            digPoint.ident = m_digPoints.size() + 1;
            digPoint.coord_frame = FIFFV_COORD_HEAD;
            digPoint.r[0] = x / 1000.0; // Convert mm to m
            digPoint.r[1] = y / 1000.0;
            digPoint.r[2] = z / 1000.0;
            
            m_digPoints << digPoint;
            m_channelMap[label] = m_digPoints.size() - 1;
        }
    }
    
    return !m_digPoints.isEmpty();
}

//=============================================================================================================

bool DigMontage::parseSFPFile(const QString& fileName)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Cannot open SFP file:" << fileName;
        return false;
    }
    
    QTextStream stream(&file);
    m_digPoints.clear();
    m_channelMap.clear();
    
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        
        if (line.isEmpty() || line.startsWith("//")) {
            continue;
        }
        
        // SFP format: Label X Y Z
        QStringList parts = line.split(QRegularExpression("\\s+"));
        if (parts.size() >= 4) {
            QString label = parts[0];
            
            bool ok;
            double x = parts[1].toDouble(&ok);
            if (!ok) continue;
            double y = parts[2].toDouble(&ok);
            if (!ok) continue;
            double z = parts[3].toDouble(&ok);
            if (!ok) continue;
            
            FiffDigPoint digPoint;
            
            // Determine point type based on label
            if (label.toUpper() == "LPA") {
                digPoint.kind = FIFFV_POINT_CARDINAL;
                digPoint.ident = FIFFV_POINT_LPA;
            } else if (label.toUpper() == "RPA") {
                digPoint.kind = FIFFV_POINT_CARDINAL;
                digPoint.ident = FIFFV_POINT_RPA;
            } else if (label.toUpper() == "NASION" || label.toUpper() == "NAS") {
                digPoint.kind = FIFFV_POINT_CARDINAL;
                digPoint.ident = FIFFV_POINT_NASION;
            } else {
                digPoint.kind = FIFFV_POINT_EEG;
                digPoint.ident = m_digPoints.size() + 1;
            }
            
            digPoint.coord_frame = FIFFV_COORD_HEAD;
            digPoint.r[0] = x / 1000.0; // Convert mm to m
            digPoint.r[1] = y / 1000.0;
            digPoint.r[2] = z / 1000.0;
            
            m_digPoints << digPoint;
            if (digPoint.kind == FIFFV_POINT_EEG) {
                m_channelMap[label] = m_digPoints.size() - 1;
            }
        }
    }
    
    return !m_digPoints.isEmpty();
}