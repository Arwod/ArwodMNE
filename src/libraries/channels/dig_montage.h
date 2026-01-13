//=============================================================================================================
/**
 * @file     dig_montage.h
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
 * @brief    DigMontage class for electrode position management
 *
 */

#ifndef DIG_MONTAGE_H
#define DIG_MONTAGE_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "channels_global.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_dig_point_set.h>
#include <fiff/fiff_coord_trans.h>
#include <fiff/fiff_info.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QObject>
#include <QString>
#include <QStringList>
#include <QSharedPointer>
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
 * DigMontage class for managing electrode positions and coordinate transformations
 *
 * @brief DigMontage class for managing electrode positions and coordinate transformations
 */
class CHANNELSSHARED_EXPORT DigMontage : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<DigMontage> SPtr;            /**< Shared pointer type for DigMontage. */
    typedef QSharedPointer<const DigMontage> ConstSPtr; /**< Const shared pointer type for DigMontage. */

    //=========================================================================================================
    /**
     * Constructs a DigMontage
     */
    explicit DigMontage(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destroys the DigMontage
     */
    ~DigMontage();

    //=========================================================================================================
    /**
     * Create a DigMontage from channel positions
     *
     * @param[in] chPos         Channel positions (3xN matrix)
     * @param[in] chNames       Channel names
     * @param[in] coordFrame    Coordinate frame
     *
     * @return Shared pointer to DigMontage
     */
    static SPtr fromChannelPositions(const Eigen::MatrixXd& chPos,
                                     const QStringList& chNames,
                                     int coordFrame = FIFFV_COORD_HEAD);

    //=========================================================================================================
    /**
     * Create a DigMontage from standard electrode positions
     *
     * @param[in] montageType   Standard montage type (e.g., "standard_1020", "standard_1005")
     * @param[in] chNames       Channel names to include
     *
     * @return Shared pointer to DigMontage
     */
    static SPtr fromStandardMontage(const QString& montageType,
                                    const QStringList& chNames = QStringList());

    //=========================================================================================================
    /**
     * Load DigMontage from file
     *
     * @param[in] fileName      File name (.elc, .sfp, .csd, .elp, .hpts)
     *
     * @return Shared pointer to DigMontage
     */
    static SPtr fromFile(const QString& fileName);
    //=========================================================================================================
    /**
     * Get channel positions
     *
     * @return Channel positions as 3xN matrix
     */
    Eigen::MatrixXd getChannelPositions() const;

    //=========================================================================================================
    /**
     * Get channel names
     *
     * @return List of channel names
     */
    QStringList getChannelNames() const;

    //=========================================================================================================
    /**
     * Get fiducial points (LPA, RPA, Nasion)
     *
     * @return Map of fiducial names to positions
     */
    QMap<QString, Eigen::Vector3d> getFiducials() const;

    //=========================================================================================================
    /**
     * Set fiducial points
     *
     * @param[in] lpa       Left preauricular point
     * @param[in] rpa       Right preauricular point
     * @param[in] nasion    Nasion point
     */
    void setFiducials(const Eigen::Vector3d& lpa,
                      const Eigen::Vector3d& rpa,
                      const Eigen::Vector3d& nasion);

    //=========================================================================================================
    /**
     * Transform montage to different coordinate frame
     *
     * @param[in] coordTrans    Coordinate transformation
     */
    void transform(const FIFFLIB::FiffCoordTrans& coordTrans);

    //=========================================================================================================
    /**
     * Scale electrode positions
     *
     * @param[in] scaleFactor   Scaling factor
     */
    void scale(double scaleFactor);

    //=========================================================================================================
    /**
     * Add electrode position
     *
     * @param[in] chName        Channel name
     * @param[in] position      3D position
     * @param[in] kind          Point kind (FIFFV_POINT_EEG, etc.)
     */
    void addElectrode(const QString& chName,
                      const Eigen::Vector3d& position,
                      int kind = FIFFV_POINT_EEG);

    //=========================================================================================================
    /**
     * Remove electrode
     *
     * @param[in] chName        Channel name to remove
     */
    void removeElectrode(const QString& chName);

    //=========================================================================================================
    /**
     * Get digitizer point set
     *
     * @return FiffDigPointSet containing all points
     */
    FIFFLIB::FiffDigPointSet getDigPointSet() const;
    //=========================================================================================================
    /**
     * Apply montage to FiffInfo
     *
     * @param[in,out] info      FiffInfo to modify
     */
    void applyToInfo(FIFFLIB::FiffInfo& info) const;

    //=========================================================================================================
    /**
     * Compute head circumference from electrode positions
     *
     * @return Head circumference in meters
     */
    double computeHeadCircumference() const;

    //=========================================================================================================
    /**
     * Find nearest electrode to a given position
     *
     * @param[in] position      Target position
     *
     * @return Channel name of nearest electrode
     */
    QString findNearestElectrode(const Eigen::Vector3d& position) const;

    //=========================================================================================================
    /**
     * Validate montage consistency
     *
     * @return true if montage is valid, false otherwise
     */
    bool validate() const;

private:
    //=========================================================================================================
    /**
     * Load standard 10-20 electrode positions
     */
    void loadStandard1020();

    //=========================================================================================================
    /**
     * Load standard 10-05 electrode positions
     */
    void loadStandard1005();

    //=========================================================================================================
    /**
     * Parse ELC file format
     *
     * @param[in] fileName      ELC file name
     */
    bool parseELCFile(const QString& fileName);

    //=========================================================================================================
    /**
     * Parse SFP file format
     *
     * @param[in] fileName      SFP file name
     */
    bool parseSFPFile(const QString& fileName);

    FIFFLIB::FiffDigPointSet m_digPoints;       /**< Digitizer points */
    QMap<QString, int> m_channelMap;            /**< Channel name to dig point index mapping */
    int m_coordFrame;                           /**< Current coordinate frame */
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE CHANNELSLIB

#endif // DIG_MONTAGE_H