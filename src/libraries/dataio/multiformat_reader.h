//=============================================================================================================
/**
 * @file     multiformat_reader.h
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
 * @brief    Multi-format data reader for various EEG/MEG file formats
 *
 */

#ifndef MULTIFORMAT_READER_H
#define MULTIFORMAT_READER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "dataio_global.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_info.h>
#include <fiff/fiff_raw_data.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QObject>
#include <QString>
#include <QSharedPointer>
#include <QIODevice>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Core>

//=============================================================================================================
// DEFINE NAMESPACE DATAIOLIB
//=============================================================================================================

namespace DATAIOLIB
{

//=============================================================================================================
// DATAIOLIB FORWARD DECLARATIONS
//=============================================================================================================

//=============================================================================================================
/**
 * Multi-format data reader supporting various EEG/MEG file formats
 *
 * @brief Multi-format data reader supporting various EEG/MEG file formats
 */
class DATAIOSHARED_EXPORT MultiFormatReader : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<MultiFormatReader> SPtr;            /**< Shared pointer type for MultiFormatReader. */
    typedef QSharedPointer<const MultiFormatReader> ConstSPtr; /**< Const shared pointer type for MultiFormatReader. */

    //=========================================================================================================
    /**
     * Supported file formats
     */
    enum FileFormat {
        FIFF,           /**< FIFF format (.fif) */
        EDF,            /**< European Data Format (.edf) */
        EDFPLUS,        /**< EDF+ format (.edf) */
        BDF,            /**< BioSemi Data Format (.bdf) */
        BRAINVISION,    /**< BrainVision format (.vhdr, .vmrk, .eeg) */
        CNT,            /**< Neuroscan CNT format (.cnt) */
        SET,            /**< EEGLAB SET format (.set) */
        AUTO            /**< Auto-detect format */
    };

    //=========================================================================================================
    /**
     * Constructs a MultiFormatReader
     */
    explicit MultiFormatReader(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destroys the MultiFormatReader
     */
    ~MultiFormatReader();

    //=========================================================================================================
    /**
     * Detect file format from filename extension
     *
     * @param[in] fileName    The file name to analyze
     *
     * @return The detected file format
     */
    static FileFormat detectFormat(const QString& fileName);

    //=========================================================================================================
    /**
     * Read raw data from file with automatic format detection
     *
     * @param[in] fileName    The file name to read
     * @param[out] info       The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readRaw(const QString& fileName,
                 FIFFLIB::FiffInfo& info,
                 Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Read raw data from file with specified format
     *
     * @param[in] fileName    The file name to read
     * @param[in] format      The file format
     * @param[out] info       The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readRaw(const QString& fileName,
                 FileFormat format,
                 FIFFLIB::FiffInfo& info,
                 Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Read EDF/EDF+ format files
     *
     * @param[in] fileName    The EDF file name
     * @param[out] info       The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readEDF(const QString& fileName,
                 FIFFLIB::FiffInfo& info,
                 Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Read BrainVision format files
     *
     * @param[in] fileName    The BrainVision header file name (.vhdr)
     * @param[out] info       The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readBrainVision(const QString& fileName,
                         FIFFLIB::FiffInfo& info,
                         Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Read CNT format files
     *
     * @param[in] fileName    The CNT file name
     * @param[out] info       The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readCNT(const QString& fileName,
                 FIFFLIB::FiffInfo& info,
                 Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Convert raw data to FiffRawData object
     *
     * @param[in] info        The measurement info
     * @param[in] rawData     The raw data matrix
     *
     * @return Shared pointer to FiffRawData object
     */
    static QSharedPointer<FIFFLIB::FiffRawData> toFiffRawData(const FIFFLIB::FiffInfo& info,
                                                              const Eigen::MatrixXd& rawData);

private:
    //=========================================================================================================
    /**
     * Parse EDF header
     *
     * @param[in] device      The input device
     * @param[out] info       The measurement info
     * @param[out] dataOffset The data offset in bytes
     * @param[out] recordSize The record size in samples
     *
     * @return true if successful, false otherwise
     */
    bool parseEDFHeader(QIODevice* device,
                        FIFFLIB::FiffInfo& info,
                        qint64& dataOffset,
                        int& recordSize);

    //=========================================================================================================
    /**
     * Parse BrainVision header file
     *
     * @param[in] headerFile  The header file name
     * @param[out] info       The measurement info
     * @param[out] dataFile   The data file name
     * @param[out] markerFile The marker file name
     *
     * @return true if successful, false otherwise
     */
    bool parseBrainVisionHeader(const QString& headerFile,
                                FIFFLIB::FiffInfo& info,
                                QString& dataFile,
                                QString& markerFile);

    //=========================================================================================================
    /**
     * Read BrainVision data file
     *
     * @param[in] dataFile    The data file name
     * @param[in] info        The measurement info
     * @param[out] rawData    The raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool readBrainVisionData(const QString& dataFile,
                             const FIFFLIB::FiffInfo& info,
                             Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Parse CNT header
     *
     * @param[in] device      The input device
     * @param[out] info       The measurement info
     * @param[out] dataOffset The data offset in bytes
     *
     * @return true if successful, false otherwise
     */
    bool parseCNTHeader(QIODevice* device,
                        FIFFLIB::FiffInfo& info,
                        qint64& dataOffset);

    //=========================================================================================================
    /**
     * Create channel info from parameters
     *
     * @param[in] chName      Channel name
     * @param[in] chType      Channel type
     * @param[in] unit        Physical unit
     * @param[in] cal         Calibration factor
     *
     * @return FiffChInfo object
     */
    static FIFFLIB::FiffChInfo createChannelInfo(const QString& chName,
                                                 int chType,
                                                 int unit,
                                                 double cal);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE DATAIOLIB

#endif // MULTIFORMAT_READER_H