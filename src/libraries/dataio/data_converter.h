//=============================================================================================================
/**
 * @file     data_converter.h
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
 * @brief    Data converter and writer for various EEG/MEG file formats
 *
 */

#ifndef DATA_CONVERTER_H
#define DATA_CONVERTER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "dataio_global.h"
#include "multiformat_reader.h"

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
/**
 * Data converter and writer supporting various EEG/MEG file formats
 *
 * @brief Data converter and writer supporting various EEG/MEG file formats
 */
class DATAIOSHARED_EXPORT DataConverter : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<DataConverter> SPtr;            /**< Shared pointer type for DataConverter. */
    typedef QSharedPointer<const DataConverter> ConstSPtr; /**< Const shared pointer type for DataConverter. */

    //=========================================================================================================
    /**
     * Constructs a DataConverter
     */
    explicit DataConverter(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destroys the DataConverter
     */
    ~DataConverter();

    //=========================================================================================================
    /**
     * Convert data between formats
     *
     * @param[in] inputFile     Input file name
     * @param[in] outputFile    Output file name
     * @param[in] inputFormat   Input file format (AUTO for auto-detection)
     * @param[in] outputFormat  Output file format
     *
     * @return true if successful, false otherwise
     */
    bool convertFile(const QString& inputFile,
                     const QString& outputFile,
                     MultiFormatReader::FileFormat inputFormat = MultiFormatReader::AUTO,
                     MultiFormatReader::FileFormat outputFormat = MultiFormatReader::AUTO);

    //=========================================================================================================
    /**
     * Write raw data to EDF format
     *
     * @param[in] fileName    Output file name
     * @param[in] info        Measurement info
     * @param[in] rawData     Raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool writeEDF(const QString& fileName,
                  const FIFFLIB::FiffInfo& info,
                  const Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Write raw data to BrainVision format
     *
     * @param[in] fileName    Output header file name (.vhdr)
     * @param[in] info        Measurement info
     * @param[in] rawData     Raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool writeBrainVision(const QString& fileName,
                          const FIFFLIB::FiffInfo& info,
                          const Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Write raw data to CSV format
     *
     * @param[in] fileName    Output file name
     * @param[in] info        Measurement info
     * @param[in] rawData     Raw data matrix
     * @param[in] includeHeader Include channel names as header
     *
     * @return true if successful, false otherwise
     */
    bool writeCSV(const QString& fileName,
                  const FIFFLIB::FiffInfo& info,
                  const Eigen::MatrixXd& rawData,
                  bool includeHeader = true);

    //=========================================================================================================
    /**
     * Export data to Python-compatible format (NumPy)
     *
     * @param[in] fileName    Output file name (.npy)
     * @param[in] rawData     Raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool exportToNumPy(const QString& fileName,
                       const Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Export measurement info to JSON format
     *
     * @param[in] fileName    Output file name (.json)
     * @param[in] info        Measurement info
     *
     * @return true if successful, false otherwise
     */
    bool exportInfoToJSON(const QString& fileName,
                          const FIFFLIB::FiffInfo& info);

    //=========================================================================================================
    /**
     * Validate data compatibility between formats
     *
     * @param[in] info            Measurement info
     * @param[in] targetFormat    Target format
     *
     * @return true if compatible, false otherwise
     */
    static bool validateCompatibility(const FIFFLIB::FiffInfo& info,
                                     MultiFormatReader::FileFormat targetFormat);

    //=========================================================================================================
    /**
     * Get format-specific limitations
     *
     * @param[in] format    File format
     *
     * @return String describing format limitations
     */
    static QString getFormatLimitations(MultiFormatReader::FileFormat format);

private:
    //=========================================================================================================
    /**
     * Write EDF header
     *
     * @param[in] device      Output device
     * @param[in] info        Measurement info
     * @param[in] nSamples    Number of samples
     *
     * @return true if successful, false otherwise
     */
    bool writeEDFHeader(QIODevice* device,
                        const FIFFLIB::FiffInfo& info,
                        int nSamples);

    //=========================================================================================================
    /**
     * Write BrainVision header file
     *
     * @param[in] headerFile  Header file name
     * @param[in] dataFile    Data file name
     * @param[in] info        Measurement info
     *
     * @return true if successful, false otherwise
     */
    bool writeBrainVisionHeader(const QString& headerFile,
                                const QString& dataFile,
                                const FIFFLIB::FiffInfo& info);

    //=========================================================================================================
    /**
     * Write BrainVision data file
     *
     * @param[in] dataFile    Data file name
     * @param[in] rawData     Raw data matrix
     *
     * @return true if successful, false otherwise
     */
    bool writeBrainVisionData(const QString& dataFile,
                              const Eigen::MatrixXd& rawData);

    //=========================================================================================================
    /**
     * Convert channel type to EDF format
     *
     * @param[in] fiffType    FIFF channel type
     *
     * @return EDF channel type string
     */
    static QString convertChannelTypeToEDF(int fiffType);

    //=========================================================================================================
    /**
     * Convert physical unit to string
     *
     * @param[in] fiffUnit    FIFF unit constant
     *
     * @return Unit string
     */
    static QString convertUnitToString(int fiffUnit);

    //=========================================================================================================
    /**
     * Sanitize string for file format compatibility
     *
     * @param[in] input       Input string
     * @param[in] maxLength   Maximum allowed length
     *
     * @return Sanitized string
     */
    static QString sanitizeString(const QString& input, int maxLength);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE DATAIOLIB

#endif // DATA_CONVERTER_H