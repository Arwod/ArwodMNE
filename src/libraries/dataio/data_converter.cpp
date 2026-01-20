//=============================================================================================================
/**
 * @file     data_converter.cpp
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
 * @brief    Implementation of the DataConverter class.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "data_converter.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_constants.h>
#include <fiff/fiff_types.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QDataStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QRegularExpression>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace DATAIOLIB;
using namespace FIFFLIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

DataConverter::DataConverter(QObject *parent)
: QObject(parent)
{
}

//=============================================================================================================

DataConverter::~DataConverter()
{
}

//=============================================================================================================

bool DataConverter::convertFile(const QString& inputFile,
                                const QString& outputFile,
                                MultiFormatReader::FileFormat inputFormat,
                                MultiFormatReader::FileFormat outputFormat)
{
    // Read input file
    MultiFormatReader reader;
    FiffInfo info;
    Eigen::MatrixXd rawData;
    
    MultiFormatReader::FileFormat actualInputFormat = inputFormat;
    if (inputFormat == MultiFormatReader::AUTO) {
        actualInputFormat = MultiFormatReader::detectFormat(inputFile);
    }
    
    if (!reader.readRaw(inputFile, actualInputFormat, info, rawData)) {
        qWarning() << "Failed to read input file:" << inputFile;
        return false;
    }
    
    // Determine output format
    MultiFormatReader::FileFormat actualOutputFormat = outputFormat;
    if (outputFormat == MultiFormatReader::AUTO) {
        actualOutputFormat = MultiFormatReader::detectFormat(outputFile);
    }
    
    // Validate compatibility
    if (!validateCompatibility(info, actualOutputFormat)) {
        qWarning() << "Data not compatible with target format:" << getFormatLimitations(actualOutputFormat);
        return false;
    }
    
    // Write output file
    switch (actualOutputFormat) {
        case MultiFormatReader::EDF:
        case MultiFormatReader::EDFPLUS:
            return writeEDF(outputFile, info, rawData);
        case MultiFormatReader::BRAINVISION:
            return writeBrainVision(outputFile, info, rawData);
        default:
            qWarning() << "Unsupported output format";
            return false;
    }
}

//=============================================================================================================

bool DataConverter::writeEDF(const QString& fileName,
                             const FiffInfo& info,
                             const Eigen::MatrixXd& rawData)
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot create EDF file:" << fileName;
        return false;
    }
    
    if (!writeEDFHeader(&file, info, rawData.cols())) {
        qWarning() << "Failed to write EDF header";
        return false;
    }
    
    // Write data
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    int nChannels = info.nchan;
    int nSamples = rawData.cols();
    
    // Write data in EDF format (16-bit integers)
    for (int sample = 0; sample < nSamples; ++sample) {
        for (int ch = 0; ch < nChannels; ++ch) {
            // Convert to digital value
            double physicalValue = rawData(ch, sample);
            double digitalValue = (physicalValue - info.chs[ch].range) / info.chs[ch].cal;
            
            // Clamp to 16-bit range
            qint16 digitalSample = qBound(-32768.0, digitalValue, 32767.0);
            stream << digitalSample;
        }
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::writeBrainVision(const QString& fileName,
                                     const FiffInfo& info,
                                     const Eigen::MatrixXd& rawData)
{
    QFileInfo fileInfo(fileName);
    QString baseName = fileInfo.completeBaseName();
    QString path = fileInfo.absolutePath();
    
    QString headerFile = fileName;
    QString dataFile = QDir(path).absoluteFilePath(baseName + ".eeg");
    QString markerFile = QDir(path).absoluteFilePath(baseName + ".vmrk");
    
    // Write header file
    if (!writeBrainVisionHeader(headerFile, dataFile, info)) {
        return false;
    }
    
    // Write data file
    if (!writeBrainVisionData(dataFile, rawData)) {
        return false;
    }
    
    // Write marker file (empty for now)
    QFile markers(markerFile);
    if (markers.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream stream(&markers);
        stream << "Brain Vision Data Exchange Marker File, Version 1.0\n";
        stream << "; Data created by MNE-CPP DataConverter\n\n";
        stream << "[Common Infos]\n";
        stream << "Codepage=UTF-8\n";
        stream << "DataFile=" << QFileInfo(dataFile).fileName() << "\n\n";
        stream << "[Marker Infos]\n";
        stream << "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n";
        stream << "; <Size in data points>, <Channel number (0 = marker is related to all channels)>\n";
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::writeCSV(const QString& fileName,
                             const FiffInfo& info,
                             const Eigen::MatrixXd& rawData,
                             bool includeHeader)
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Cannot create CSV file:" << fileName;
        return false;
    }
    
    QTextStream stream(&file);
    
    // Write header if requested
    if (includeHeader) {
        QStringList headers;
        for (const QString& chName : info.ch_names) {
            headers << chName;
        }
        stream << headers.join(",") << "\n";
    }
    
    // Write data
    int nSamples = rawData.cols();
    int nChannels = rawData.rows();
    
    for (int sample = 0; sample < nSamples; ++sample) {
        QStringList values;
        for (int ch = 0; ch < nChannels; ++ch) {
            values << QString::number(rawData(ch, sample), 'e', 6);
        }
        stream << values.join(",") << "\n";
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::exportToNumPy(const QString& fileName,
                                  const Eigen::MatrixXd& rawData)
{
    // Simplified NumPy export (binary format)
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot create NumPy file:" << fileName;
        return false;
    }
    
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Write NumPy header (simplified)
    QByteArray header = "\x93NUMPY\x01\x00";
    file.write(header);
    
    // Write shape information
    int nChannels = rawData.rows();
    int nSamples = rawData.cols();
    
    stream << static_cast<qint32>(nChannels);
    stream << static_cast<qint32>(nSamples);
    
    // Write data
    for (int ch = 0; ch < nChannels; ++ch) {
        for (int sample = 0; sample < nSamples; ++sample) {
            double value = rawData(ch, sample);
            stream << value;
        }
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::exportInfoToJSON(const QString& fileName,
                                     const FiffInfo& info)
{
    QJsonObject jsonInfo;
    
    // Basic info
    jsonInfo["nchan"] = info.nchan;
    jsonInfo["sfreq"] = info.sfreq;
    
    // Channel information
    QJsonArray channels;
    for (int i = 0; i < info.chs.size(); ++i) {
        QJsonObject channel;
        channel["ch_name"] = info.chs[i].ch_name;
        channel["kind"] = info.chs[i].kind;
        channel["unit"] = info.chs[i].unit;
        channel["cal"] = info.chs[i].cal;
        channel["range"] = info.chs[i].range;
        channels.append(channel);
    }
    jsonInfo["chs"] = channels;
    
    // Channel names
    QJsonArray chNames;
    for (const QString& name : info.ch_names) {
        chNames.append(name);
    }
    jsonInfo["ch_names"] = chNames;
    
    // Write to file
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Cannot create JSON file:" << fileName;
        return false;
    }
    
    QJsonDocument doc(jsonInfo);
    file.write(doc.toJson());
    
    return true;
}

//=============================================================================================================

bool DataConverter::validateCompatibility(const FiffInfo& info,
                                          MultiFormatReader::FileFormat targetFormat)
{
    switch (targetFormat) {
        case MultiFormatReader::EDF:
        case MultiFormatReader::EDFPLUS:
            // EDF has channel and sampling rate limitations
            if (info.nchan > 256) {
                return false;
            }
            if (info.sfreq > 1000.0) {
                return false; // Typical EDF limitation
            }
            break;
        case MultiFormatReader::BRAINVISION:
            // BrainVision is more flexible
            break;
        default:
            return true;
    }
    
    return true;
}

//=============================================================================================================

QString DataConverter::getFormatLimitations(MultiFormatReader::FileFormat format)
{
    switch (format) {
        case MultiFormatReader::EDF:
            return "EDF format: Max 256 channels, sampling rate <= 1000 Hz, 16-bit resolution";
        case MultiFormatReader::EDFPLUS:
            return "EDF+ format: Max 256 channels, supports annotations";
        case MultiFormatReader::BDF:
            return "BDF format: 24-bit resolution, BioSemi specific";
        case MultiFormatReader::BRAINVISION:
            return "BrainVision format: Flexible, supports multiple data types";
        case MultiFormatReader::CNT:
            return "CNT format: Neuroscan specific, limited metadata";
        default:
            return "No specific limitations";
    }
}

//=============================================================================================================

bool DataConverter::writeEDFHeader(QIODevice* device,
                                   const FiffInfo& info,
                                   int nSamples)
{
    QDataStream stream(device);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Fixed header (256 bytes)
    QByteArray header(256, ' ');
    
    // Version
    header.replace(0, 8, "0       ");
    
    // Patient identification (80 bytes)
    QString patientId = "X X X X";
    header.replace(8, qMin(80, patientId.length()), patientId.toLatin1());
    
    // Recording identification (80 bytes)
    QString recordId = QString("Startdate %1 MNE-CPP").arg(QDateTime::currentDateTime().toString("dd-MMM-yyyy"));
    header.replace(88, qMin(80, recordId.length()), recordId.toLatin1());
    
    // Start date and time
    QString startDate = QDateTime::currentDateTime().toString("dd.MM.yy");
    QString startTime = QDateTime::currentDateTime().toString("hh.mm.ss");
    header.replace(168, 8, startDate.toLatin1());
    header.replace(176, 8, startTime.toLatin1());
    
    // Number of bytes in header
    int headerBytes = 256 + info.nchan * 256;
    QString headerBytesStr = QString("%1").arg(headerBytes, 8);
    header.replace(184, 8, headerBytesStr.toLatin1());
    
    // Reserved (44 bytes)
    header.replace(192, 44, QByteArray(44, ' '));
    
    // Number of data records
    int recordDuration = 1; // 1 second records
    int nRecords = nSamples / static_cast<int>(info.sfreq);
    QString nRecordsStr = QString("%1").arg(nRecords, 8);
    header.replace(236, 8, nRecordsStr.toLatin1());
    
    // Duration of data record
    QString durationStr = QString("%1").arg(recordDuration, 8);
    header.replace(244, 8, durationStr.toLatin1());
    
    // Number of signals
    QString nSignalsStr = QString("%1").arg(info.nchan, 4);
    header.replace(252, 4, nSignalsStr.toLatin1());
    
    device->write(header);
    
    // Channel information (256 bytes per channel)
    for (int ch = 0; ch < info.nchan; ++ch) {
        QByteArray chInfo(256, ' ');
        
        // Channel label (16 bytes)
        QString chName = sanitizeString(info.ch_names[ch], 16);
        chInfo.replace(0, chName.length(), chName.toLatin1());
        
        // Transducer type (80 bytes)
        QString transducer = convertChannelTypeToEDF(info.chs[ch].kind);
        chInfo.replace(16, transducer.length(), transducer.toLatin1());
        
        // Physical dimension (8 bytes)
        QString unit = convertUnitToString(info.chs[ch].unit);
        chInfo.replace(96, unit.length(), unit.toLatin1());
        
        // Physical minimum and maximum
        double physMin = -32768.0 * info.chs[ch].cal + info.chs[ch].range;
        double physMax = 32767.0 * info.chs[ch].cal + info.chs[ch].range;
        
        QString physMinStr = QString("%1").arg(physMin, 8, 'g', 6);
        QString physMaxStr = QString("%1").arg(physMax, 8, 'g', 6);
        chInfo.replace(104, physMinStr.length(), physMinStr.toLatin1());
        chInfo.replace(112, physMaxStr.length(), physMaxStr.toLatin1());
        
        // Digital minimum and maximum
        chInfo.replace(120, 8, "-32768  ");
        chInfo.replace(128, 8, "32767   ");
        
        // Prefiltering (80 bytes)
        chInfo.replace(136, 80, QByteArray(80, ' '));
        
        // Number of samples per record
        int samplesPerRecord = static_cast<int>(info.sfreq * recordDuration);
        QString samplesStr = QString("%1").arg(samplesPerRecord, 8);
        chInfo.replace(216, samplesStr.length(), samplesStr.toLatin1());
        
        // Reserved (32 bytes)
        chInfo.replace(224, 32, QByteArray(32, ' '));
        
        device->write(chInfo);
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::writeBrainVisionHeader(const QString& headerFile,
                                           const QString& dataFile,
                                           const FiffInfo& info)
{
    QFile file(headerFile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Cannot create BrainVision header file:" << headerFile;
        return false;
    }
    
    QTextStream stream(&file);
    
    // Write header
    stream << "Brain Vision Data Exchange Header File Version 1.0\n";
    stream << "; Data created by MNE-CPP DataConverter\n\n";
    
    // Common Infos
    stream << "[Common Infos]\n";
    stream << "Codepage=UTF-8\n";
    stream << "DataFile=" << QFileInfo(dataFile).fileName() << "\n";
    stream << "MarkerFile=" << QFileInfo(headerFile).completeBaseName() << ".vmrk\n";
    stream << "DataFormat=BINARY\n";
    stream << "DataOrientation=MULTIPLEXED\n";
    stream << "NumberOfChannels=" << info.nchan << "\n";
    stream << "SamplingInterval=" << static_cast<int>(1000000.0 / info.sfreq) << "\n\n";
    
    // Binary Infos
    stream << "[Binary Infos]\n";
    stream << "BinaryFormat=IEEE_FLOAT_32\n\n";
    
    // Channel Infos
    stream << "[Channel Infos]\n";
    stream << "; Each entry: Ch<Channel number>=<Name>,<Reference>,<Resolution in \"Unit\">,<Unit>, Future extensions..\n";
    stream << "; Fields are delimited by commas, some fields might be omitted (empty).\n";
    stream << "; Commas in channel names are coded as \"\\1\".\n";
    
    for (int ch = 0; ch < info.nchan; ++ch) {
        QString chName = info.ch_names[ch];
        chName.replace(",", "\\1"); // Escape commas
        
        QString unit = convertUnitToString(info.chs[ch].unit);
        stream << "Ch" << (ch + 1) << "=" << chName << ",,1," << unit << "\n";
    }
    
    return true;
}

//=============================================================================================================

bool DataConverter::writeBrainVisionData(const QString& dataFile,
                                         const Eigen::MatrixXd& rawData)
{
    QFile file(dataFile);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot create BrainVision data file:" << dataFile;
        return false;
    }
    
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    
    int nChannels = rawData.rows();
    int nSamples = rawData.cols();
    
    // Write data in multiplexed format
    for (int sample = 0; sample < nSamples; ++sample) {
        for (int ch = 0; ch < nChannels; ++ch) {
            float value = static_cast<float>(rawData(ch, sample));
            stream << value;
        }
    }
    
    return true;
}

//=============================================================================================================

QString DataConverter::convertChannelTypeToEDF(int fiffType)
{
    switch (fiffType) {
        case FIFFV_EEG_CH:
            return "EEG";
        case FIFFV_MEG_CH:
            return "MEG";
        case FIFFV_EOG_CH:
            return "EOG";
        case FIFFV_ECG_CH:
            return "ECG";
        case FIFFV_EMG_CH:
            return "EMG";
        case FIFFV_STIM_CH:
            return "TRIG";
        default:
            return "MISC";
    }
}

//=============================================================================================================

QString DataConverter::convertUnitToString(int fiffUnit)
{
    switch (fiffUnit) {
        case FIFF_UNIT_V:
            return "µV";
        case FIFF_UNIT_T:
            return "fT";
        case FIFF_UNIT_AM:
            return "fT/cm";
        default:
            return "µV";
    }
}

//=============================================================================================================

QString DataConverter::sanitizeString(const QString& input, int maxLength)
{
    QString sanitized = input;
    
    // Remove or replace problematic characters
    sanitized.replace(QRegularExpression("[^A-Za-z0-9_\\-\\s]"), "_");
    
    // Truncate if necessary
    if (sanitized.length() > maxLength) {
        sanitized = sanitized.left(maxLength);
    }
    
    // Pad with spaces if needed
    while (sanitized.length() < maxLength) {
        sanitized += " ";
    }
    
    return sanitized;
}