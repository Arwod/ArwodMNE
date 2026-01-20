//=============================================================================================================
/**
 * @file     multiformat_reader.cpp
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
 * @brief    Implementation of the MultiFormatReader class.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "multiformat_reader.h"

//=============================================================================================================
// FIFF INCLUDES
//=============================================================================================================

#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_constants.h>
#include <fiff/fiff_types.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QStringList>
#include <QDebug>
#include <QDataStream>
#include <QDir>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace DATAIOLIB;
using namespace FIFFLIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

MultiFormatReader::MultiFormatReader(QObject *parent)
: QObject(parent)
{
}

//=============================================================================================================

MultiFormatReader::~MultiFormatReader()
{
}

//=============================================================================================================

MultiFormatReader::FileFormat MultiFormatReader::detectFormat(const QString& fileName)
{
    QFileInfo fileInfo(fileName);
    QString suffix = fileInfo.suffix().toLower();
    
    if (suffix == "fif") {
        return FIFF;
    } else if (suffix == "edf") {
        return EDF;
    } else if (suffix == "bdf") {
        return BDF;
    } else if (suffix == "vhdr") {
        return BRAINVISION;
    } else if (suffix == "cnt") {
        return CNT;
    } else if (suffix == "set") {
        return SET;
    }
    
    return AUTO;
}

//=============================================================================================================

bool MultiFormatReader::readRaw(const QString& fileName,
                                FiffInfo& info,
                                Eigen::MatrixXd& rawData)
{
    FileFormat format = detectFormat(fileName);
    return readRaw(fileName, format, info, rawData);
}

//=============================================================================================================

bool MultiFormatReader::readRaw(const QString& fileName,
                                FileFormat format,
                                FiffInfo& info,
                                Eigen::MatrixXd& rawData)
{
    switch (format) {
        case EDF:
        case EDFPLUS:
        case BDF:
            return readEDF(fileName, info, rawData);
        case BRAINVISION:
            return readBrainVision(fileName, info, rawData);
        case CNT:
            return readCNT(fileName, info, rawData);
        case FIFF:
            qWarning() << "FIFF format should be handled by existing FiffIO class";
            return false;
        default:
            qWarning() << "Unsupported file format";
            return false;
    }
}

//=============================================================================================================

bool MultiFormatReader::readEDF(const QString& fileName,
                                FiffInfo& info,
                                Eigen::MatrixXd& rawData)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Cannot open EDF file:" << fileName;
        return false;
    }
    
    qint64 dataOffset;
    int recordSize;
    
    if (!parseEDFHeader(&file, info, dataOffset, recordSize)) {
        qWarning() << "Failed to parse EDF header";
        return false;
    }
    
    // Read data records
    file.seek(dataOffset);
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    int nChannels = info.nchan;
    int nSamples = recordSize * info.sfreq; // Approximate total samples
    
    rawData.resize(nChannels, nSamples);
    
    // Read data in records
    int sampleIdx = 0;
    while (!stream.atEnd() && sampleIdx < nSamples) {
        for (int ch = 0; ch < nChannels; ++ch) {
            qint16 sample;
            stream >> sample;
            
            if (sampleIdx < nSamples) {
                // Apply calibration
                double physicalValue = info.chs[ch].cal * sample + info.chs[ch].range;
                rawData(ch, sampleIdx) = physicalValue;
            }
        }
        sampleIdx++;
    }
    
    // Resize to actual data size
    if (sampleIdx < nSamples) {
        rawData.conservativeResize(nChannels, sampleIdx);
    }
    
    return true;
}

//=============================================================================================================

bool MultiFormatReader::readBrainVision(const QString& fileName,
                                       FiffInfo& info,
                                       Eigen::MatrixXd& rawData)
{
    QString dataFile, markerFile;
    
    if (!parseBrainVisionHeader(fileName, info, dataFile, markerFile)) {
        qWarning() << "Failed to parse BrainVision header";
        return false;
    }
    
    return readBrainVisionData(dataFile, info, rawData);
}

//=============================================================================================================

bool MultiFormatReader::readCNT(const QString& fileName,
                                FiffInfo& info,
                                Eigen::MatrixXd& rawData)
{
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Cannot open CNT file:" << fileName;
        return false;
    }
    
    qint64 dataOffset;
    
    if (!parseCNTHeader(&file, info, dataOffset)) {
        qWarning() << "Failed to parse CNT header";
        return false;
    }
    
    // Read continuous data
    file.seek(dataOffset);
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    int nChannels = info.nchan;
    qint64 remainingBytes = file.size() - dataOffset;
    int nSamples = remainingBytes / (nChannels * sizeof(float));
    
    rawData.resize(nChannels, nSamples);
    
    for (int sample = 0; sample < nSamples; ++sample) {
        for (int ch = 0; ch < nChannels; ++ch) {
            float value;
            stream >> value;
            rawData(ch, sample) = value * info.chs[ch].cal;
        }
    }
    
    return true;
}

//=============================================================================================================

QSharedPointer<FiffRawData> MultiFormatReader::toFiffRawData(const FiffInfo& info,
                                                            const Eigen::MatrixXd& rawData)
{
    QSharedPointer<FiffRawData> fiffRaw(new FiffRawData());
    
    // Set basic info
    fiffRaw->info = info;
    fiffRaw->first_samp = 0;
    fiffRaw->last_samp = rawData.cols() - 1;
    
    // Note: FiffRawData doesn't store data directly, it reads from file
    // This is a simplified implementation for demonstration
    
    return fiffRaw;
}

//=============================================================================================================

bool MultiFormatReader::parseEDFHeader(QIODevice* device,
                                       FiffInfo& info,
                                       qint64& dataOffset,
                                       int& recordSize)
{
    QDataStream stream(device);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Read fixed header (256 bytes)
    char version[8], patientId[80], recordId[80], startDate[8], startTime[8];
    char reserved[44];
    qint32 nDataRecords, recordDuration;
    qint16 nChannels;
    
    stream.readRawData(version, 8);
    stream.readRawData(patientId, 80);
    stream.readRawData(recordId, 80);
    stream.readRawData(startDate, 8);
    stream.readRawData(startTime, 8);
    stream >> nChannels;
    stream.readRawData(reserved, 44);
    stream >> nDataRecords >> recordDuration;
    
    info.nchan = nChannels;
    info.sfreq = 1.0 / recordDuration; // Simplified
    
    // Read channel information
    info.chs.clear();
    info.ch_names.clear();
    
    QStringList channelLabels;
    QList<QString> transducerTypes, physicalDimensions;
    QList<double> physicalMins, physicalMaxs, digitalMins, digitalMaxs;
    QList<QString> prefiltering;
    QList<int> samplesPerRecord;
    
    // Read channel labels
    for (int i = 0; i < nChannels; ++i) {
        char label[16];
        stream.readRawData(label, 16);
        channelLabels.append(QString::fromLatin1(label, 16).trimmed());
    }
    
    // Read transducer types
    for (int i = 0; i < nChannels; ++i) {
        char transducer[80];
        stream.readRawData(transducer, 80);
        transducerTypes.append(QString::fromLatin1(transducer, 80).trimmed());
    }
    
    // Read physical dimensions
    for (int i = 0; i < nChannels; ++i) {
        char dimension[8];
        stream.readRawData(dimension, 8);
        physicalDimensions.append(QString::fromLatin1(dimension, 8).trimmed());
    }
    
    // Read physical minimums
    for (int i = 0; i < nChannels; ++i) {
        char physMin[8];
        stream.readRawData(physMin, 8);
        physicalMins.append(QString::fromLatin1(physMin, 8).trimmed().toDouble());
    }
    
    // Read physical maximums
    for (int i = 0; i < nChannels; ++i) {
        char physMax[8];
        stream.readRawData(physMax, 8);
        physicalMaxs.append(QString::fromLatin1(physMax, 8).trimmed().toDouble());
    }
    
    // Read digital minimums
    for (int i = 0; i < nChannels; ++i) {
        char digMin[8];
        stream.readRawData(digMin, 8);
        digitalMins.append(QString::fromLatin1(digMin, 8).trimmed().toDouble());
    }
    
    // Read digital maximums
    for (int i = 0; i < nChannels; ++i) {
        char digMax[8];
        stream.readRawData(digMax, 8);
        digitalMaxs.append(QString::fromLatin1(digMax, 8).trimmed().toDouble());
    }
    
    // Create channel info
    for (int i = 0; i < nChannels; ++i) {
        FiffChInfo chInfo = createChannelInfo(
            channelLabels[i],
            FIFFV_EEG_CH, // Default to EEG
            FIFF_UNIT_V,  // Default to volts
            (physicalMaxs[i] - physicalMins[i]) / (digitalMaxs[i] - digitalMins[i])
        );
        
        chInfo.range = physicalMins[i];
        info.chs.append(chInfo);
        info.ch_names.append(channelLabels[i]);
    }
    
    dataOffset = device->pos();
    recordSize = nDataRecords;
    
    return true;
}

//=============================================================================================================

bool MultiFormatReader::parseBrainVisionHeader(const QString& headerFile,
                                              FiffInfo& info,
                                              QString& dataFile,
                                              QString& markerFile)
{
    QFile file(headerFile);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Cannot open BrainVision header file:" << headerFile;
        return false;
    }
    
    QTextStream stream(&file);
    QString line;
    QString section;
    
    QFileInfo headerInfo(headerFile);
    QString basePath = headerInfo.absolutePath();
    
    while (!stream.atEnd()) {
        line = stream.readLine().trimmed();
        
        if (line.startsWith("[") && line.endsWith("]")) {
            section = line.mid(1, line.length() - 2);
            continue;
        }
        
        if (line.isEmpty() || line.startsWith(";")) {
            continue;
        }
        
        QStringList parts = line.split("=");
        if (parts.size() != 2) {
            continue;
        }
        
        QString key = parts[0].trimmed();
        QString value = parts[1].trimmed();
        
        if (section == "Common Infos") {
            if (key == "DataFile") {
                dataFile = QDir(basePath).absoluteFilePath(value);
            } else if (key == "MarkerFile") {
                markerFile = QDir(basePath).absoluteFilePath(value);
            } else if (key == "NumberOfChannels") {
                info.nchan = value.toInt();
            } else if (key == "SamplingInterval") {
                info.sfreq = 1000000.0 / value.toDouble(); // Convert from microseconds
            }
        } else if (section == "Channel Infos") {
            // Parse channel information
            QStringList channelParts = value.split(",");
            if (channelParts.size() >= 3) {
                QString chName = channelParts[0].trimmed();
                QString chType = channelParts[2].trimmed();
                
                FiffChInfo chInfo = createChannelInfo(
                    chName,
                    FIFFV_EEG_CH, // Default to EEG
                    FIFF_UNIT_V,  // Default to volts
                    1.0           // Default calibration
                );
                
                info.chs.append(chInfo);
                info.ch_names.append(chName);
            }
        }
    }
    
    return !dataFile.isEmpty();
}

//=============================================================================================================

bool MultiFormatReader::readBrainVisionData(const QString& dataFile,
                                           const FiffInfo& info,
                                           Eigen::MatrixXd& rawData)
{
    QFile file(dataFile);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Cannot open BrainVision data file:" << dataFile;
        return false;
    }
    
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    int nChannels = info.nchan;
    qint64 remainingBytes = file.size();
    int nSamples = remainingBytes / (nChannels * sizeof(float));
    
    rawData.resize(nChannels, nSamples);
    
    for (int sample = 0; sample < nSamples; ++sample) {
        for (int ch = 0; ch < nChannels; ++ch) {
            float value;
            stream >> value;
            rawData(ch, sample) = value;
        }
    }
    
    return true;
}

//=============================================================================================================

bool MultiFormatReader::parseCNTHeader(QIODevice* device,
                                       FiffInfo& info,
                                       qint64& dataOffset)
{
    QDataStream stream(device);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Read CNT header (simplified)
    char signature[12];
    qint32 nextFile, prevFile, type, id, version, size;
    
    stream.readRawData(signature, 12);
    stream >> nextFile >> prevFile >> type >> id >> version >> size;
    
    // Skip to channel information
    device->seek(370); // CNT channel info starts at offset 370
    
    qint16 nChannels;
    stream >> nChannels;
    
    info.nchan = nChannels;
    info.sfreq = 250.0; // Default sampling rate for CNT
    
    // Read channel information (simplified)
    info.chs.clear();
    info.ch_names.clear();
    
    for (int i = 0; i < nChannels; ++i) {
        char chName[10];
        stream.readRawData(chName, 10);
        
        QString channelName = QString::fromLatin1(chName, 10).trimmed();
        
        FiffChInfo chInfo = createChannelInfo(
            channelName,
            FIFFV_EEG_CH,
            FIFF_UNIT_V,
            1.0
        );
        
        info.chs.append(chInfo);
        info.ch_names.append(channelName);
        
        // Skip other channel parameters
        device->seek(device->pos() + 65); // Skip to next channel
    }
    
    dataOffset = 900; // Typical CNT data offset
    
    return true;
}

//=============================================================================================================

FiffChInfo MultiFormatReader::createChannelInfo(const QString& chName,
                                               int chType,
                                               int unit,
                                               double cal)
{
    FiffChInfo chInfo;
    
    chInfo.ch_name = chName;
    chInfo.kind = chType;
    chInfo.unit = unit;
    chInfo.cal = cal;
    chInfo.range = 1.0;
    chInfo.eeg_loc.setZero();
    chInfo.coord_frame = FIFFV_COORD_HEAD;
    
    return chInfo;
}