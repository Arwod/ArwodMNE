//=============================================================================================================
/**
 * @file     module_integration.cpp
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Implementation of module integration utilities.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "integration.h"

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QDebug>
#include <QDateTime>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace MNELIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

Integration::Integration(QObject *parent)
: QObject(parent)
, m_isInitialized(false)
{
    // Define all available modules
    m_availableModules << "mne_utils"
                      << "mne_fiff"
                      << "mne_fs"
                      << "mne_mne"
                      << "mne_tfr"
                      << "mne_preprocessing"
                      << "mne_inverse"
                      << "mne_stats"
                      << "mne_decoding"
                      << "mne_simulation"
                      << "mne_fwd"
                      << "mne_dataio"
                      << "mne_channels"
                      << "mne_connectivity"
                      << "mne_rtprocessing"
                      << "mne_events"
                      << "mne_communication";
}

//=============================================================================================================

Integration::~Integration()
{
}

//=============================================================================================================

bool Integration::initializeModules()
{
    qDebug() << "[ModuleIntegration] Initializing MNE-CPP modules...";
    
    m_initializedModules.clear();
    
    // Initialize core libraries first
    if (!initializeCoreLibraries()) {
        qWarning() << "[ModuleIntegration] Failed to initialize core libraries";
        return false;
    }
    
    // Initialize algorithm modules
    if (!initializeAlgorithmModules()) {
        qWarning() << "[ModuleIntegration] Failed to initialize algorithm modules";
        return false;
    }
    
    // Resolve inter-module dependencies
    if (!resolveModuleDependencies()) {
        qWarning() << "[ModuleIntegration] Failed to resolve module dependencies";
        return false;
    }
    
    m_isInitialized = true;
    qDebug() << "[ModuleIntegration] Successfully initialized" << m_initializedModules.size() << "modules";
    
    return true;
}

//=============================================================================================================

QStringList Integration::getAvailableModules() const
{
    return m_availableModules;
}

//=============================================================================================================

bool Integration::isModuleAvailable(const QString& moduleName) const
{
    return m_initializedModules.contains(moduleName);
}

//=============================================================================================================

QString Integration::getVersionInfo() const
{
    QString versionInfo;
    versionInfo += "MNE-CPP Module Integration\n";
    versionInfo += "==========================\n";
    versionInfo += QString("Build Date: %1\n").arg(__DATE__ " " __TIME__);
    versionInfo += QString("Initialized Modules: %1\n").arg(m_initializedModules.size());
    versionInfo += "\nModule List:\n";
    
    for (const QString& module : m_initializedModules) {
        versionInfo += QString("  - %1\n").arg(module);
    }
    
    return versionInfo;
}

//=============================================================================================================

bool Integration::verifyModuleCompatibility() const
{
    // Check that all critical modules are initialized
    QStringList criticalModules = {"mne_utils", "mne_fiff", "mne_mne"};
    
    for (const QString& module : criticalModules) {
        if (!m_initializedModules.contains(module)) {
            qWarning() << "[ModuleIntegration] Critical module not initialized:" << module;
            return false;
        }
    }
    
    // Verify algorithm module dependencies
    QMap<QString, QStringList> dependencies;
    dependencies["mne_connectivity"] = {"mne_utils", "mne_fiff", "mne_mne", "mne_tfr"};
    dependencies["mne_rtprocessing"] = {"mne_utils", "mne_fiff", "mne_mne", "mne_connectivity", "mne_fwd", "mne_inverse"};
    dependencies["mne_decoding"] = {"mne_utils", "mne_fiff"};
    dependencies["mne_dataio"] = {"mne_utils", "mne_fiff"};
    dependencies["mne_channels"] = {"mne_utils", "mne_fiff"};
    
    for (auto it = dependencies.begin(); it != dependencies.end(); ++it) {
        const QString& module = it.key();
        const QStringList& deps = it.value();
        
        if (m_initializedModules.contains(module)) {
            for (const QString& dep : deps) {
                if (!m_initializedModules.contains(dep)) {
                    qWarning() << "[ModuleIntegration] Module" << module << "missing dependency:" << dep;
                    return false;
                }
            }
        }
    }
    
    return true;
}

//=============================================================================================================

QString Integration::getPerformanceInfo() const
{
    QString perfInfo;
    perfInfo += "MNE-CPP Performance Information\n";
    perfInfo += "===============================\n";
    perfInfo += QString("Initialization Status: %1\n").arg(m_isInitialized ? "Success" : "Failed");
    perfInfo += QString("Total Modules: %1\n").arg(m_availableModules.size());
    perfInfo += QString("Initialized Modules: %1\n").arg(m_initializedModules.size());
    perfInfo += QString("Success Rate: %1%\n").arg(
        m_availableModules.isEmpty() ? 0 : (100 * m_initializedModules.size() / m_availableModules.size()));
    
    return perfInfo;
}

//=============================================================================================================

bool Integration::initializeCoreLibraries()
{
    QStringList coreLibraries = {"mne_utils", "mne_fiff", "mne_fs", "mne_mne"};
    
    for (const QString& lib : coreLibraries) {
        // In a real implementation, we would call actual initialization functions
        // For now, we assume they are available if they're in the available list
        if (m_availableModules.contains(lib)) {
            m_initializedModules.append(lib);
            qDebug() << "[ModuleIntegration] Initialized core library:" << lib;
        } else {
            qWarning() << "[ModuleIntegration] Core library not available:" << lib;
            return false;
        }
    }
    
    return true;
}

//=============================================================================================================

bool Integration::initializeAlgorithmModules()
{
    QStringList algorithmModules = {
        "mne_tfr",           // Task 2: Time-frequency analysis
        "mne_preprocessing", // Task 5: Preprocessing module
        "mne_inverse",       // Task 7: Minimum norm estimation
        "mne_stats",         // Task 8: Statistical analysis
        "mne_decoding",      // Task 9: Decoding and machine learning
        "mne_simulation",    // Task 12: Simulation module
        "mne_fwd",          // Task 13: Forward modeling
        "mne_dataio",       // Task 14: Data I/O module
        "mne_channels",     // Task 15: Channel and montage management
        "mne_connectivity", // Task 16: Connectivity analysis
        "mne_rtprocessing", // Task 17: Filter algorithm enhancement
        "mne_events",
        "mne_communication"
    };
    
    for (const QString& module : algorithmModules) {
        if (m_availableModules.contains(module)) {
            m_initializedModules.append(module);
            qDebug() << "[ModuleIntegration] Initialized algorithm module:" << module;
        } else {
            qDebug() << "[ModuleIntegration] Algorithm module not available (optional):" << module;
        }
    }
    
    return true;
}

//=============================================================================================================

bool Integration::resolveModuleDependencies()
{
    // Verify that module dependencies are satisfied
    if (!verifyModuleCompatibility()) {
        return false;
    }
    
    qDebug() << "[ModuleIntegration] All module dependencies resolved successfully";
    return true;
}