//=============================================================================================================
/**
 * @file     module_integration.h
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
 *
 * @brief    Module integration utilities for MNE-CPP libraries.
 *
 */

#ifndef INTEGRATION_H
#define INTEGRATION_H

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QObject>
#include <QSharedPointer>
#include <QString>
#include <QStringList>

//=============================================================================================================
// DEFINE NAMESPACE MNELIB
//=============================================================================================================

namespace MNELIB
{

//=============================================================================================================
/**
 * Module integration manager for all MNE-CPP libraries.
 * This class provides utilities to manage inter-module dependencies and interfaces.
 *
 * @brief Module integration manager for MNE-CPP libraries
 */
class Integration : public QObject
{
    Q_OBJECT

public:
    typedef QSharedPointer<Integration> SPtr;            /**< Shared pointer type for Integration. */
    typedef QSharedPointer<const Integration> ConstSPtr; /**< Const shared pointer type for Integration. */

    //=========================================================================================================
    /**
     * Constructs the module integration manager.
     *
     * @param[in] parent     Parent QObject (optional).
     */
    explicit Integration(QObject *parent = nullptr);

    //=========================================================================================================
    /**
     * Destructor.
     */
    ~Integration();

    //=========================================================================================================
    /**
     * Initialize all algorithm modules and resolve inter-module dependencies.
     *
     * @return true if initialization successful, false otherwise.
     */
    bool initializeModules();

    //=========================================================================================================
    /**
     * Get list of available algorithm modules.
     *
     * @return QStringList of module names.
     */
    QStringList getAvailableModules() const;

    //=========================================================================================================
    /**
     * Check if a specific module is available and initialized.
     *
     * @param[in] moduleName    Name of the module to check.
     *
     * @return true if module is available, false otherwise.
     */
    bool isModuleAvailable(const QString& moduleName) const;

    //=========================================================================================================
    /**
     * Get version information for all integrated modules.
     *
     * @return QString containing version information.
     */
    QString getVersionInfo() const;

    //=========================================================================================================
    /**
     * Verify inter-module compatibility and dependencies.
     *
     * @return true if all modules are compatible, false otherwise.
     */
    bool verifyModuleCompatibility() const;

    //=========================================================================================================
    /**
     * Get performance statistics for integrated modules.
     *
     * @return QString containing performance information.
     */
    QString getPerformanceInfo() const;

private:
    //=========================================================================================================
    /**
     * Initialize core MNE libraries.
     *
     * @return true if successful, false otherwise.
     */
    bool initializeCoreLibraries();

    //=========================================================================================================
    /**
     * Initialize algorithm modules.
     *
     * @return true if successful, false otherwise.
     */
    bool initializeAlgorithmModules();

    //=========================================================================================================
    /**
     * Resolve inter-module dependencies.
     *
     * @return true if successful, false otherwise.
     */
    bool resolveModuleDependencies();

    QStringList         m_availableModules;     /**< List of available modules. */
    QStringList         m_initializedModules;   /**< List of successfully initialized modules. */
    bool                m_isInitialized;        /**< Flag indicating if integration is initialized. */
};

} // namespace MNELIB

#endif // INTEGRATION_H