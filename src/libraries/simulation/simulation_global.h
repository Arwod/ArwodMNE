#ifndef SIMULATION_GLOBAL_H
#define SIMULATION_GLOBAL_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <QtCore/qglobal.h>

//=============================================================================================================
// DEFINES
//=============================================================================================================

#if defined(MNE_SIMULATION_LIBRARY)
#  define SIMULATIONSHARED_EXPORT Q_DECL_EXPORT   /**< Q_DECL_EXPORT must be added to the declarations of symbols used when compiling a shared library. */
#else
#  define SIMULATIONSHARED_EXPORT Q_DECL_IMPORT   /**< Q_DECL_IMPORT must be added to the declarations of symbols used when compiling a client that uses the shared library. */
#endif

#endif // SIMULATION_GLOBAL_H