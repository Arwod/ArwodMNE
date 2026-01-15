#ifndef DECODING_GLOBAL_H
#define DECODING_GLOBAL_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <QtCore/qglobal.h>
#include <utils/buildinfo.h>

//=============================================================================================================
// DEFINES
//=============================================================================================================

#if defined(STATICBUILD)
#  define DECODINGSHARED_EXPORT
#elif defined(MNE_DECODING_LIBRARY)
#  define DECODINGSHARED_EXPORT Q_DECL_EXPORT    /**< Q_DECL_EXPORT must be added to the declarations of symbols used when compiling a shared library. */
#else
#  define DECODINGSHARED_EXPORT Q_DECL_IMPORT    /**< Q_DECL_IMPORT must be added to the declarations of symbols used when compiling a client that uses the shared library. */
#endif

namespace DECODINGLIB{

//=============================================================================================================
/**
 * Returns build date and time.
 */
DECODINGSHARED_EXPORT const char* buildDateTime();

//=============================================================================================================
/**
 * Returns abbreviated build git hash.
 */
DECODINGSHARED_EXPORT const char* buildHash();

//=============================================================================================================
/**
 * Returns full build git hash.
 */
DECODINGSHARED_EXPORT const char* buildHashLong();
}

#endif // DECODING_GLOBAL_H
