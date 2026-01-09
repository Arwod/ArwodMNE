#ifndef TFR_GLOBAL_H
#define TFR_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(STATICBUILD)
#  define TFRSHARED_EXPORT
#elif defined(MNE_TFR_LIBRARY)
#  define TFRSHARED_EXPORT Q_DECL_EXPORT
#else
#  define TFRSHARED_EXPORT Q_DECL_IMPORT
#endif

namespace TFRLIB {
}

#endif // TFR_GLOBAL_H
