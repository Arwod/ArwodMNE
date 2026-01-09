#ifndef STATS_GLOBAL_H
#define STATS_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(STATICBUILD)
#  define STATSSHARED_EXPORT
#elif defined(MNE_STATS_LIBRARY)
#  define STATSSHARED_EXPORT Q_DECL_EXPORT
#else
#  define STATSSHARED_EXPORT Q_DECL_IMPORT
#endif

namespace STATSLIB {
}

#endif // STATS_GLOBAL_H
