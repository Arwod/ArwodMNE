#include "timefrequency.h"

namespace TFRLIB {

TimeFrequency::TimeFrequency()
: sfreq(0.0)
{
}

TimeFrequency::TimeFrequency(const TimeFrequency& other)
: data(other.data)
, freqs(other.freqs)
, times(other.times)
, sfreq(other.sfreq)
, method(other.method)
{
}

TimeFrequency::~TimeFrequency()
{
}

} // NAMESPACE
