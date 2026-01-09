#ifndef TIMEFREQUENCY_H
#define TIMEFREQUENCY_H

#include "tfr_global.h"
#include <Eigen/Core>
#include <vector>
#include <QString>

namespace TFRLIB {

class TFRSHARED_EXPORT TimeFrequency
{
public:
    TimeFrequency();
    TimeFrequency(const TimeFrequency& other);
    ~TimeFrequency();

    // Data structure: [channel][freq] -> time_series
    // This allows ragged arrays if needed, but usually rectangular.
    // Alternatively, flatten to MatrixXd (n_channels * n_freqs, n_times)?
    // Keeping it vector of vector of VectorXd is flexible for now.
    std::vector<std::vector<Eigen::VectorXd>> data; 
    
    Eigen::VectorXd freqs;      /**< Frequencies. */
    Eigen::VectorXd times;      /**< Time points. */
    double sfreq;               /**< Sampling frequency. */
    QString method;             /**< Method used (e.g. "morlet"). */
    
    // Metadata could be added (channel names, etc.)
};

} // NAMESPACE

#endif // TIMEFREQUENCY_H
