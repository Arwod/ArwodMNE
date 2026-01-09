#ifndef ICA_H
#define ICA_H

#include "preprocessing_global.h"
#include "fastica.h"
#include <Eigen/Core>
#include <vector>
#include <string>

namespace PREPROCESSINGLIB {

class PREPROCESSINGSHARED_EXPORT ICA
{
public:
    ICA(int n_components = 0, 
        const std::string& method = "fastica", 
        const std::string& fit_params = "logcosh",
        int random_state = 0);
        
    void fit(const Eigen::MatrixXd& data);
    
    /**
     * Apply ICA cleaning to data.
     * Removes components in 'exclude'.
     * 
     * @param[in] data Input data (n_channels x n_times).
     * @param[in] exclude List of component indices to exclude.
     * @return Cleaned data.
     */
    Eigen::MatrixXd apply(const Eigen::MatrixXd& data, const std::vector<int>& exclude = std::vector<int>()) const;
    
    // Getters
    Eigen::MatrixXd get_mixing_matrix() const;
    Eigen::MatrixXd get_unmixing_matrix() const;
    Eigen::MatrixXd get_sources(const Eigen::MatrixXd& data) const;
    
    // Set components to exclude
    void set_exclude(const std::vector<int>& exclude) { m_exclude = exclude; }
    std::vector<int> get_exclude() const { return m_exclude; }

private:
    int m_n_components;
    std::string m_method;
    std::string m_fit_params;
    int m_random_state;
    
    std::vector<int> m_exclude;
    
    // Underlying algorithm
    // Currently only FastICA
    FastICA* m_fastica;
};

} // NAMESPACE

#endif // ICA_H
