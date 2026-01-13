#ifndef ICA_H
#define ICA_H

#include "preprocessing_global.h"
#include "fastica.h"
#include "infomax.h"
#include "picard.h"
#include <Eigen/Core>
#include <vector>
#include <map>

namespace PREPROCESSINGLIB {

class PREPROCESSINGSHARED_EXPORT ICA
{
public:
    ICA(int n_components = 0, 
        const std::string& method = "fastica", 
        const std::string& fit_params = "logcosh",
        int random_state = 0,
        int max_iter = 200,
        double tol = 1e-4);
        
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
    
    // Enhanced functionality
    bool converged() const;
    int get_n_iter() const;
    std::string get_method() const { return m_method; }
    
    /**
     * Fit ICA with specific algorithm parameters.
     * 
     * @param[in] data Input data (n_channels x n_times).
     * @param[in] algorithm_params Additional parameters specific to the algorithm.
     */
    void fit_with_params(const Eigen::MatrixXd& data, const std::map<std::string, double>& algorithm_params);

private:
    int m_n_components;
    std::string m_method;
    std::string m_fit_params;
    int m_random_state;
    int m_max_iter;
    double m_tol;
    
    std::vector<int> m_exclude;
    
    // Underlying algorithms
    FastICA* m_fastica;
    Infomax* m_infomax;
    Picard* m_picard;
};

} // NAMESPACE

#endif // ICA_H
