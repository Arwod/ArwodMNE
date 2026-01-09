#include "ica.h"
#include <iostream>

namespace PREPROCESSINGLIB {

ICA::ICA(int n_components, const std::string& method, const std::string& fit_params, int random_state)
    : m_n_components(n_components)
    , m_method(method)
    , m_fit_params(fit_params)
    , m_random_state(random_state)
    , m_fastica(nullptr)
{
}

void ICA::fit(const Eigen::MatrixXd& data)
{
    if (m_method == "fastica") {
        if (m_fastica) delete m_fastica;
        m_fastica = new FastICA(m_n_components, "parallel", m_fit_params, 200, 1e-4, m_random_state);
        m_fastica->fit(data);
    } else {
        std::cerr << "ICA method " << m_method << " not implemented." << std::endl;
    }
}

Eigen::MatrixXd ICA::apply(const Eigen::MatrixXd& data, const std::vector<int>& exclude) const
{
    if (!m_fastica) return data;
    
    // 1. Get sources
    Eigen::MatrixXd sources = m_fastica->transform(data);
    
    // 2. Zero out excluded components
    std::vector<int> final_exclude = exclude;
    if (final_exclude.empty()) final_exclude = m_exclude;
    
    for (int idx : final_exclude) {
        if (idx >= 0 && idx < sources.rows()) {
            sources.row(idx).setZero();
        }
    }
    
    // 3. Reconstruct
    // X = A * S + mean (handled by inverse_transform)
    return m_fastica->inverse_transform(sources);
}

Eigen::MatrixXd ICA::get_mixing_matrix() const
{
    if (m_fastica) return m_fastica->get_mixing_matrix();
    return Eigen::MatrixXd();
}

Eigen::MatrixXd ICA::get_unmixing_matrix() const
{
    if (m_fastica) return m_fastica->get_unmixing_matrix();
    return Eigen::MatrixXd();
}

Eigen::MatrixXd ICA::get_sources(const Eigen::MatrixXd& data) const
{
    if (m_fastica) return m_fastica->transform(data);
    return Eigen::MatrixXd();
}

} // NAMESPACE
