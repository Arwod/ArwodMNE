#include "ica.h"
#include "pca.h"
#include <iostream>

namespace PREPROCESSINGLIB {

ICA::ICA(int n_components, const std::string& method, const std::string& fit_params, 
         int random_state, int max_iter, double tol)
    : m_n_components(n_components)
    , m_method(method)
    , m_fit_params(fit_params)
    , m_random_state(random_state)
    , m_max_iter(max_iter)
    , m_tol(tol)
    , m_fastica(nullptr)
    , m_infomax(nullptr)
    , m_picard(nullptr)
{
}

void ICA::fit(const Eigen::MatrixXd& data)
{
    if (m_method == "fastica") {
        if (m_fastica) delete m_fastica;
        m_fastica = new FastICA(m_n_components, "parallel", m_fit_params, m_max_iter, m_tol, m_random_state);
        m_fastica->fit(data);
    } else if (m_method == "infomax") {
        if (m_infomax) delete m_infomax;
        bool extended = (m_fit_params == "extended");
        m_infomax = new Infomax(m_n_components, extended, 0.001, m_max_iter, m_tol, m_random_state);
        
        // For Infomax, we need to do PCA whitening first
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        
        m_infomax->fit(whitened_data);
    } else if (m_method == "picard") {
        if (m_picard) delete m_picard;
        m_picard = new Picard(m_n_components, m_fit_params, true, m_max_iter, m_tol, m_random_state);
        
        // For Picard, we need to do PCA whitening first
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        
        m_picard->fit(whitened_data);
    } else {
        std::cerr << "ICA method " << m_method << " not implemented. Available methods: fastica, infomax, picard" << std::endl;
    }
}

void ICA::fit_with_params(const Eigen::MatrixXd& data, const std::map<std::string, double>& algorithm_params)
{
    if (m_method == "fastica") {
        if (m_fastica) delete m_fastica;
        
        // Extract parameters with defaults
        int max_iter = algorithm_params.count("max_iter") ? static_cast<int>(algorithm_params.at("max_iter")) : m_max_iter;
        double tol = algorithm_params.count("tol") ? algorithm_params.at("tol") : m_tol;
        
        m_fastica = new FastICA(m_n_components, "parallel", m_fit_params, max_iter, tol, m_random_state);
        m_fastica->fit(data);
        
    } else if (m_method == "infomax") {
        if (m_infomax) delete m_infomax;
        
        // Extract parameters with defaults
        bool extended = (m_fit_params == "extended");
        double learning_rate = algorithm_params.count("learning_rate") ? algorithm_params.at("learning_rate") : 0.001;
        int max_iter = algorithm_params.count("max_iter") ? static_cast<int>(algorithm_params.at("max_iter")) : m_max_iter;
        double tol = algorithm_params.count("tol") ? algorithm_params.at("tol") : m_tol;
        
        m_infomax = new Infomax(m_n_components, extended, learning_rate, max_iter, tol, m_random_state);
        
        // PCA whitening
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        
        m_infomax->fit(whitened_data);
        
    } else if (m_method == "picard") {
        if (m_picard) delete m_picard;
        
        // Extract parameters with defaults
        bool ortho = algorithm_params.count("ortho") ? (algorithm_params.at("ortho") > 0.5) : true;
        int max_iter = algorithm_params.count("max_iter") ? static_cast<int>(algorithm_params.at("max_iter")) : m_max_iter;
        double tol = algorithm_params.count("tol") ? algorithm_params.at("tol") : m_tol;
        
        m_picard = new Picard(m_n_components, m_fit_params, ortho, max_iter, tol, m_random_state);
        
        // PCA whitening
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        
        m_picard->fit(whitened_data);
        
    } else {
        std::cerr << "ICA method " << m_method << " not implemented." << std::endl;
    }
}

Eigen::MatrixXd ICA::apply(const Eigen::MatrixXd& data, const std::vector<int>& exclude) const
{
    Eigen::MatrixXd sources;
    
    if (m_method == "fastica" && m_fastica) {
        sources = m_fastica->transform(data);
    } else if (m_method == "infomax" && m_infomax) {
        // Need to apply PCA whitening first
        PCA pca(m_n_components, true);
        pca.fit(data); // This should ideally be stored from fit()
        Eigen::MatrixXd whitened_data = pca.transform(data);
        sources = m_infomax->transform(whitened_data);
    } else if (m_method == "picard" && m_picard) {
        // Need to apply PCA whitening first
        PCA pca(m_n_components, true);
        pca.fit(data); // This should ideally be stored from fit()
        Eigen::MatrixXd whitened_data = pca.transform(data);
        sources = m_picard->transform(whitened_data);
    } else {
        return data; // No processing if algorithm not available
    }
    
    // Zero out excluded components
    std::vector<int> final_exclude = exclude;
    if (final_exclude.empty()) final_exclude = m_exclude;
    
    for (int idx : final_exclude) {
        if (idx >= 0 && idx < sources.rows()) {
            sources.row(idx).setZero();
        }
    }
    
    // Reconstruct
    if (m_method == "fastica" && m_fastica) {
        return m_fastica->inverse_transform(sources);
    } else if (m_method == "infomax" && m_infomax) {
        Eigen::MatrixXd reconstructed = m_infomax->inverse_transform(sources);
        // Apply inverse PCA whitening (this is simplified - should store PCA from fit)
        return reconstructed;
    } else if (m_method == "picard" && m_picard) {
        Eigen::MatrixXd reconstructed = m_picard->inverse_transform(sources);
        // Apply inverse PCA whitening (this is simplified - should store PCA from fit)
        return reconstructed;
    }
    
    return data;
}

Eigen::MatrixXd ICA::get_mixing_matrix() const
{
    if (m_method == "fastica" && m_fastica) {
        return m_fastica->get_mixing_matrix();
    } else if (m_method == "infomax" && m_infomax) {
        return m_infomax->get_mixing_matrix();
    } else if (m_method == "picard" && m_picard) {
        return m_picard->get_mixing_matrix();
    }
    return Eigen::MatrixXd();
}

Eigen::MatrixXd ICA::get_unmixing_matrix() const
{
    if (m_method == "fastica" && m_fastica) {
        return m_fastica->get_unmixing_matrix();
    } else if (m_method == "infomax" && m_infomax) {
        return m_infomax->get_unmixing_matrix();
    } else if (m_method == "picard" && m_picard) {
        return m_picard->get_unmixing_matrix();
    }
    return Eigen::MatrixXd();
}

Eigen::MatrixXd ICA::get_sources(const Eigen::MatrixXd& data) const
{
    if (m_method == "fastica" && m_fastica) {
        return m_fastica->transform(data);
    } else if (m_method == "infomax" && m_infomax) {
        // Need PCA whitening first (simplified)
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        return m_infomax->transform(whitened_data);
    } else if (m_method == "picard" && m_picard) {
        // Need PCA whitening first (simplified)
        PCA pca(m_n_components, true);
        pca.fit(data);
        Eigen::MatrixXd whitened_data = pca.transform(data);
        return m_picard->transform(whitened_data);
    }
    return Eigen::MatrixXd();
}

bool ICA::converged() const
{
    if (m_method == "fastica") {
        return true; // FastICA doesn't track convergence in current implementation
    } else if (m_method == "infomax" && m_infomax) {
        return m_infomax->converged();
    } else if (m_method == "picard" && m_picard) {
        return m_picard->converged();
    }
    return false;
}

int ICA::get_n_iter() const
{
    if (m_method == "fastica") {
        return -1; // FastICA doesn't track iterations in current implementation
    } else if (m_method == "infomax" && m_infomax) {
        return m_infomax->get_n_iter();
    } else if (m_method == "picard" && m_picard) {
        return m_picard->get_n_iter();
    }
    return -1;
}

} // NAMESPACE
