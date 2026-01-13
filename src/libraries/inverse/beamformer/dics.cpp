#include "dics.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace INVERSELIB {

DicsFilter DICS::make_dics(const Eigen::MatrixXd& leadfield,
                           const TFRLIB::CSD& csd,
                           double reg,
                           const TFRLIB::CSD* noise_csd,
                           const std::string& pick_ori,
                           const std::string& weight_norm,
                           const std::string& inversion,
                           bool real_filter,
                           int n_ori)
{
    DicsFilter filters;
    
    // Validate inputs
    if (csd.data.empty()) {
        std::cerr << "DICS::make_dics: CSD is empty." << std::endl;
        return filters;
    }
    
    int n_channels = leadfield.rows();
    if (n_channels != csd.data[0].rows()) {
        std::cerr << "DICS::make_dics: Leadfield rows (" << n_channels 
                  << ") mismatch CSD channels (" << csd.data[0].rows() << ")." << std::endl;
        return filters;
    }
    
    // Set filter parameters
    filters.pick_ori = pick_ori;
    filters.weight_norm = weight_norm;
    filters.inversion = inversion;
    filters.real_filter = real_filter;
    filters.n_channels = n_channels;
    filters.frequencies = csd.freqs;
    filters.is_free_ori = (n_ori > 1);
    
    int n_dipoles = leadfield.cols();
    if (n_dipoles % n_ori != 0) {
        std::cerr << "DICS::make_dics: Leadfield cols not divisible by n_ori." << std::endl;
        return filters;
    }
    
    filters.n_sources = n_dipoles / n_ori;
    
    // Compute the filters
    filters.weights = compute_dics_filters(leadfield, csd, reg, noise_csd, 
                                          n_ori, real_filter, pick_ori, 
                                          weight_norm, inversion);
    
    return filters;
}

std::pair<Eigen::MatrixXd, std::vector<double>> DICS::apply_dics_csd(
    const TFRLIB::CSD& csd,
    const DicsFilter& filters)
{
    if (filters.weights.empty()) {
        std::cerr << "DICS::apply_dics_csd: Filters are empty." << std::endl;
        return std::make_pair(Eigen::MatrixXd(), std::vector<double>());
    }
    
    int n_freqs = filters.frequencies.size();
    int n_sources = filters.n_sources;
    
    Eigen::MatrixXd source_power = Eigen::MatrixXd::Zero(n_sources, n_freqs);
    
    // Apply filters to each frequency
    for (int f = 0; f < n_freqs; ++f) {
        if (f >= static_cast<int>(csd.data.size())) {
            std::cerr << "DICS::apply_dics_csd: CSD frequency index out of bounds." << std::endl;
            continue;
        }
        
        Eigen::MatrixXcd Cm = csd.data[f];
        Eigen::MatrixXcd W = filters.weights[f];
        
        // Compute source power: P = W * C * W^H
        Eigen::MatrixXcd P_complex = W * Cm * W.adjoint();
        
        // Extract real power (diagonal elements)
        for (int src = 0; src < n_sources; ++src) {
            source_power(src, f) = P_complex(src, src).real();
        }
    }
    
    return std::make_pair(source_power, filters.frequencies);
}

Eigen::MatrixXd DICS::compute_source_power(const Eigen::MatrixXd& leadfield, 
                                           const TFRLIB::CSD& csd, 
                                           double reg,
                                           int n_ori,
                                           bool real_filter)
{
    if (csd.data.empty()) {
        std::cerr << "DICS::compute_source_power: CSD is empty." << std::endl;
        return Eigen::MatrixXd();
    }
    
    int n_channels = leadfield.rows();
    if (n_channels != csd.data[0].rows()) {
        std::cerr << "DICS::compute_source_power: Leadfield rows (" << n_channels 
                  << ") mismatch CSD channels (" << csd.data[0].rows() << ")." << std::endl;
        return Eigen::MatrixXd();
    }
    
    if (n_ori < 1) n_ori = 1;
    int n_dipoles = leadfield.cols();
    if (n_dipoles % n_ori != 0) {
        std::cerr << "DICS::compute_source_power: Leadfield cols not divisible by n_ori." << std::endl;
        return Eigen::MatrixXd();
    }
    
    int n_sources = n_dipoles / n_ori;
    int n_freqs = csd.freqs.size();
    
    Eigen::MatrixXd power_map = Eigen::MatrixXd::Zero(n_sources, n_freqs);
    
    // Iterate over frequencies
    for (int i = 0; i < n_freqs; ++i) {
        Eigen::MatrixXcd Cm = csd.data[i]; // Original CSD (Complex)
        
        // Prepare C for inversion (Real or Complex)
        // Usually DICS uses Real part for stable filter direction
        
        Eigen::MatrixXcd C_inv;
        
        if (real_filter) {
            Eigen::MatrixXd Cr = Cm.real();
            
            // Regularization
            double trace = Cr.trace();
            double lambda = reg * trace / (double)n_channels;
            Cr += lambda * Eigen::MatrixXd::Identity(n_channels, n_channels);
            
            // Invert
            // Use SelfAdjointEigenSolver for robust inversion of symmetric positive definite matrix
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cr);
            Eigen::MatrixXd inv_Cr = es.eigenvectors() * es.eigenvalues().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
            
            // Cast back to complex for subsequent math or keep real?
            // Filter W calculation involves L (real) and C_inv (real). So W is real.
            // But Power = W * Cm * W^T involves Cm (complex).
            // So let's store C_inv as complex for uniform handling.
            C_inv = inv_Cr.cast<std::complex<double>>();
        } else {
            // Complex case
            // Regularization: add to real diagonal
            double trace = Cm.trace().real();
            double lambda = reg * trace / (double)n_channels;
            
            for(int k=0; k<n_channels; ++k) {
                Cm(k,k) += std::complex<double>(lambda, 0.0);
            }
            
            // Invert Complex Hermitian
            // Eigen doesn't have Complex SelfAdjoint solver fully exposed/convenient in all versions,
            // but LLT or LDLT works for Hermitian Positive Definite.
            // Or use FullPivLU for safety.
            C_inv = Cm.inverse();
        }
        
        // Loop over sources
        // TODO: Parallelize with OpenMP
        #pragma omp parallel for
        for (int src = 0; src < n_sources; ++src) {
            // Extract Leadfield for this source: L_s (Ch x n_ori)
            Eigen::MatrixXd L_s = leadfield.block(0, src * n_ori, n_channels, n_ori);
            
            // Cast to complex for multiplication with C_inv
            Eigen::MatrixXcd L_sc = L_s.cast<std::complex<double>>();
            
            // Compute Denominator: (L^T * C_inv * L)
            // L_sc.adjoint() is L^T (since L is real)
            Eigen::MatrixXcd denom = L_sc.adjoint() * C_inv * L_sc;
            
            // Inverse denominator
            // denom is n_ori x n_ori (small, e.g. 1x1 or 3x3)
            Eigen::MatrixXcd denom_inv = denom.inverse();
            
            // Compute Filter: W = denom_inv * L^T * C_inv
            // W is (n_ori x n_channels)
            Eigen::MatrixXcd W = denom_inv * L_sc.adjoint() * C_inv;
            
            // Compute Power: P = W * C_original * W^H
            // Note: Use the original un-regularized CSD for power estimation
            Eigen::MatrixXcd P_mat = W * csd.data[i] * W.adjoint();
            
            // Trace (sum of power of all orientations)
            power_map(src, i) = P_mat.trace().real();
        }
    }
    
    return power_map;
}

std::vector<Eigen::MatrixXcd> DICS::apply_dics_epochs(
    const std::vector<Eigen::MatrixXd>& epochs,
    const DicsFilter& filters)
{
    std::vector<Eigen::MatrixXcd> source_epochs;
    
    if (filters.weights.empty()) {
        std::cerr << "DICS::apply_dics_epochs: Filters are empty." << std::endl;
        return source_epochs;
    }
    
    int n_sources = filters.n_sources;
    int n_freqs = filters.weights.size();
    
    // Apply filters to each epoch
    for (const auto& epoch : epochs) {
        int n_times = epoch.cols();
        
        // Convert epoch to complex
        Eigen::MatrixXcd epoch_complex = epoch.cast<std::complex<double>>();
        
        // Apply filter for each frequency (simplified approach - average across frequencies)
        Eigen::MatrixXcd source_epoch = Eigen::MatrixXcd::Zero(n_sources, n_times);
        
        for (int f = 0; f < n_freqs; ++f) {
            source_epoch += filters.weights[f] * epoch_complex;
        }
        source_epoch /= static_cast<double>(n_freqs);
        
        source_epochs.push_back(source_epoch);
    }
    
    return source_epochs;
}

Eigen::MatrixXcd DICS::apply_dics_raw(
    const Eigen::MatrixXd& data,
    const DicsFilter& filters)
{
    if (filters.weights.empty()) {
        std::cerr << "DICS::apply_dics_raw: Filters are empty." << std::endl;
        return Eigen::MatrixXcd();
    }
    
    int n_sources = filters.n_sources;
    int n_times = data.cols();
    int n_freqs = filters.weights.size();
    
    // Convert data to complex
    Eigen::MatrixXcd data_complex = data.cast<std::complex<double>>();
    
    // Apply filters (simplified approach - average across frequencies)
    Eigen::MatrixXcd source_data = Eigen::MatrixXcd::Zero(n_sources, n_times);
    
    for (int f = 0; f < n_freqs; ++f) {
        source_data += filters.weights[f] * data_complex;
    }
    source_data /= static_cast<double>(n_freqs);
    
    return source_data;
}

std::vector<std::vector<Eigen::MatrixXcd>> DICS::apply_dics_tfr_epochs(
    const std::vector<std::vector<Eigen::MatrixXcd>>& epochs_tfr,
    const DicsFilter& filters)
{
    std::vector<std::vector<Eigen::MatrixXcd>> source_epochs_tfr;
    
    if (filters.weights.empty()) {
        std::cerr << "DICS::apply_dics_tfr_epochs: Filters are empty." << std::endl;
        return source_epochs_tfr;
    }
    
    int n_sources = filters.n_sources;
    
    // Apply filters to each epoch and frequency
    for (const auto& epoch_tfr : epochs_tfr) {
        std::vector<Eigen::MatrixXcd> source_epoch_freqs;
        
        for (size_t f = 0; f < epoch_tfr.size() && f < filters.weights.size(); ++f) {
            // Apply filter for this frequency
            Eigen::MatrixXcd source_f = filters.weights[f] * epoch_tfr[f];
            source_epoch_freqs.push_back(source_f);
        }
        
        source_epochs_tfr.push_back(source_epoch_freqs);
    }
    
    return source_epochs_tfr;
}

std::vector<Eigen::MatrixXcd> DICS::compute_dics_filters(
    const Eigen::MatrixXd& leadfield,
    const TFRLIB::CSD& csd,
    double reg,
    const TFRLIB::CSD* noise_csd,
    int n_ori,
    bool real_filter,
    const std::string& pick_ori,
    const std::string& weight_norm,
    const std::string& inversion)
{
    if (csd.data.empty()) {
        std::cerr << "DICS::compute_dics_filters: CSD is empty." << std::endl;
        return {};
    }
    
    int n_channels = leadfield.rows();
    int n_dipoles = leadfield.cols();
    int n_sources = n_dipoles / n_ori;
    int n_freqs = csd.freqs.size();
    
    std::vector<Eigen::MatrixXcd> filters;
    
    // Compute whitening matrix if noise CSD is provided
    Eigen::MatrixXcd whitener = Eigen::MatrixXcd::Identity(n_channels, n_channels);
    if (noise_csd != nullptr) {
        whitener = compute_whitener(*noise_csd, reg);
    }
    
    // Compute filters for each frequency
    for (int f = 0; f < n_freqs; ++f) {
        Eigen::MatrixXcd Cm = csd.data[f];
        
        // Apply whitening
        if (noise_csd != nullptr) {
            Cm = whitener * Cm * whitener.adjoint();
        }
        
        // Prepare C for inversion
        Eigen::MatrixXcd C_inv;
        
        if (real_filter) {
            Eigen::MatrixXd Cr = Cm.real();
            double trace = Cr.trace();
            double lambda = reg * trace / static_cast<double>(n_channels);
            Cr += lambda * Eigen::MatrixXd::Identity(n_channels, n_channels);
            
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cr);
            Eigen::MatrixXd inv_Cr = es.eigenvectors() * 
                es.eigenvalues().cwiseInverse().asDiagonal() * 
                es.eigenvectors().transpose();
            C_inv = inv_Cr.cast<std::complex<double>>();
        } else {
            double trace = Cm.trace().real();
            double lambda = reg * trace / static_cast<double>(n_channels);
            
            for (int k = 0; k < n_channels; ++k) {
                Cm(k, k) += std::complex<double>(lambda, 0.0);
            }
            C_inv = Cm.inverse();
        }
        
        // Apply whitening to leadfield
        Eigen::MatrixXcd L_white = whitener * leadfield.cast<std::complex<double>>();
        
        // Compute filters for this frequency
        int out_rows = (pick_ori == "vector") ? n_dipoles : n_sources;
        Eigen::MatrixXcd W_f = Eigen::MatrixXcd::Zero(out_rows, n_channels);
        
        if (inversion == "matrix") {
            // Matrix inversion approach - compute all sources jointly
            for (int src = 0; src < n_sources; ++src) {
                Eigen::MatrixXcd L_s = L_white.block(0, src * n_ori, n_channels, n_ori);
                
                // Compute filter for this source
                Eigen::MatrixXcd denom = L_s.adjoint() * C_inv * L_s;
                Eigen::MatrixXcd denom_inv = denom.inverse();
                Eigen::MatrixXcd W_s = denom_inv * L_s.adjoint() * C_inv;
                
                // Apply orientation selection
                if (pick_ori == "vector") {
                    W_f.block(src * n_ori, 0, n_ori, n_channels) = W_s;
                } else if (pick_ori == "max-power") {
                    // Find max power orientation using real part of CSD
                    Eigen::MatrixXd P = W_s.real() * csd.data[f].real() * W_s.real().transpose();
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_p(P);
                    Eigen::VectorXd max_ori = es_p.eigenvectors().col(n_ori - 1);
                    
                    Eigen::MatrixXcd w_scalar = max_ori.transpose().cast<std::complex<double>>() * W_s;
                    W_f.row(src) = w_scalar;
                } else if (pick_ori == "normal") {
                    // Use first (or last) orientation as normal
                    W_f.row(src) = W_s.row(0);
                }
            }
        } else {
            // Single inversion approach - compute each source separately
            for (int src = 0; src < n_sources; ++src) {
                for (int ori = 0; ori < n_ori; ++ori) {
                    int col_idx = src * n_ori + ori;
                    Eigen::VectorXcd l = L_white.col(col_idx);
                    
                    // Single dipole filter: w = (l^T * C_inv * l)^-1 * l^T * C_inv
                    std::complex<double> denom = l.adjoint() * C_inv * l;
                    Eigen::VectorXcd w = (1.0 / denom) * l.adjoint() * C_inv;
                    
                    if (pick_ori == "vector") {
                        W_f.row(col_idx) = w.transpose();
                    } else if (ori == 0) { // For scalar orientations, use first orientation
                        W_f.row(src) = w.transpose();
                    }
                }
            }
        }
        
        filters.push_back(W_f);
    }
    
    // Apply weight normalization if requested
    if (weight_norm != "none") {
        filters = apply_weight_normalization(filters, leadfield, csd, noise_csd, weight_norm, n_ori);
    }
    
    return filters;
}

std::vector<Eigen::MatrixXcd> DICS::apply_weight_normalization(
    const std::vector<Eigen::MatrixXcd>& filters,
    const Eigen::MatrixXd& leadfield,
    const TFRLIB::CSD& csd,
    const TFRLIB::CSD* noise_csd,
    const std::string& weight_norm,
    int n_ori)
{
    std::vector<Eigen::MatrixXcd> normalized_filters = filters;
    
    if (weight_norm == "unit-noise-gain") {
        // Normalize each filter to have unit noise gain
        for (size_t f = 0; f < filters.size(); ++f) {
            Eigen::MatrixXcd& W = normalized_filters[f];
            
            for (int src = 0; src < W.rows(); ++src) {
                Eigen::VectorXcd w = W.row(src);
                double noise_gain = std::sqrt((w.adjoint() * w).real());
                if (noise_gain > 0) {
                    W.row(src) /= noise_gain;
                }
            }
        }
    } else if (weight_norm == "unit-noise-gain-invariant") {
        // More sophisticated normalization that accounts for leadfield
        for (size_t f = 0; f < filters.size(); ++f) {
            Eigen::MatrixXcd& W = normalized_filters[f];
            Eigen::MatrixXcd L = leadfield.cast<std::complex<double>>();
            
            for (int src = 0; src < W.rows(); ++src) {
                Eigen::VectorXcd w = W.row(src);
                
                // Compute leadfield for this source
                int start_col = (src / (W.rows() / (leadfield.cols() / n_ori))) * n_ori;
                Eigen::MatrixXcd L_s = L.block(0, start_col, L.rows(), n_ori);
                
                // Normalization factor
                std::complex<double> norm_factor = w.adjoint() * L_s * L_s.adjoint() * w;
                if (std::abs(norm_factor) > 0) {
                    W.row(src) /= std::sqrt(norm_factor);
                }
            }
        }
    }
    
    return normalized_filters;
}

Eigen::MatrixXcd DICS::compute_whitener(const TFRLIB::CSD& noise_csd, double reg)
{
    if (noise_csd.data.empty()) {
        std::cerr << "DICS::compute_whitener: Noise CSD is empty." << std::endl;
        return Eigen::MatrixXcd::Identity(1, 1);
    }
    
    // Use the first frequency or average across frequencies
    Eigen::MatrixXcd noise_C = noise_csd.data[0];
    
    // If multiple frequencies, average them
    if (noise_csd.data.size() > 1) {
        noise_C.setZero();
        for (const auto& C_f : noise_csd.data) {
            noise_C += C_f;
        }
        noise_C /= static_cast<double>(noise_csd.data.size());
    }
    
    // Take real part for stability
    Eigen::MatrixXd noise_C_real = noise_C.real();
    
    // Regularize
    double trace = noise_C_real.trace();
    double lambda = reg * trace / static_cast<double>(noise_C_real.rows());
    noise_C_real += lambda * Eigen::MatrixXd::Identity(noise_C_real.rows(), noise_C_real.cols());
    
    // Compute whitening matrix using eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(noise_C_real);
    Eigen::MatrixXd whitener_real = es.eigenvectors() * 
        es.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * 
        es.eigenvectors().transpose();
    
    return whitener_real.cast<std::complex<double>>();
}

std::vector<Eigen::MatrixXcd> DICS::apply_filters_fft(
    const std::vector<Eigen::MatrixXcd>& filters,
    const std::vector<Eigen::MatrixXcd>& data_fft)
{
    if (filters.size() != data_fft.size()) {
        std::cerr << "DICS::apply_filters_fft: Filter and data frequency dimensions mismatch." << std::endl;
        return {};
    }
    
    std::vector<Eigen::MatrixXcd> source_fft;
    
    for (size_t f = 0; f < filters.size(); ++f) {
        // Apply filter: source_fft[f] = filters[f] * data_fft[f]
        Eigen::MatrixXcd source_f = filters[f] * data_fft[f];
        source_fft.push_back(source_f);
    }
    
    return source_fft;
}

} // NAMESPACE
