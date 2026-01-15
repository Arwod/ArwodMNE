#include "dics.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

namespace INVERSELIB {

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

} // NAMESPACE
