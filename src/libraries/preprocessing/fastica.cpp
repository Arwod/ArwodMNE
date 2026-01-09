#include "fastica.h"
#include "pca.h"
#include <cmath>
#include <iostream>
#include <random>
#include <Eigen/Dense>

namespace PREPROCESSINGLIB {

FastICA::FastICA(int n_components, 
                 const std::string& algorithm, 
                 const std::string& fun, 
                 int max_iter, 
                 double tol, 
                 int random_state)
    : m_n_components(n_components)
    , m_algorithm(algorithm)
    , m_fun(fun)
    , m_max_iter(max_iter)
    , m_tol(tol)
    , m_random_state(random_state)
{
}

void FastICA::fit(const Eigen::MatrixXd& data)
{
    // 1. PCA & Whitening
    PCA pca(m_n_components, true);
    pca.fit(data);
    
    // X_white is (n_components x n_samples)
    // PCA::transform returns (n_components x n_samples)
    Eigen::MatrixXd X_white = pca.transform(data);
    m_whitening_matrix = pca.get_whitening_matrix(); // (n_comp x n_channels)
    m_mean = data.rowwise().mean();
    
    int n_comp = X_white.rows();
    
    // 2. FastICA Loop
    m_w_ica = _ica_par(X_white, n_comp, m_fun, m_tol, m_max_iter);
    
    // 3. Compute Unmixing/Mixing
    // Unmixing = W * K
    m_unmixing = m_w_ica * m_whitening_matrix;
    
    // Mixing = A = W_total^+ = (W K)^+ = K^+ W^T
    // Since W is orthogonal, W^+ = W^T.
    // K = S^{-0.5} U^T. K^+ = U S^{0.5}.
    // Actually, simply pseudoinverse of unmixing.
    // Or: X = A S. S = W X. -> A = W^+
    // If W is square (n_comp x n_comp), A = W^-1.
    // But unmixing is (n_comp x n_channels).
    // So mixing is (n_channels x n_comp).
    // A = pinv(W_total).
    
    // Using SVD for pseudo-inverse
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m_unmixing, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-7 * std::max(m_unmixing.cols(), m_unmixing.rows()) * svd.singularValues().array().abs().maxCoeff();
    Eigen::VectorXd singularValuesInv = svd.singularValues();
    for (int i = 0; i < singularValuesInv.size(); ++i) {
        if (std::abs(singularValuesInv[i]) > tolerance)
            singularValuesInv[i] = 1.0 / singularValuesInv[i];
        else
            singularValuesInv[i] = 0.0;
    }
    m_mixing = svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd FastICA::transform(const Eigen::MatrixXd& data) const
{
    // S = W_total * (X - mean)
    Eigen::MatrixXd centered = data.colwise() - m_mean;
    return m_unmixing * centered;
}

Eigen::MatrixXd FastICA::inverse_transform(const Eigen::MatrixXd& data) const
{
    // X = A * S + mean
    return (m_mixing * data).colwise() + m_mean;
}

Eigen::MatrixXd FastICA::_sym_decorrelation(const Eigen::MatrixXd& W)
{
    // W <- (W W^T)^{-1/2} W
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(W * W.transpose());
    Eigen::VectorXd D = es.eigenvalues();
    Eigen::MatrixXd E = es.eigenvectors();
    
    // D^{-1/2}
    Eigen::VectorXd D_inv_sqrt(D.size());
    for(int i=0; i<D.size(); ++i) {
        if(D[i] > 1e-15)
            D_inv_sqrt[i] = 1.0 / std::sqrt(D[i]);
        else
            D_inv_sqrt[i] = 0.0;
    }
    
    return E * D_inv_sqrt.asDiagonal() * E.transpose() * W;
}

Eigen::MatrixXd FastICA::_ica_par(const Eigen::MatrixXd& X, int n_comp, const std::string& fun, double tol, int max_iter)
{
    int n_samples = X.cols();
    
    // Initialize W random orthogonal
    std::mt19937 gen(m_random_state);
    std::normal_distribution<> d(0, 1);
    
    Eigen::MatrixXd W(n_comp, n_comp);
    for(int i=0; i<n_comp; ++i)
        for(int j=0; j<n_comp; ++j)
            W(i, j) = d(gen);
            
    W = _sym_decorrelation(W);
    
    double alpha = 1.0; // for logcosh
    
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd W_old = W;
        
        // 1. Linear combination
        Eigen::MatrixXd wx = W * X; // (n_comp x n_samples)
        
        Eigen::MatrixXd g_wx(n_comp, n_samples);
        Eigen::MatrixXd g_prime_wx(n_comp, n_samples);
        
        // 2. Non-linearity
        if (fun == "logcosh") {
            // g(u) = tanh(alpha * u)
            // g'(u) = alpha * (1 - tanh^2(alpha * u))
            for(int i=0; i<wx.rows(); ++i) {
                for(int j=0; j<wx.cols(); ++j) {
                    double u = wx(i, j);
                    double th = std::tanh(alpha * u);
                    g_wx(i, j) = th;
                    g_prime_wx(i, j) = alpha * (1.0 - th * th);
                }
            }
        } else if (fun == "exp") {
            // g(u) = u * exp(-u^2/2)
            // g'(u) = (1 - u^2) * exp(-u^2/2)
            for(int i=0; i<wx.rows(); ++i) {
                for(int j=0; j<wx.cols(); ++j) {
                    double u = wx(i, j);
                    double exp_val = std::exp(-0.5 * u * u);
                    g_wx(i, j) = u * exp_val;
                    g_prime_wx(i, j) = (1.0 - u * u) * exp_val;
                }
            }
        } else if (fun == "cube") {
            // g(u) = u^3
            // g'(u) = 3u^2
             for(int i=0; i<wx.rows(); ++i) {
                for(int j=0; j<wx.cols(); ++j) {
                    double u = wx(i, j);
                    g_wx(i, j) = u * u * u;
                    g_prime_wx(i, j) = 3.0 * u * u;
                }
            }
        }
        
        // 3. Update
        // W+ = E[x g(wTx)] - E[g'(wTx)] w
        // Matrix form: W+ = 1/N * g_wx * X^T - diag(mean(g_prime_wx)) * W
        
        Eigen::MatrixXd term1 = (g_wx * X.transpose()) / (double)n_samples;
        
        Eigen::VectorXd mean_g_prime = g_prime_wx.rowwise().mean();
        Eigen::MatrixXd term2 = mean_g_prime.asDiagonal() * W;
        
        W = term1 - term2;
        
        // 4. Decorrelate
        W = _sym_decorrelation(W);
        
        // 5. Check convergence
        // lim = max_i (1 - |<w_i, w_old_i>|)
        double max_diff = 0.0;
        for(int i=0; i<n_comp; ++i) {
            double dot = W.row(i).dot(W_old.row(i));
            double diff = 1.0 - std::abs(dot);
            if (diff > max_diff) max_diff = diff;
        }
        
        if (max_diff < tol) {
            // Converged
            // std::cout << "FastICA converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }
    
    return W;
}

} // NAMESPACE
