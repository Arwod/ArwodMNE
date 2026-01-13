#include "picard.h"
#include <cmath>
#include <iostream>
#include <random>
#include <Eigen/Dense>

namespace PREPROCESSINGLIB {

Picard::Picard(int n_components,
               const std::string& fun,
               double ortho,
               int max_iter,
               double tol,
               int random_state)
    : m_n_components(n_components)
    , m_fun(fun)
    , m_ortho(ortho)
    , m_max_iter(max_iter)
    , m_tol(tol)
    , m_random_state(random_state)
    , m_converged(false)
    , m_n_iter(0)
    , m_alpha(1.0)
{
}

void Picard::fit(const Eigen::MatrixXd& data)
{
    int n_comp = data.rows();
    int n_samples = data.cols();
    
    if (m_n_components == 0) {
        m_n_components = n_comp;
    }
    
    // Initialize unmixing matrix W randomly and orthogonalize
    std::mt19937 gen(m_random_state);
    std::normal_distribution<> d(0, 1);
    
    m_W = Eigen::MatrixXd(m_n_components, n_comp);
    for (int i = 0; i < m_n_components; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            m_W(i, j) = d(gen);
        }
    }
    
    // Initial orthogonalization
    if (m_ortho) {
        m_W = orthogonalize(m_W);
    }
    
    // Picard iteration
    double prev_objective = compute_objective(m_W, data);
    
    for (m_n_iter = 0; m_n_iter < m_max_iter; ++m_n_iter) {
        // Compute sources
        Eigen::MatrixXd Y = m_W * data;
        
        // Compute contrast function and derivatives
        Eigen::MatrixXd psi;
        Eigen::VectorXd psiy;
        compute_contrast(Y, psi, psiy);
        
        // Compute gradient
        Eigen::MatrixXd gradient = (psi * data.transpose()) / static_cast<double>(n_samples) -
                                  psiy.asDiagonal() * m_W;
        
        // Picard update with line search
        double step_size = line_search(m_W, gradient, data);
        
        // Update W
        Eigen::MatrixXd W_new = m_W + step_size * gradient;
        
        // Orthogonalization
        if (m_ortho) {
            W_new = orthogonalize(W_new);
        }
        
        // Check convergence
        double objective = compute_objective(W_new, data);
        double relative_change = std::abs(objective - prev_objective) / std::abs(prev_objective);
        
        if (relative_change < m_tol) {
            m_converged = true;
            std::cout << "Picard converged after " << m_n_iter + 1 << " iterations." << std::endl;
            break;
        }
        
        m_W = W_new;
        prev_objective = objective;
        
        // Print progress every 20 iterations
        if ((m_n_iter + 1) % 20 == 0) {
            std::cout << "Picard iteration " << m_n_iter + 1 
                      << ", objective: " << objective 
                      << ", change: " << relative_change << std::endl;
        }
    }
    
    if (!m_converged) {
        std::cout << "Picard did not converge after " << m_max_iter << " iterations." << std::endl;
    }
    
    // Compute mixing matrix (pseudo-inverse of W)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m_W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-10 * std::max(m_W.cols(), m_W.rows()) * svd.singularValues().array().abs().maxCoeff();
    
    Eigen::VectorXd singularValuesInv = svd.singularValues();
    for (int i = 0; i < singularValuesInv.size(); ++i) {
        if (std::abs(singularValuesInv[i]) > tolerance) {
            singularValuesInv[i] = 1.0 / singularValuesInv[i];
        } else {
            singularValuesInv[i] = 0.0;
        }
    }
    
    m_A = svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd Picard::transform(const Eigen::MatrixXd& data) const
{
    return m_W * data;
}

Eigen::MatrixXd Picard::inverse_transform(const Eigen::MatrixXd& sources) const
{
    return m_A * sources;
}

void Picard::compute_contrast(const Eigen::MatrixXd& Y, 
                             Eigen::MatrixXd& psi, 
                             Eigen::VectorXd& psiy) const
{
    int n_comp = Y.rows();
    int n_samples = Y.cols();
    
    psi = Eigen::MatrixXd::Zero(n_comp, n_samples);
    psiy = Eigen::VectorXd::Zero(n_comp);
    
    if (m_fun == "logcosh") {
        // psi(u) = tanh(alpha * u)
        // psi'(u) = alpha * (1 - tanh^2(alpha * u))
        for (int i = 0; i < n_comp; ++i) {
            double sum_psiy = 0.0;
            for (int j = 0; j < n_samples; ++j) {
                double u = Y(i, j);
                double tanh_au = std::tanh(m_alpha * u);
                psi(i, j) = tanh_au;
                sum_psiy += m_alpha * (1.0 - tanh_au * tanh_au);
            }
            psiy(i) = sum_psiy / static_cast<double>(n_samples);
        }
    } else if (m_fun == "exp") {
        // psi(u) = u * exp(-u^2/2)
        // psi'(u) = (1 - u^2) * exp(-u^2/2)
        for (int i = 0; i < n_comp; ++i) {
            double sum_psiy = 0.0;
            for (int j = 0; j < n_samples; ++j) {
                double u = Y(i, j);
                double exp_val = std::exp(-0.5 * u * u);
                psi(i, j) = u * exp_val;
                sum_psiy += (1.0 - u * u) * exp_val;
            }
            psiy(i) = sum_psiy / static_cast<double>(n_samples);
        }
    } else if (m_fun == "cube") {
        // psi(u) = u^3
        // psi'(u) = 3 * u^2
        for (int i = 0; i < n_comp; ++i) {
            double sum_psiy = 0.0;
            for (int j = 0; j < n_samples; ++j) {
                double u = Y(i, j);
                psi(i, j) = u * u * u;
                sum_psiy += 3.0 * u * u;
            }
            psiy(i) = sum_psiy / static_cast<double>(n_samples);
        }
    }
}

Eigen::MatrixXd Picard::orthogonalize(const Eigen::MatrixXd& W) const
{
    // Use QR decomposition for orthogonalization
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(W.transpose());
    Eigen::MatrixXd Q = qr.householderQ();
    
    // Take the first n_components columns and transpose back
    return Q.leftCols(W.rows()).transpose();
}

double Picard::line_search(const Eigen::MatrixXd& W, 
                          const Eigen::MatrixXd& direction,
                          const Eigen::MatrixXd& data) const
{
    // Simple backtracking line search
    double step_size = 1.0;
    double c1 = 1e-4; // Armijo condition parameter
    double rho = 0.5; // Step size reduction factor
    
    double current_obj = compute_objective(W, data);
    
    // Compute directional derivative
    Eigen::MatrixXd Y = W * data;
    Eigen::MatrixXd psi;
    Eigen::VectorXd psiy;
    compute_contrast(Y, psi, psiy);
    
    Eigen::MatrixXd gradient = (psi * data.transpose()) / static_cast<double>(data.cols()) -
                              psiy.asDiagonal() * W;
    
    double directional_deriv = (gradient.cwiseProduct(direction)).sum();
    
    // Backtracking line search
    for (int i = 0; i < 10; ++i) {
        Eigen::MatrixXd W_new = W + step_size * direction;
        if (m_ortho) {
            W_new = orthogonalize(W_new);
        }
        
        double new_obj = compute_objective(W_new, data);
        
        // Armijo condition
        if (new_obj <= current_obj + c1 * step_size * directional_deriv) {
            break;
        }
        
        step_size *= rho;
    }
    
    return step_size;
}

double Picard::compute_objective(const Eigen::MatrixXd& W, const Eigen::MatrixXd& data) const
{
    Eigen::MatrixXd Y = W * data;
    double objective = 0.0;
    int n_samples = data.cols();
    
    if (m_fun == "logcosh") {
        for (int i = 0; i < Y.rows(); ++i) {
            for (int j = 0; j < Y.cols(); ++j) {
                objective += std::log(std::cosh(m_alpha * Y(i, j)));
            }
        }
    } else if (m_fun == "exp") {
        for (int i = 0; i < Y.rows(); ++i) {
            for (int j = 0; j < Y.cols(); ++j) {
                objective -= std::exp(-0.5 * Y(i, j) * Y(i, j));
            }
        }
    } else if (m_fun == "cube") {
        for (int i = 0; i < Y.rows(); ++i) {
            for (int j = 0; j < Y.cols(); ++j) {
                double u = Y(i, j);
                objective += 0.25 * u * u * u * u;
            }
        }
    }
    
    return objective / static_cast<double>(n_samples);
}

} // NAMESPACE