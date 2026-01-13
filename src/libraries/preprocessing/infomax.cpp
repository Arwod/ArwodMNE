#include "infomax.h"
#include <cmath>
#include <iostream>
#include <random>
#include <Eigen/Dense>

namespace PREPROCESSINGLIB {

Infomax::Infomax(int n_components,
                 bool extended,
                 double learning_rate,
                 int max_iter,
                 double tol,
                 int random_state)
    : m_n_components(n_components)
    , m_extended(extended)
    , m_learning_rate(learning_rate)
    , m_max_iter(max_iter)
    , m_tol(tol)
    , m_random_state(random_state)
    , m_converged(false)
    , m_n_iter(0)
{
}

void Infomax::fit(const Eigen::MatrixXd& data)
{
    int n_comp = data.rows();
    int n_samples = data.cols();
    
    if (m_n_components == 0) {
        m_n_components = n_comp;
    }
    
    // Initialize unmixing matrix W randomly
    std::mt19937 gen(m_random_state);
    std::normal_distribution<> d(0, 1);
    
    m_W = Eigen::MatrixXd(m_n_components, n_comp);
    for (int i = 0; i < m_n_components; ++i) {
        for (int j = 0; j < n_comp; ++j) {
            m_W(i, j) = d(gen) * 0.1; // Small random initialization
        }
    }
    
    // Initialize signs for extended Infomax
    Eigen::VectorXd signs = Eigen::VectorXd::Ones(m_n_components);
    
    // Infomax learning loop
    double prev_cost = std::numeric_limits<double>::max();
    
    for (m_n_iter = 0; m_n_iter < m_max_iter; ++m_n_iter) {
        // Forward pass: compute sources
        Eigen::MatrixXd sources = m_W * data;
        
        // Compute activation and its derivative
        Eigen::MatrixXd activation, activation_deriv;
        
        if (m_extended) {
            // Update signs based on kurtosis every 10 iterations
            if (m_n_iter % 10 == 0) {
                signs = update_signs(sources);
            }
            
            activation = extended_activation(sources, signs);
            
            // Derivative for extended Infomax
            activation_deriv = Eigen::MatrixXd::Zero(m_n_components, n_samples);
            for (int i = 0; i < m_n_components; ++i) {
                for (int j = 0; j < n_samples; ++j) {
                    double u = sources(i, j);
                    if (signs(i) > 0) {
                        // Super-Gaussian: tanh derivative
                        double tanh_u = std::tanh(u);
                        activation_deriv(i, j) = 1.0 - tanh_u * tanh_u;
                    } else {
                        // Sub-Gaussian: u - tanh(u) derivative  
                        double tanh_u = std::tanh(u);
                        activation_deriv(i, j) = 1.0 - (1.0 - tanh_u * tanh_u);
                    }
                }
            }
        } else {
            // Standard Infomax with sigmoid
            activation = sigmoid(sources);
            
            // Sigmoid derivative: sigmoid(u) * (1 - sigmoid(u))
            activation_deriv = activation.cwiseProduct((1.0 - activation.array()).matrix());
        }
        
        // Compute gradient
        // dW = learning_rate * (I + activation_deriv * sources^T / n_samples) * W
        Eigen::MatrixXd gradient_term = Eigen::MatrixXd::Identity(m_n_components, m_n_components) +
            (activation_deriv * sources.transpose()) / static_cast<double>(n_samples);
        
        Eigen::MatrixXd dW = m_learning_rate * gradient_term * m_W;
        
        // Update W
        Eigen::MatrixXd W_new = m_W + dW;
        
        // Check convergence
        double cost = (W_new - m_W).norm();
        
        if (cost < m_tol) {
            m_converged = true;
            std::cout << "Infomax converged after " << m_n_iter + 1 << " iterations." << std::endl;
            break;
        }
        
        // Adaptive learning rate
        if (cost > prev_cost) {
            m_learning_rate *= 0.9; // Reduce learning rate if cost increased
        } else if (m_n_iter > 10 && cost < prev_cost * 0.99) {
            m_learning_rate *= 1.01; // Slightly increase if making good progress
        }
        
        m_W = W_new;
        prev_cost = cost;
        
        // Print progress every 50 iterations
        if ((m_n_iter + 1) % 50 == 0) {
            std::cout << "Infomax iteration " << m_n_iter + 1 
                      << ", cost: " << cost 
                      << ", lr: " << m_learning_rate << std::endl;
        }
    }
    
    if (!m_converged) {
        std::cout << "Infomax did not converge after " << m_max_iter << " iterations." << std::endl;
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

Eigen::MatrixXd Infomax::transform(const Eigen::MatrixXd& data) const
{
    return m_W * data;
}

Eigen::MatrixXd Infomax::inverse_transform(const Eigen::MatrixXd& sources) const
{
    return m_A * sources;
}

Eigen::MatrixXd Infomax::sigmoid(const Eigen::MatrixXd& x) const
{
    Eigen::MatrixXd result(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            // Numerically stable sigmoid
            double val = x(i, j);
            if (val > 0) {
                double exp_neg_val = std::exp(-val);
                result(i, j) = 1.0 / (1.0 + exp_neg_val);
            } else {
                double exp_val = std::exp(val);
                result(i, j) = exp_val / (1.0 + exp_val);
            }
        }
    }
    
    return result;
}

Eigen::MatrixXd Infomax::extended_activation(const Eigen::MatrixXd& x, const Eigen::VectorXd& signs) const
{
    Eigen::MatrixXd result(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            double u = x(i, j);
            if (signs(i) > 0) {
                // Super-Gaussian: tanh
                result(i, j) = std::tanh(u);
            } else {
                // Sub-Gaussian: u - tanh(u)
                result(i, j) = u - std::tanh(u);
            }
        }
    }
    
    return result;
}

Eigen::VectorXd Infomax::update_signs(const Eigen::MatrixXd& sources) const
{
    Eigen::VectorXd kurtosis = compute_kurtosis(sources);
    Eigen::VectorXd signs(sources.rows());
    
    for (int i = 0; i < sources.rows(); ++i) {
        // Positive kurtosis (> 0) indicates super-Gaussian (use tanh)
        // Negative kurtosis (< 0) indicates sub-Gaussian (use u - tanh(u))
        signs(i) = (kurtosis(i) > 0) ? 1.0 : -1.0;
    }
    
    return signs;
}

Eigen::VectorXd Infomax::compute_kurtosis(const Eigen::MatrixXd& sources) const
{
    Eigen::VectorXd kurtosis(sources.rows());
    
    for (int i = 0; i < sources.rows(); ++i) {
        Eigen::VectorXd component = sources.row(i);
        
        // Standardize the component
        double mean = component.mean();
        component = component.array() - mean;
        double std_dev = std::sqrt(component.array().square().mean());
        
        if (std_dev > 1e-10) {
            component /= std_dev;
            
            // Compute kurtosis: E[x^4] - 3 (excess kurtosis)
            double fourth_moment = component.array().pow(4).mean();
            kurtosis(i) = fourth_moment - 3.0;
        } else {
            kurtosis(i) = 0.0;
        }
    }
    
    return kurtosis;
}

} // NAMESPACE