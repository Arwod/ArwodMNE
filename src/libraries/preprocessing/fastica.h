#ifndef FASTICA_H
#define FASTICA_H

#include "preprocessing_global.h"
#include <Eigen/Core>
#include <string>

namespace PREPROCESSINGLIB {

class PREPROCESSINGSHARED_EXPORT FastICA
{
public:
    FastICA(int n_components = 0, 
            const std::string& algorithm = "parallel", 
            const std::string& fun = "logcosh", 
            int max_iter = 200, 
            double tol = 1e-4, 
            int random_state = 0);

    /**
     * Fit FastICA on data.
     * 
     * @param[in] data Input data (n_channels x n_times).
     *                 If not whitened, it should be whitened first? 
     *                 sklearn FastICA includes whitening.
     *                 But MNE separates them often?
     *                 MNE ICA.fit does: 
     *                   1. PCA/Whitening
     *                   2. FastICA on whitened data.
     *                 
     *                 Here, let's assume this FastICA class does the whole pipeline 
     *                 (Whitening + ICA) to be self-contained like sklearn.
     *                 Or we can provide a method to fit on whitened data.
     */
    void fit(const Eigen::MatrixXd& data);
    
    Eigen::MatrixXd transform(const Eigen::MatrixXd& data) const;
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& data) const;

    // Getters
    Eigen::MatrixXd get_components() const { return m_unmixing; } // Unmixing matrix (W) or Sources? 
                                                                    // sklearn components_ is (n_components, n_features) which is Mixing^-1?
                                                                    // Wait, sklearn transform returns S = X * W^T ?
                                                                    // sklearn: S = (X - mean) * K * W^T
                                                                    // components_ = W * K^T ?
                                                                    // Let's clarify terms.
                                                                    // X = A * S. S = W_unmix * X.
                                                                    // Here components usually refers to W_unmix?
                                                                    // In MNE: ica.unmixing_matrix_ is W. ica.pca_components_ is K.
    
    Eigen::MatrixXd get_mixing_matrix() const { return m_mixing; }  // A
    Eigen::MatrixXd get_unmixing_matrix() const { return m_unmixing; } // W_total = W_ica * K_white

private:
    // Core ICA step on whitened data
    Eigen::MatrixXd _ica_par(const Eigen::MatrixXd& X, int n_comp, const std::string& fun, double tol, int max_iter);
    Eigen::MatrixXd _sym_decorrelation(const Eigen::MatrixXd& W);
    
    int m_n_components;
    std::string m_algorithm; // "parallel" or "deflation"
    std::string m_fun;       // "logcosh", "exp", "cube"
    int m_max_iter;
    double m_tol;
    int m_random_state;
    
    Eigen::VectorXd m_mean;
    Eigen::MatrixXd m_whitening_matrix; // K
    Eigen::MatrixXd m_w_ica;            // W (from ICA loop)
    Eigen::MatrixXd m_unmixing;         // W_total = W * K
    Eigen::MatrixXd m_mixing;           // A = pseudoinverse(W_total)
};

} // NAMESPACE

#endif // FASTICA_H
