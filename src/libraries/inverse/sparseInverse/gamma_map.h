#ifndef GAMMA_MAP_H
#define GAMMA_MAP_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "../inverse_global.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <memory>

//=============================================================================================================
// DEFINE NAMESPACE INVERSELIB
//=============================================================================================================

namespace INVERSELIB
{

//=============================================================================================================
/**
 * Gamma-MAP (Maximum A Posteriori) sparse Bayesian reconstruction
 * 
 * Implements the Gamma-MAP algorithm for sparse source reconstruction in MEG/EEG.
 * This algorithm uses a hierarchical Bayesian model with Gamma priors on the source
 * variances to promote sparsity in the solution.
 */
class INVERSESHARED_EXPORT GammaMAP
{
public:
    typedef std::shared_ptr<GammaMAP> SPtr;            /**< Shared pointer type for GammaMAP. */
    typedef std::shared_ptr<const GammaMAP> ConstSPtr; /**< Const shared pointer type for GammaMAP. */

    /**
     * @brief Convergence criteria for the algorithm
     */
    struct ConvergenceCriteria {
        double tolerance;           /**< Convergence tolerance (default: 1e-6) */
        int max_iterations;         /**< Maximum number of iterations (default: 100) */
        bool check_cost_function;   /**< Whether to check cost function convergence (default: true) */
        double cost_tolerance;      /**< Cost function convergence tolerance (default: 1e-8) */
    };

    /**
     * @brief Hyperparameters for the Gamma-MAP algorithm
     */
    struct Hyperparameters {
        double alpha;               /**< Shape parameter for Gamma prior (default: 1.0) */
        double beta;                /**< Rate parameter for Gamma prior (default: 1.0) */
        double noise_variance;      /**< Noise variance estimate (default: 1.0) */
        bool update_noise;          /**< Whether to update noise variance (default: true) */
        bool use_ard;              /**< Use Automatic Relevance Determination (default: true) */
    };

    //=========================================================================================================
    /**
     * Constructs a GammaMAP object.
     *
     * @param[in] leadfield     Forward operator/leadfield matrix (n_sensors x n_sources).
     * @param[in] hyperparams   Hyperparameters for the algorithm.
     * @param[in] convergence   Convergence criteria.
     */
    explicit GammaMAP(const Eigen::MatrixXd& leadfield,
                      const Hyperparameters& hyperparams = Hyperparameters(),
                      const ConvergenceCriteria& convergence = ConvergenceCriteria());

    //=========================================================================================================
    /**
     * Destroys the GammaMAP object.
     */
    ~GammaMAP() = default;

    //=========================================================================================================
    /**
     * Fit the Gamma-MAP model to the data.
     * 
     * @param[in] data  Sensor data matrix (n_sensors x n_times).
     * @return true if successful.
     */
    bool fit(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Apply the fitted model to reconstruct sources.
     * 
     * @param[in] data  Sensor data matrix (n_sensors x n_times).
     * @return Source estimates (n_sources x n_times).
     */
    Eigen::MatrixXd apply(const Eigen::MatrixXd& data) const;

    //=========================================================================================================
    /**
     * Fit and apply in one step.
     * 
     * @param[in] data  Sensor data matrix (n_sensors x n_times).
     * @return Source estimates (n_sources x n_times).
     */
    Eigen::MatrixXd fitApply(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Get the estimated source variances (hyperparameters).
     */
    Eigen::VectorXd getSourceVariances() const { return m_vecSourceVariances; }

    //=========================================================================================================
    /**
     * Get the estimated noise variance.
     */
    double getNoiseVariance() const { return m_dNoiseVariance; }

    //=========================================================================================================
    /**
     * Get the cost function values during optimization.
     */
    std::vector<double> getCostHistory() const { return m_vecCostHistory; }

    //=========================================================================================================
    /**
     * Get convergence information.
     */
    bool hasConverged() const { return m_bConverged; }
    int getIterations() const { return m_iIterations; }

    //=========================================================================================================
    /**
     * Set new hyperparameters.
     */
    void setHyperparameters(const Hyperparameters& hyperparams) { m_hyperparams = hyperparams; }

    //=========================================================================================================
    /**
     * Set new convergence criteria.
     */
    void setConvergenceCriteria(const ConvergenceCriteria& convergence) { m_convergence = convergence; }

    //=========================================================================================================
    /**
     * Get sparsity level (percentage of near-zero sources).
     * 
     * @param[in] threshold  Threshold for considering a source as zero (default: 1e-6).
     * @return Sparsity level between 0 and 1.
     */
    double getSparsityLevel(double threshold = 1e-6) const;

private:
    //=========================================================================================================
    /**
     * Initialize default hyperparameters.
     */
    static Hyperparameters getDefaultHyperparameters();

    //=========================================================================================================
    /**
     * Initialize default convergence criteria.
     */
    static ConvergenceCriteria getDefaultConvergenceCriteria();

    //=========================================================================================================
    /**
     * Perform one iteration of the EM algorithm.
     * 
     * @param[in] data  Sensor data matrix.
     * @return Cost function value.
     */
    double performEMIteration(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * E-step: Update posterior mean and covariance.
     * 
     * @param[in] data  Sensor data matrix.
     */
    void eStep(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * M-step: Update hyperparameters.
     * 
     * @param[in] data  Sensor data matrix.
     */
    void mStep(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Compute the negative log-likelihood (cost function).
     * 
     * @param[in] data  Sensor data matrix.
     * @return Negative log-likelihood value.
     */
    double computeCostFunction(const Eigen::MatrixXd& data) const;

    //=========================================================================================================
    /**
     * Check convergence based on various criteria.
     * 
     * @param[in] cost_new  New cost function value.
     * @param[in] cost_old  Previous cost function value.
     * @return true if converged.
     */
    bool checkConvergence(double cost_new, double cost_old) const;

    //=========================================================================================================
    /**
     * Update the precision matrix and its inverse.
     */
    void updatePrecisionMatrix();

    //=========================================================================================================
    /**
     * Compute the posterior covariance matrix efficiently.
     */
    void computePosteriorCovariance();

    // Algorithm parameters
    Hyperparameters m_hyperparams;
    ConvergenceCriteria m_convergence;

    // Data matrices
    Eigen::MatrixXd m_matLeadfield;         // Forward operator (n_sensors x n_sources)
    Eigen::VectorXd m_vecSourceVariances;   // Source variances (hyperparameters)
    double m_dNoiseVariance;                // Noise variance

    // Posterior statistics
    Eigen::MatrixXd m_matPosteriorMean;     // Posterior mean (n_sources x n_times)
    Eigen::MatrixXd m_matPosteriorCov;      // Posterior covariance (n_sources x n_sources)

    // Precision matrices for efficient computation
    Eigen::MatrixXd m_matPrecision;         // Precision matrix (inverse covariance)
    Eigen::MatrixXd m_matPrecisionInv;      // Inverse of precision matrix

    // Convergence tracking
    std::vector<double> m_vecCostHistory;   // Cost function history
    bool m_bConverged;                      // Convergence flag
    int m_iIterations;                      // Number of iterations performed
    bool m_bFitted;                         // Whether the model has been fitted

    // Dimensions
    int m_iNSensors;                        // Number of sensors
    int m_iNSources;                        // Number of sources
    int m_iNTimes;                          // Number of time points
};

//=============================================================================================================
/**
 * Convenience function for Gamma-MAP reconstruction.
 * 
 * @param[in] leadfield     Forward operator matrix.
 * @param[in] data          Sensor data matrix.
 * @param[in] hyperparams   Optional hyperparameters.
 * @param[in] convergence   Optional convergence criteria.
 * @return Source estimates.
 */
INVERSESHARED_EXPORT Eigen::MatrixXd gamma_map(const Eigen::MatrixXd& leadfield,
                                              const Eigen::MatrixXd& data,
                                              const GammaMAP::Hyperparameters& hyperparams = GammaMAP::Hyperparameters(),
                                              const GammaMAP::ConvergenceCriteria& convergence = GammaMAP::ConvergenceCriteria());

} // namespace INVERSELIB

#endif // GAMMA_MAP_H