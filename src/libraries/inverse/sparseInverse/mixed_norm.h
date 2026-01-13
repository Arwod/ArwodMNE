#ifndef MIXED_NORM_H
#define MIXED_NORM_H

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
 * Mixed norm inverse solver (L1/L2 regularization)
 * 
 * Implements mixed norm regularization for MEG/EEG source reconstruction.
 * Combines L1 norm (sparsity) and L2 norm (smoothness) regularization.
 */
class INVERSESHARED_EXPORT MixedNorm
{
public:
    typedef std::shared_ptr<MixedNorm> SPtr;            /**< Shared pointer type for MixedNorm. */
    typedef std::shared_ptr<const MixedNorm> ConstSPtr; /**< Const shared pointer type for MixedNorm. */

    /**
     * @brief Optimization parameters for mixed norm solver
     */
    struct OptimizationParams {
        double alpha;               /**< L1 regularization parameter (default: 0.1) */
        double l1_ratio;           /**< L1 ratio in elastic net (default: 0.5) */
        int max_iterations;        /**< Maximum iterations (default: 1000) */
        double tolerance;          /**< Convergence tolerance (default: 1e-6) */
        bool normalize;            /**< Normalize leadfield (default: true) */
        bool positive;             /**< Enforce positive sources (default: false) */
        double step_size;          /**< Step size for optimization (default: 1.0) */
        bool adaptive_step;        /**< Use adaptive step size (default: true) */
    };

    /**
     * @brief Time-frequency mixed norm parameters
     */
    struct TFParams {
        bool use_tf_mixed_norm;    /**< Use time-frequency mixed norm (default: false) */
        double tf_alpha;           /**< TF regularization parameter (default: 0.01) */
        int n_tfmxne_iter;         /**< TF-MxNE iterations (default: 10) */
        bool debias;               /**< Debias solution (default: true) */
        std::string solver;        /**< Solver type: "auto", "cd", "bcd" (default: "auto") */
    };

    //=========================================================================================================
    /**
     * Constructs a MixedNorm object.
     *
     * @param[in] leadfield     Forward operator/leadfield matrix.
     * @param[in] opt_params    Optimization parameters.
     * @param[in] tf_params     Time-frequency parameters.
     */
    explicit MixedNorm(const Eigen::MatrixXd& leadfield,
                       const OptimizationParams& opt_params = OptimizationParams(),
                       const TFParams& tf_params = TFParams());

    //=========================================================================================================
    /**
     * Destroys the MixedNorm object.
     */
    ~MixedNorm() = default;

    //=========================================================================================================
    /**
     * Solve the mixed norm inverse problem.
     * 
     * @param[in] data  Sensor data matrix (n_sensors x n_times).
     * @return Source estimates (n_sources x n_times).
     */
    Eigen::MatrixXd solve(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Solve with time-frequency mixed norm.
     * 
     * @param[in] data  Sensor data matrix (n_sensors x n_times).
     * @return Source estimates (n_sources x n_times).
     */
    Eigen::MatrixXd solveTF(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Get optimization parameters.
     */
    OptimizationParams getOptimizationParams() const { return m_optParams; }

    //=========================================================================================================
    /**
     * Set optimization parameters.
     */
    void setOptimizationParams(const OptimizationParams& params) { m_optParams = params; }

    //=========================================================================================================
    /**
     * Get time-frequency parameters.
     */
    TFParams getTFParams() const { return m_tfParams; }

    //=========================================================================================================
    /**
     * Set time-frequency parameters.
     */
    void setTFParams(const TFParams& params) { m_tfParams = params; }

    //=========================================================================================================
    /**
     * Get convergence information.
     */
    bool hasConverged() const { return m_bConverged; }
    int getIterations() const { return m_iIterations; }
    std::vector<double> getCostHistory() const { return m_vecCostHistory; }

    //=========================================================================================================
    /**
     * Get sparsity level of the solution.
     */
    double getSparsityLevel(double threshold = 1e-6) const;

private:
    //=========================================================================================================
    /**
     * Initialize default optimization parameters.
     */
    static OptimizationParams getDefaultOptimizationParams();

    //=========================================================================================================
    /**
     * Initialize default time-frequency parameters.
     */
    static TFParams getDefaultTFParams();

    //=========================================================================================================
    /**
     * Coordinate descent solver.
     */
    Eigen::MatrixXd solveCoordinateDescent(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Block coordinate descent solver.
     */
    Eigen::MatrixXd solveBlockCoordinateDescent(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Proximal gradient solver.
     */
    Eigen::MatrixXd solveProximalGradient(const Eigen::MatrixXd& data);

    //=========================================================================================================
    /**
     * Soft thresholding operator.
     */
    double softThreshold(double x, double threshold) const;

    //=========================================================================================================
    /**
     * Group soft thresholding operator.
     */
    Eigen::VectorXd groupSoftThreshold(const Eigen::VectorXd& x, double threshold) const;

    //=========================================================================================================
    /**
     * Compute elastic net penalty.
     */
    double computeElasticNetPenalty(const Eigen::MatrixXd& X) const;

    //=========================================================================================================
    /**
     * Compute cost function.
     */
    double computeCostFunction(const Eigen::MatrixXd& data, const Eigen::MatrixXd& X) const;

    //=========================================================================================================
    /**
     * Check convergence.
     */
    bool checkConvergence(const Eigen::MatrixXd& X_new, const Eigen::MatrixXd& X_old) const;

    //=========================================================================================================
    /**
     * Normalize leadfield matrix.
     */
    void normalizeLeadfield();

    //=========================================================================================================
    /**
     * Debias the solution.
     */
    Eigen::MatrixXd debiasolution(const Eigen::MatrixXd& X, const Eigen::MatrixXd& data) const;

    // Parameters
    OptimizationParams m_optParams;
    TFParams m_tfParams;

    // Data matrices
    Eigen::MatrixXd m_matLeadfield;         // Forward operator
    Eigen::MatrixXd m_matLeadfieldOrig;     // Original leadfield (before normalization)
    Eigen::VectorXd m_vecNormFactors;       // Normalization factors

    // Solution and convergence
    Eigen::MatrixXd m_matSolution;          // Current solution
    std::vector<double> m_vecCostHistory;   // Cost function history
    bool m_bConverged;                      // Convergence flag
    int m_iIterations;                      // Number of iterations

    // Dimensions
    int m_iNSensors;                        // Number of sensors
    int m_iNSources;                        // Number of sources
    int m_iNTimes;                          // Number of time points
};

//=============================================================================================================
/**
 * Convenience function for mixed norm reconstruction.
 * 
 * @param[in] leadfield     Forward operator matrix.
 * @param[in] data          Sensor data matrix.
 * @param[in] alpha         L1 regularization parameter.
 * @param[in] l1_ratio      L1 ratio in elastic net.
 * @param[in] opt_params    Optional optimization parameters.
 * @return Source estimates.
 */
INVERSESHARED_EXPORT Eigen::MatrixXd mixed_norm(const Eigen::MatrixXd& leadfield,
                                               const Eigen::MatrixXd& data,
                                               double alpha = 0.1,
                                               double l1_ratio = 0.5,
                                               const MixedNorm::OptimizationParams& opt_params = MixedNorm::OptimizationParams());

//=============================================================================================================
/**
 * Convenience function for time-frequency mixed norm reconstruction.
 * 
 * @param[in] leadfield     Forward operator matrix.
 * @param[in] data          Sensor data matrix.
 * @param[in] alpha         L1 regularization parameter.
 * @param[in] l1_ratio      L1 ratio in elastic net.
 * @param[in] tf_params     Time-frequency parameters.
 * @return Source estimates.
 */
INVERSESHARED_EXPORT Eigen::MatrixXd tf_mixed_norm(const Eigen::MatrixXd& leadfield,
                                                  const Eigen::MatrixXd& data,
                                                  double alpha = 0.01,
                                                  double l1_ratio = 0.5,
                                                  const MixedNorm::TFParams& tf_params = MixedNorm::TFParams());

} // namespace INVERSELIB

#endif // MIXED_NORM_H