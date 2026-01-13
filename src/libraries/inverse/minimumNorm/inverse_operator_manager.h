//=============================================================================================================
/**
 * @file     inverse_operator_manager.h
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, Kiro AI Assistant. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief    Enhanced inverse operator creation and management
 *
 */

#ifndef INVERSE_OPERATOR_MANAGER_H
#define INVERSE_OPERATOR_MANAGER_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "../inverse_global.h"
#include <mne/mne_inverse_operator.h>
#include <mne/mne_forwardsolution.h>
#include <mne/mne_sourceestimate.h>
#include <fiff/fiff_cov.h>
#include <fiff/fiff_info.h>
#include <fiff/fiff_evoked.h>
#include <fiff/fiff_raw_data.h>

#include <QSharedPointer>
#include <QString>
#include <QMap>
#include <QList>
#include <QVector>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Dense>

//=============================================================================================================
// DEFINE NAMESPACE INVERSELIB
//=============================================================================================================

namespace INVERSELIB
{

//=============================================================================================================
// FORWARD DECLARATIONS
//=============================================================================================================

//=============================================================================================================
/**
 * Simple epochs data structure (since FiffEpochs doesn't exist in this codebase)
 */
struct INVERSESHARED_EXPORT EpochsData
{
    QList<Eigen::MatrixXd> epochs;      /**< List of epoch data matrices */
    FIFFLIB::FiffInfo info;             /**< Measurement info */
    Eigen::VectorXd times;              /**< Time vector */
    double tmin;                        /**< Start time */
    
    EpochsData() : tmin(0.0) {}
};

//=============================================================================================================
/**
 * Regularization parameters structure
 */
struct INVERSESHARED_EXPORT RegularizationParams
{
    double lambda;                  /**< Main regularization parameter */
    QString method;                 /**< Regularization method ("auto", "fixed", "adaptive") */
    double depth_weighting;         /**< Depth weighting exponent (default: 0.8) */
    bool limit_depth_chs;          /**< Limit depth weighting to specific channel types */
    double depth_limit;             /**< Maximum depth for weighting (default: 0.0 = no limit) */
    QString depth_method;           /**< Depth weighting method ("exp", "linear") */
    bool use_fwhm;                  /**< Use FWHM-based regularization */
    double fwhm;                    /**< FWHM for spatial smoothing */
    
    RegularizationParams() 
        : lambda(1.0/9.0), method("auto"), depth_weighting(0.8), 
          limit_depth_chs(false), depth_limit(0.0), depth_method("exp"),
          use_fwhm(false), fwhm(0.0) {}
};

//=============================================================================================================
/**
 * Orientation constraint parameters
 */
struct INVERSESHARED_EXPORT OrientationParams
{
    bool fixed;                     /**< Use fixed orientation constraint */
    bool loose;                     /**< Use loose orientation constraint */
    double loose_value;             /**< Loose orientation parameter (0-1) */
    QString constraint_method;      /**< Constraint method ("surface_normal", "free") */
    bool use_cps;                   /**< Use cortical patch statistics */
    
    OrientationParams()
        : fixed(false), loose(true), loose_value(0.2), 
          constraint_method("surface_normal"), use_cps(false) {}
};

//=============================================================================================================
/**
 * Enhanced inverse operator creation and management
 *
 * @brief The InverseOperatorManager class provides advanced methods for creating,
 * managing, and optimizing inverse operators with various regularization and
 * constraint options.
 */
class INVERSESHARED_EXPORT InverseOperatorManager
{

public:
    typedef QSharedPointer<InverseOperatorManager> SPtr;            /**< Shared pointer type for InverseOperatorManager. */
    typedef QSharedPointer<const InverseOperatorManager> ConstSPtr; /**< Const shared pointer type for InverseOperatorManager. */

    //=========================================================================================================
    /**
     * Constructs an InverseOperatorManager object.
     */
    explicit InverseOperatorManager();

    //=========================================================================================================
    /**
     * Destructor
     */
    ~InverseOperatorManager();

    //=========================================================================================================
    /**
     * Create an enhanced inverse operator with advanced regularization options
     *
     * @param[in] info              Measurement info
     * @param[in] forward           Forward solution
     * @param[in] noise_cov         Noise covariance matrix
     * @param[in] reg_params        Regularization parameters
     * @param[in] orient_params     Orientation constraint parameters
     * @param[in] rank              Rank of the data covariance (auto if empty)
     * @param[in] use_cps           Use cortical patch statistics
     * @param[in] verbose           Verbose output
     *
     * @return Enhanced inverse operator
     */
    static MNELIB::MNEInverseOperator make_inverse_operator(const FIFFLIB::FiffInfo& info,
                                                           const MNELIB::MNEForwardSolution& forward,
                                                           const FIFFLIB::FiffCov& noise_cov,
                                                           const RegularizationParams& reg_params = RegularizationParams(),
                                                           const OrientationParams& orient_params = OrientationParams(),
                                                           const QMap<QString, int>& rank = QMap<QString, int>(),
                                                           bool use_cps = false,
                                                           bool verbose = true);

    //=========================================================================================================
    /**
     * Prepare inverse operator for computation with optimized settings
     *
     * @param[in,out] inverse_operator  Inverse operator to prepare
     * @param[in] nave                  Number of averages
     * @param[in] lambda                Regularization parameter (override)
     * @param[in] method                Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori              Orientation picking ("normal", "vector", "max-power")
     * @param[in] use_cps               Use cortical patch statistics
     * @param[in] verbose               Verbose output
     *
     * @return Prepared inverse operator
     */
    static MNELIB::MNEInverseOperator prepare_inverse_operator(MNELIB::MNEInverseOperator& inverse_operator,
                                                              int nave,
                                                              double lambda = -1.0,
                                                              const QString& method = "dSPM",
                                                              const QString& pick_ori = "normal",
                                                              bool use_cps = true,
                                                              bool verbose = true);

    //=========================================================================================================
    /**
     * Compute optimal regularization parameter using various methods
     *
     * @param[in] forward           Forward solution
     * @param[in] noise_cov         Noise covariance matrix
     * @param[in] data_cov          Data covariance matrix (optional)
     * @param[in] method            Method ("auto", "cross_validation", "lcurve", "gcv")
     * @param[in] alpha_range       Range of alpha values to test
     * @param[in] n_jobs            Number of parallel jobs
     *
     * @return Optimal regularization parameter
     */
    static double compute_regularization(const MNELIB::MNEForwardSolution& forward,
                                       const FIFFLIB::FiffCov& noise_cov,
                                       const FIFFLIB::FiffCov& data_cov = FIFFLIB::FiffCov(),
                                       const QString& method = "auto",
                                       const Eigen::VectorXd& alpha_range = Eigen::VectorXd(),
                                       int n_jobs = 1);

    //=========================================================================================================
    /**
     * Apply depth weighting to forward solution
     *
     * @param[in,out] forward       Forward solution to weight
     * @param[in] depth_weighting   Depth weighting exponent
     * @param[in] depth_limit       Maximum depth for weighting
     * @param[in] method            Weighting method ("exp", "linear")
     * @param[in] limit_depth_chs   Limit to specific channel types
     *
     * @return Success status
     */
    static bool apply_depth_weighting(MNELIB::MNEForwardSolution& forward,
                                     double depth_weighting = 0.8,
                                     double depth_limit = 0.0,
                                     const QString& method = "exp",
                                     bool limit_depth_chs = false);

    //=========================================================================================================
    /**
     * Apply orientation constraints to forward solution
     *
     * @param[in,out] forward       Forward solution to constrain
     * @param[in] orient_params     Orientation parameters
     * @param[in] use_cps           Use cortical patch statistics
     *
     * @return Success status
     */
    static bool apply_orientation_constraint(MNELIB::MNEForwardSolution& forward,
                                           const OrientationParams& orient_params,
                                           bool use_cps = false);

    //=========================================================================================================
    /**
     * Estimate SNR from inverse operator
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] evoked            Evoked data (optional)
     *
     * @return SNR estimate
     */
    static double estimate_snr(const MNELIB::MNEInverseOperator& inverse_operator,
                              const FIFFLIB::FiffEvoked& evoked = FIFFLIB::FiffEvoked());

    //=========================================================================================================
    /**
     * Compute rank of covariance matrix
     *
     * @param[in] cov               Covariance matrix
     * @param[in] info              Measurement info
     * @param[in] ch_type           Channel type to analyze
     * @param[in] tol               Tolerance for rank computation
     *
     * @return Rank of the covariance matrix
     */
    static int compute_rank_inverse(const FIFFLIB::FiffCov& cov,
                                   const FIFFLIB::FiffInfo& info,
                                   const QString& ch_type = "all",
                                   double tol = 1e-4);

    //=========================================================================================================
    /**
     * Read inverse operator from file
     *
     * @param[in] fname             Filename
     * @param[in] verbose           Verbose output
     *
     * @return Inverse operator
     */
    static MNELIB::MNEInverseOperator read_inverse_operator(const QString& fname,
                                                           bool verbose = true);

    //=========================================================================================================
    /**
     * Write inverse operator to file
     *
     * @param[in] fname             Filename
     * @param[in] inverse_operator  Inverse operator to write
     * @param[in] verbose           Verbose output
     *
     * @return Success status
     */
    static bool write_inverse_operator(const QString& fname,
                                      const MNELIB::MNEInverseOperator& inverse_operator,
                                      bool verbose = true);

    //=========================================================================================================
    /**
     * Apply inverse operator to evoked data
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] evoked            Evoked data
     * @param[in] lambda            Regularization parameter (override)
     * @param[in] method            Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori          Orientation picking ("normal", "vector", "max-power")
     * @param[in] nave              Number of averages
     * @param[in] verbose           Verbose output
     *
     * @return Source estimate
     */
    static MNELIB::MNESourceEstimate apply_inverse(const MNELIB::MNEInverseOperator& inverse_operator,
                                                  const FIFFLIB::FiffEvoked& evoked,
                                                  double lambda = -1.0,
                                                  const QString& method = "dSPM",
                                                  const QString& pick_ori = "normal",
                                                  int nave = 1,
                                                  bool verbose = true);

    //=========================================================================================================
    /**
     * Apply inverse operator to covariance matrix
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] cov               Covariance matrix
     * @param[in] lambda            Regularization parameter (override)
     * @param[in] method            Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori          Orientation picking ("normal", "vector", "max-power")
     * @param[in] nave              Number of averages
     * @param[in] verbose           Verbose output
     *
     * @return Source covariance estimate
     */
    static MNELIB::MNESourceEstimate apply_inverse_cov(const MNELIB::MNEInverseOperator& inverse_operator,
                                                      const FIFFLIB::FiffCov& cov,
                                                      double lambda = -1.0,
                                                      const QString& method = "dSPM",
                                                      const QString& pick_ori = "normal",
                                                      int nave = 1,
                                                      bool verbose = true);

    //=========================================================================================================
    /**
     * Apply inverse operator to epochs data
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] epochs            Epochs data
     * @param[in] lambda            Regularization parameter (override)
     * @param[in] method            Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori          Orientation picking ("normal", "vector", "max-power")
     * @param[in] nave              Number of averages
     * @param[in] verbose           Verbose output
     *
     * @return List of source estimates for each epoch
     */
    static QList<MNELIB::MNESourceEstimate> apply_inverse_epochs(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                const EpochsData& epochs,
                                                                double lambda = -1.0,
                                                                const QString& method = "dSPM",
                                                                const QString& pick_ori = "normal",
                                                                int nave = 1,
                                                                bool verbose = true);

    //=========================================================================================================
    /**
     * Apply inverse operator to raw data
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] raw               Raw data
     * @param[in] start             Start sample
     * @param[in] stop              Stop sample
     * @param[in] lambda            Regularization parameter (override)
     * @param[in] method            Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori          Orientation picking ("normal", "vector", "max-power")
     * @param[in] nave              Number of averages
     * @param[in] verbose           Verbose output
     *
     * @return Source estimate
     */
    static MNELIB::MNESourceEstimate apply_inverse_raw(const MNELIB::MNEInverseOperator& inverse_operator,
                                                      const FIFFLIB::FiffRawData& raw,
                                                      int start = 0,
                                                      int stop = -1,
                                                      double lambda = -1.0,
                                                      const QString& method = "dSPM",
                                                      const QString& pick_ori = "normal",
                                                      int nave = 1,
                                                      bool verbose = true);

    //=========================================================================================================
    /**
     * Apply inverse operator to time-frequency epochs data
     *
     * @param[in] inverse_operator  Inverse operator
     * @param[in] epochs_tfr        Time-frequency epochs data
     * @param[in] lambda            Regularization parameter (override)
     * @param[in] method            Method ("MNE", "dSPM", "sLORETA", "eLORETA")
     * @param[in] pick_ori          Orientation picking ("normal", "vector", "max-power")
     * @param[in] nave              Number of averages
     * @param[in] verbose           Verbose output
     *
     * @return List of source estimates for each epoch and frequency
     */
    static QList<QList<MNELIB::MNESourceEstimate>> apply_inverse_tfr_epochs(const MNELIB::MNEInverseOperator& inverse_operator,
                                                                           const QList<QList<Eigen::MatrixXcd>>& epochs_tfr,
                                                                           const QVector<double>& freqs,
                                                                           double lambda = -1.0,
                                                                           const QString& method = "dSPM",
                                                                           const QString& pick_ori = "normal",
                                                                           int nave = 1,
                                                                           bool verbose = true);

private:
    //=========================================================================================================
    /**
     * Compute depth weights for source locations
     *
     * @param[in] source_rr         Source locations
     * @param[in] exp               Depth weighting exponent
     * @param[in] limit             Depth limit
     * @param[in] method            Weighting method
     *
     * @return Depth weights
     */
    static Eigen::VectorXd compute_depth_weights(const Eigen::MatrixXd& source_rr,
                                                double exp = 0.8,
                                                double limit = 0.0,
                                                const QString& method = "exp");

    //=========================================================================================================
    /**
     * Apply loose orientation constraint
     *
     * @param[in,out] gain_matrix   Gain matrix to modify
     * @param[in] loose_value       Loose constraint parameter
     * @param[in] source_ori        Source orientations
     *
     * @return Success status
     */
    static bool apply_loose_constraint(Eigen::MatrixXd& gain_matrix,
                                      double loose_value,
                                      const Eigen::MatrixXd& source_ori);

    //=========================================================================================================
    /**
     * Compute cross-validation score for regularization
     *
     * @param[in] forward           Forward solution
     * @param[in] noise_cov         Noise covariance
     * @param[in] data_cov          Data covariance
     * @param[in] alpha             Regularization parameter
     *
     * @return Cross-validation score
     */
    static double compute_cv_score(const MNELIB::MNEForwardSolution& forward,
                                  const FIFFLIB::FiffCov& noise_cov,
                                  const FIFFLIB::FiffCov& data_cov,
                                  double alpha);

    //=========================================================================================================
    /**
     * Compute L-curve for regularization parameter selection
     *
     * @param[in] forward           Forward solution
     * @param[in] noise_cov         Noise covariance
     * @param[in] alpha_range       Range of alpha values
     *
     * @return Optimal alpha from L-curve
     */
    static double compute_lcurve_alpha(const MNELIB::MNEForwardSolution& forward,
                                      const FIFFLIB::FiffCov& noise_cov,
                                      const Eigen::VectorXd& alpha_range);
};

//=============================================================================================================
// INLINE DEFINITIONS
//=============================================================================================================

} // NAMESPACE INVERSELIB

#endif // INVERSE_OPERATOR_MANAGER_H