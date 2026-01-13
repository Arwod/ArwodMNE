//=============================================================================================================
/**
 * @file     fwd_leadfield_enhanced.h
 * @author   Kiro AI Assistant
 * @since    0.1.0
 * @date     January, 2025
 *
 * @section  LICENSE
 *
 * Copyright (C) 2025, MNE-CPP authors. All rights reserved.
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
 * @brief    Enhanced leadfield matrix computation for forward modeling.
 *
 */

#ifndef FWD_LEADFIELD_ENHANCED_H
#define FWD_LEADFIELD_ENHANCED_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "fwd_global.h"
#include "fwd_bem_model.h"
#include "fwd_coil_set.h"
#include "fwd_eeg_sphere_model.h"

#include <mne/mne_sourcespace.h>
#include <fiff/fiff_coord_trans.h>
#include <fiff/fiff_info.h>

#include <Eigen/Core>
#include <QSharedPointer>
#include <QStringList>

//=============================================================================================================
// DEFINE NAMESPACE FWDLIB
//=============================================================================================================

namespace FWDLIB
{

//=============================================================================================================
/**
 * Enhanced leadfield matrix computation
 *
 * @brief Enhanced leadfield matrix computation with optimizations
 */
class FWDSHARED_EXPORT FwdLeadfieldEnhanced
{
public:
    typedef QSharedPointer<FwdLeadfieldEnhanced> SPtr;            /**< Shared pointer type for FwdLeadfieldEnhanced. */
    typedef QSharedPointer<const FwdLeadfieldEnhanced> ConstSPtr; /**< Const shared pointer type for FwdLeadfieldEnhanced. */

    //=========================================================================================================
    /**
     * Enhanced dipole modeling types
     */
    enum DipoleModelType {
        SINGLE_DIPOLE,          /**< Single current dipole */
        CURRENT_MULTIPOLE,      /**< Current multipole expansion */
        MAGNETIC_DIPOLE,        /**< Magnetic dipole */
        EQUIVALENT_CURRENT      /**< Equivalent current density */
    };

    //=========================================================================================================
    /**
     * Head model types for enhanced computation
     */
    enum HeadModelType {
        SPHERE_MODEL,           /**< Spherical head model */
        BEM_MODEL,              /**< Boundary element method model */
        FEM_MODEL,              /**< Finite element method model */
        REALISTIC_MODEL         /**< Realistic head model */
    };

    //=========================================================================================================
    /**
     * Optimization methods for leadfield computation
     */
    enum OptimizationMethod {
        STANDARD,               /**< Standard computation */
        PARALLEL,               /**< Parallel computation */
        BLOCK_WISE,             /**< Block-wise computation */
        ADAPTIVE,               /**< Adaptive computation based on geometry */
        GPU_ACCELERATED         /**< GPU-accelerated computation */
    };

    //=========================================================================================================
    /**
     * Default Constructor
     */
    FwdLeadfieldEnhanced();

    //=========================================================================================================
    /**
     * Destructor
     */
    ~FwdLeadfieldEnhanced();

    //=========================================================================================================
    /**
     * Compute enhanced leadfield matrix with optimizations
     *
     * @param[in] src               The source space.
     * @param[in] info              The measurement info.
     * @param[in] trans             The coordinate transformation.
     * @param[in] bemModel          The BEM model (optional).
     * @param[in] eegModel          The EEG sphere model (optional).
     * @param[in] megCoils          The MEG coil definitions (optional).
     * @param[in] eegElectrodes     The EEG electrode definitions (optional).
     * @param[in] dipoleType        The dipole model type.
     * @param[in] headModelType     The head model type.
     * @param[in] optimization      The optimization method.
     * @param[in] nJobs             Number of parallel jobs (default: 1).
     *
     * @return The computed leadfield matrix.
     */
    static Eigen::MatrixXd compute_leadfield_enhanced(const MNELIB::MNESourceSpace& src,
                                                    const FIFFLIB::FiffInfo& info,
                                                    const FIFFLIB::FiffCoordTrans& trans,
                                                    const FwdBemModel* bemModel = nullptr,
                                                    const FwdEegSphereModel* eegModel = nullptr,
                                                    const FwdCoilSet* megCoils = nullptr,
                                                    const FwdCoilSet* eegElectrodes = nullptr,
                                                    DipoleModelType dipoleType = SINGLE_DIPOLE,
                                                    HeadModelType headModelType = BEM_MODEL,
                                                    OptimizationMethod optimization = PARALLEL,
                                                    int nJobs = 1);

    //=========================================================================================================
    /**
     * Compute leadfield for single dipole with enhanced accuracy
     *
     * @param[in] dipolePos         The dipole position.
     * @param[in] dipoleOri         The dipole orientation.
     * @param[in] sensorPos         The sensor positions.
     * @param[in] sensorOri         The sensor orientations.
     * @param[in] bemModel          The BEM model.
     * @param[in] headModelType     The head model type.
     *
     * @return The leadfield vector for the dipole.
     */
    static Eigen::VectorXd compute_single_dipole_leadfield(const Eigen::Vector3f& dipolePos,
                                                         const Eigen::Vector3f& dipoleOri,
                                                         const Eigen::MatrixX3f& sensorPos,
                                                         const Eigen::MatrixX3f& sensorOri,
                                                         const FwdBemModel* bemModel,
                                                         HeadModelType headModelType = BEM_MODEL);

    //=========================================================================================================
    /**
     * Compute leadfield matrix in parallel blocks
     *
     * @param[in] sourcePositions   The source positions.
     * @param[in] sourceOrientations The source orientations.
     * @param[in] sensorPositions   The sensor positions.
     * @param[in] sensorOrientations The sensor orientations.
     * @param[in] bemModel          The BEM model.
     * @param[in] blockSize         The block size for parallel computation.
     * @param[in] nJobs             Number of parallel jobs.
     *
     * @return The computed leadfield matrix.
     */
    static Eigen::MatrixXd compute_leadfield_parallel(const Eigen::MatrixX3f& sourcePositions,
                                                    const Eigen::MatrixX3f& sourceOrientations,
                                                    const Eigen::MatrixX3f& sensorPositions,
                                                    const Eigen::MatrixX3f& sensorOrientations,
                                                    const FwdBemModel* bemModel,
                                                    int blockSize = 1000,
                                                    int nJobs = 4);

    //=========================================================================================================
    /**
     * Optimize leadfield computation based on geometry
     *
     * @param[in] sourcePositions   The source positions.
     * @param[in] sensorPositions   The sensor positions.
     * @param[in] bemModel          The BEM model.
     *
     * @return The optimal computation strategy.
     */
    static OptimizationMethod determine_optimal_strategy(const Eigen::MatrixX3f& sourcePositions,
                                                       const Eigen::MatrixX3f& sensorPositions,
                                                       const FwdBemModel* bemModel);

    //=========================================================================================================
    /**
     * Compute multipole expansion for enhanced accuracy
     *
     * @param[in] dipolePos         The dipole position.
     * @param[in] sensorPos         The sensor position.
     * @param[in] order             The multipole expansion order.
     * @param[in] bemModel          The BEM model.
     *
     * @return The multipole expansion coefficients.
     */
    static Eigen::VectorXd compute_multipole_expansion(const Eigen::Vector3f& dipolePos,
                                                     const Eigen::Vector3f& sensorPos,
                                                     int order,
                                                     const FwdBemModel* bemModel);

    //=========================================================================================================
    /**
     * Apply adaptive mesh refinement for improved accuracy
     *
     * @param[in] bemModel          The BEM model.
     * @param[in] sourcePositions   The source positions.
     * @param[in] tolerance         The refinement tolerance.
     *
     * @return The refined BEM model.
     */
    static FwdBemModel apply_adaptive_refinement(const FwdBemModel& bemModel,
                                                const Eigen::MatrixX3f& sourcePositions,
                                                double tolerance = 1e-6);

    //=========================================================================================================
    /**
     * Validate leadfield matrix quality
     *
     * @param[in] leadfield         The leadfield matrix.
     * @param[in] sourcePositions   The source positions.
     * @param[in] sensorPositions   The sensor positions.
     *
     * @return Quality metrics (condition number, rank, etc.).
     */
    static QMap<QString, double> validate_leadfield_quality(const Eigen::MatrixXd& leadfield,
                                                          const Eigen::MatrixX3f& sourcePositions,
                                                          const Eigen::MatrixX3f& sensorPositions);

private:
    //=========================================================================================================
    /**
     * Helper function to compute BEM leadfield contribution
     *
     * @param[in] dipolePos         The dipole position.
     * @param[in] dipoleOri         The dipole orientation.
     * @param[in] sensorPos         The sensor position.
     * @param[in] sensorOri         The sensor orientation.
     * @param[in] bemModel          The BEM model.
     *
     * @return The BEM contribution to leadfield.
     */
    static double compute_bem_contribution(const Eigen::Vector3f& dipolePos,
                                         const Eigen::Vector3f& dipoleOri,
                                         const Eigen::Vector3f& sensorPos,
                                         const Eigen::Vector3f& sensorOri,
                                         const FwdBemModel* bemModel);

    //=========================================================================================================
    /**
     * Helper function to compute sphere model leadfield
     *
     * @param[in] dipolePos         The dipole position.
     * @param[in] dipoleOri         The dipole orientation.
     * @param[in] sensorPos         The sensor position.
     * @param[in] sensorOri         The sensor orientation.
     * @param[in] sphereModel       The sphere model parameters.
     *
     * @return The sphere model leadfield.
     */
    static double compute_sphere_leadfield(const Eigen::Vector3f& dipolePos,
                                         const Eigen::Vector3f& dipoleOri,
                                         const Eigen::Vector3f& sensorPos,
                                         const Eigen::Vector3f& sensorOri,
                                         const FwdEegSphereModel* sphereModel);

    //=========================================================================================================
    /**
     * Helper function for parallel block computation
     *
     * @param[in] blockStart        The block start index.
     * @param[in] blockEnd          The block end index.
     * @param[in] sourcePositions   The source positions.
     * @param[in] sourceOrientations The source orientations.
     * @param[in] sensorPositions   The sensor positions.
     * @param[in] sensorOrientations The sensor orientations.
     * @param[in] bemModel          The BEM model.
     *
     * @return The computed leadfield block.
     */
    static Eigen::MatrixXd compute_leadfield_block(int blockStart,
                                                  int blockEnd,
                                                  const Eigen::MatrixX3f& sourcePositions,
                                                  const Eigen::MatrixX3f& sourceOrientations,
                                                  const Eigen::MatrixX3f& sensorPositions,
                                                  const Eigen::MatrixX3f& sensorOrientations,
                                                  const FwdBemModel* bemModel);
};

} // NAMESPACE

#endif // FWD_LEADFIELD_ENHANCED_H