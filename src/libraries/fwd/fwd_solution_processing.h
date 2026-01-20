//=============================================================================================================
/**
 * @file     fwd_solution_processing.h
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
 * @brief    Forward solution processing functions declaration.
 *
 */

#ifndef FWD_SOLUTION_PROCESSING_H
#define FWD_SOLUTION_PROCESSING_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "fwd_global.h"

#include <mne/mne_forwardsolution.h>
#include <mne/mne_sourceestimate.h>
#include <fiff/fiff_evoked.h>
#include <fiff/fiff_raw_data.h>
#include <fs/label.h>

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
 * Forward solution processing utilities
 *
 * @brief Forward solution processing utilities
 */
class FWDSHARED_EXPORT FwdSolutionProcessing
{
public:
    typedef QSharedPointer<FwdSolutionProcessing> SPtr;            /**< Shared pointer type for FwdSolutionProcessing. */
    typedef QSharedPointer<const FwdSolutionProcessing> ConstSPtr; /**< Const shared pointer type for FwdSolutionProcessing. */

    //=========================================================================================================
    /**
     * Apply forward solution to source estimate data
     *
     * @param[in] fwd           The forward solution.
     * @param[in] stc           The source estimate to apply forward solution to.
     * @param[in] info          The measurement info (optional, uses fwd.info if not provided).
     * @param[in] use_cps       Whether to use cortical patch statistics (default: true).
     *
     * @return The simulated sensor data as Evoked object.
     */
    static FIFFLIB::FiffEvoked apply_forward(const MNELIB::MNEForwardSolution& fwd,
                                           const MNELIB::MNESourceEstimate& stc,
                                           const FIFFLIB::FiffInfo& info = FIFFLIB::FiffInfo(),
                                           bool use_cps = true);

    //=========================================================================================================
    /**
     * Apply forward solution to raw source estimate data
     *
     * @param[in] fwd           The forward solution.
     * @param[in] stc           The source estimate to apply forward solution to.
     * @param[in] info          The measurement info (optional, uses fwd.info if not provided).
     * @param[in] start         Start sample (default: 0).
     * @param[in] stop          Stop sample (default: -1 for all).
     *
     * @return The simulated sensor data as Raw object.
     */
    static FIFFLIB::FiffRawData apply_forward_raw(const MNELIB::MNEForwardSolution& fwd,
                                                const MNELIB::MNESourceEstimate& stc,
                                                const FIFFLIB::FiffInfo& info = FIFFLIB::FiffInfo(),
                                                qint32 start = 0,
                                                qint32 stop = -1);

    //=========================================================================================================
    /**
     * Convert forward solution between different coordinate frames and orientations
     *
     * @param[in] fwd           The forward solution to convert.
     * @param[in] surf_ori      Whether to use surface-based source orientations (default: false).
     * @param[in] force_fixed   Whether to force fixed source orientations (default: false).
     * @param[in] copy          Whether to copy the forward solution (default: true).
     * @param[in] use_cps       Whether to use cortical patch statistics (default: true).
     * @param[in] verbose       Verbosity level (default: false).
     *
     * @return The converted forward solution.
     */
    static MNELIB::MNEForwardSolution convert_forward_solution(const MNELIB::MNEForwardSolution& fwd,
                                                             bool surf_ori = false,
                                                             bool force_fixed = false,
                                                             bool copy = true,
                                                             bool use_cps = true,
                                                             bool verbose = false);

    //=========================================================================================================
    /**
     * Restrict forward solution to a specific label
     *
     * @param[in] fwd           The forward solution to restrict.
     * @param[in] label         The label to restrict to.
     * @param[in] copy          Whether to copy the forward solution (default: true).
     *
     * @return The restricted forward solution.
     */
    static MNELIB::MNEForwardSolution restrict_forward_to_label(const MNELIB::MNEForwardSolution& fwd,
                                                              const FSLIB::Label& label,
                                                              bool copy = true);

    //=========================================================================================================
    /**
     * Restrict forward solution to multiple labels
     *
     * @param[in] fwd           The forward solution to restrict.
     * @param[in] labels        The labels to restrict to.
     * @param[in] copy          Whether to copy the forward solution (default: true).
     *
     * @return The restricted forward solution.
     */
    static MNELIB::MNEForwardSolution restrict_forward_to_labels(const MNELIB::MNEForwardSolution& fwd,
                                                               const QList<FSLIB::Label>& labels,
                                                               bool copy = true);

    //=========================================================================================================
    /**
     * Restrict forward solution to source estimate
     *
     * @param[in] fwd           The forward solution to restrict.
     * @param[in] stc           The source estimate defining the restriction.
     * @param[in] copy          Whether to copy the forward solution (default: true).
     *
     * @return The restricted forward solution.
     */
    static MNELIB::MNEForwardSolution restrict_forward_to_stc(const MNELIB::MNEForwardSolution& fwd,
                                                            const MNELIB::MNESourceEstimate& stc,
                                                            bool copy = true);

private:
    //=========================================================================================================
    /**
     * Helper function to validate forward solution and source estimate compatibility
     *
     * @param[in] fwd           The forward solution.
     * @param[in] stc           The source estimate.
     *
     * @return True if compatible, false otherwise.
     */
    static bool validate_fwd_stc_compatibility(const MNELIB::MNEForwardSolution& fwd,
                                             const MNELIB::MNESourceEstimate& stc);

    //=========================================================================================================
    /**
     * Helper function to apply cortical patch statistics
     *
     * @param[in] fwd           The forward solution.
     * @param[in] data          The source data.
     * @param[in] use_cps       Whether to use cortical patch statistics.
     *
     * @return The processed data.
     */
    static Eigen::MatrixXd apply_cortical_patch_statistics(const MNELIB::MNEForwardSolution& fwd,
                                                         const Eigen::MatrixXd& data,
                                                         bool use_cps);

    //=========================================================================================================
    /**
     * Helper function to convert source orientations
     *
     * @param[in] fwd           The forward solution.
     * @param[in] surf_ori      Whether to use surface-based orientations.
     * @param[in] force_fixed   Whether to force fixed orientations.
     *
     * @return The converted forward solution.
     */
    static MNELIB::MNEForwardSolution convert_source_orientations(const MNELIB::MNEForwardSolution& fwd,
                                                                bool surf_ori,
                                                                bool force_fixed);
};

} // NAMESPACE

#endif // FWD_SOLUTION_PROCESSING_H