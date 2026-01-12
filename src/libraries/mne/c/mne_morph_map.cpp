//=============================================================================================================
/**
 * @file     mne_morph_map.cpp
 * @author   Lorenz Esch <lesch@mgh.harvard.edu>;
 *           Matti Hamalainen <msh@nmr.mgh.harvard.edu>
 * @since    0.1.0
 * @date     April, 2017
 *
 * @section  LICENSE
 *
 * Copyright (C) 2017, Lorenz Esch, Matti Hamalainen. All rights reserved.
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
 * @brief    Definition of the MneMorphMap Class.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "mne_morph_map.h"
#include <fiff/fiff_stream.h>
#include <QFile>
#include <QDebug>

#define FREE_45(x)           \
    if ((char *)(x) != NULL) \
    free((char *)(x))

#define SURF_UNKNOWN -1

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace MNELIB;

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

MneMorphMap::MneMorphMap()
{
    from_kind = SURF_UNKNOWN;
    from_subj = Q_NULLPTR;
    map = Q_NULLPTR;
    best = Q_NULLPTR;
}

//=============================================================================================================

MneMorphMap::~MneMorphMap()
{
    FREE_45(from_subj);
    delete map;
    FREE_45(best);
}

//=============================================================================================================

bool MneMorphMap::readMorphMap(const QString &subject_from,
                               const QString &subject_to,
                               const QString &subjects_dir,
                               MneMorphMap::SPtr &map_lh,
                               MneMorphMap::SPtr &map_rh)
{
    QString fileName = QString("%1/morph-maps/%2-%3-morph.fif")
                           .arg(subjects_dir)
                           .arg(subject_from)
                           .arg(subject_to);

    QFile file(fileName);
    if (!file.exists())
    {
        // Try opposite direction? No, morph maps are directional usually.
        // MNE-Python tries to create if not found. We only read.
        qWarning() << "MneMorphMap::readMorphMap - File not found:" << fileName;
        return false;
    }

    FIFFLIB::FiffStream::SPtr t_pStream(new FIFFLIB::FiffStream(&file));
    if (!t_pStream->open())
    {
        qWarning() << "MneMorphMap::readMorphMap - Could not open file:" << fileName;
        return false;
    }

    FIFFLIB::FiffDirNode::SPtr Tree = t_pStream->dirtree();
    QList<FIFFLIB::FiffDirNode::SPtr> morph_maps = Tree->dir_tree_find(FIFFB_MNE_MORPH_MAP);

    if (morph_maps.isEmpty())
    {
        qWarning() << "MneMorphMap::readMorphMap - No morph map found in file.";
        return false;
    }

    // We expect one block with multiple maps (usually 2 for LH/RH)
    // Or multiple blocks? Standard is one block.

    QList<FIFFLIB::FiffTag::SPtr> mat_tags;

    // Find all tags in the first morph_map block
    // We can't iterate tags easily without reading.
    // FiffStream doesn't have "read_all_tags_in_block".
    // We have to iterate the directory entries of the block.

    // Actually, FiffDirTree has 'find_tag' but that might find only one.
    // Let's iterate over dir entries of the block.
    // But FiffDirEntry points to file position.

    // Better way: use read_tag on the stream, navigating to the block.
    // Or assume standard structure.

    // Let's look at how FiffStream reads matrices.
    // We can use t_pStream->read_named_matrix? No, these are sparse.

    // Let's assume we read all sparse matrices in the block.
    // MNE-Python reads all tags.

    QList<MneMorphMap::SPtr> maps;

    for (int k = 0; k < morph_maps[0]->nent(); ++k)
    {
        FIFFLIB::FiffDirEntry::SPtr &ent = morph_maps[0]->dir[k];
        if (ent->kind == FIFF_MNE_MORPH_MAP)
        {
            FIFFLIB::FiffTag::SPtr t_pTag;
            if (t_pStream->read_tag(t_pTag, ent->pos))
            {
                MneMorphMap::SPtr pMap(new MneMorphMap());
                pMap->map = FIFFLIB::FiffSparseMatrix::fiff_get_float_sparse_matrix(t_pTag);
                if (pMap->map)
                {
                    maps.append(pMap);
                }
            }
        }
    }

    if (maps.size() == 2)
    {
        map_lh = maps[0];
        map_rh = maps[1];
        return true;
    }
    else if (maps.size() == 1)
    {
        // Maybe only one hemi?
        map_lh = maps[0];
        return true;
    }
    else
    {
        qWarning() << "MneMorphMap::readMorphMap - Found" << maps.size() << "maps. Expected 1 or 2.";
        return false;
    }
}

//=============================================================================================================

MNESourceEstimate MneMorphMap::morphSourceEstimate(const MNESourceEstimate &stc,
                                                   const QString &subject_from,
                                                   const QString &subject_to,
                                                   const QString &subjects_dir)
{
    MneMorphMap::SPtr map_lh, map_rh;
    if (!readMorphMap(subject_from, subject_to, subjects_dir, map_lh, map_rh))
    {
        return MNESourceEstimate();
    }

    // Convert to Eigen Sparse
    Eigen::SparseMatrix<double> mat_lh = map_lh ? map_lh->toEigen() : Eigen::SparseMatrix<double>();
    Eigen::SparseMatrix<double> mat_rh = map_rh ? map_rh->toEigen() : Eigen::SparseMatrix<double>();

    // Split STC data
    // Assuming stc.vertices is [lh_indices, rh_indices] and stc.data is [lh_data; rh_data]
    // We need to find the split point.

    int n_lh = 0;
    int n_rh = 0;
    int split_idx = 0;

    // Heuristic to find split: when index decreases
    if (stc.vertices.size() > 0)
    {
        for (int i = 1; i < stc.vertices.size(); ++i)
        {
            if (stc.vertices[i] < stc.vertices[i - 1])
            {
                split_idx = i;
                break;
            }
        }
    }

    if (split_idx == 0 && stc.vertices.size() > 0)
    {
        // No split found? Maybe only one hemi?
        // Or sorted?
        // If we have both maps, we expect split.
        // If only one map, maybe entire STC is for that hemi.
        if (map_lh && map_rh)
        {
            // Problem. Assume equal split? No.
            // Assume first half? No.
            // Maybe check against map dimensions?
            // map_lh->map->n should match number of source vertices in LH.
            if (map_lh->map->n == stc.vertices.size())
            {
                split_idx = stc.vertices.size(); // All LH
            }
            else if (map_rh && map_rh->map->n == stc.vertices.size())
            {
                split_idx = 0; // All RH
            }
            else
            {
                // Try to deduce from map dimensions
                if (map_lh->map->n + map_rh->map->n == stc.vertices.size())
                {
                    split_idx = map_lh->map->n;
                }
                else
                {
                    qWarning() << "MneMorphMap::morphSourceEstimate - Dimension mismatch.";
                    return MNESourceEstimate();
                }
            }
        }
        else if (map_lh)
        {
            split_idx = stc.vertices.size();
        }
        else
        {
            split_idx = 0;
        }
    }

    n_lh = split_idx;
    n_rh = stc.vertices.size() - split_idx;

    Eigen::MatrixXd data_lh = stc.data.topRows(n_lh);
    Eigen::MatrixXd data_rh = stc.data.bottomRows(n_rh);

    Eigen::MatrixXd res_lh, res_rh;

    if (map_lh && n_lh > 0)
    {
        if (map_lh->map->n != n_lh)
        {
            qWarning() << "MneMorphMap::morphSourceEstimate - LH dimension mismatch:" << map_lh->map->n << "vs" << n_lh;
        }
        res_lh = mat_lh * data_lh;
    }

    if (map_rh && n_rh > 0)
    {
        if (map_rh->map->n != n_rh)
        {
            qWarning() << "MneMorphMap::morphSourceEstimate - RH dimension mismatch:" << map_rh->map->n << "vs" << n_rh;
        }
        res_rh = mat_rh * data_rh;
    }

    // Combine results
    int n_out_lh = res_lh.rows();
    int n_out_rh = res_rh.rows();
    int n_out = n_out_lh + n_out_rh;

    if (n_out == 0)
        return MNESourceEstimate();

    Eigen::MatrixXd res_data(n_out, stc.data.cols());
    if (n_out_lh > 0)
        res_data.topRows(n_out_lh) = res_lh;
    if (n_out_rh > 0)
        res_data.bottomRows(n_out_rh) = res_rh;

    // Construct new STC
    // Vertices? We need the target vertices.
    // The morph map maps FROM source TO target.
    // But does it provide target vertices indices?
    // FiffSparseMatrix doesn't store target indices, just the mapping.
    // Usually, morphing to 'fsaverage' implies we map to fsaverage's source space (e.g. ico-5).
    // The resulting data rows correspond to the target source space vertices in order.
    // So we need to know the target vertices.
    // Typically, one loads the target source space to get vertices.
    // OR, if the morph map is defined on a standard grid, the indices are 0..N-1?
    // MNE-Python `morph_mat` is (n_target_vertices, n_source_vertices).
    // The output is data on the target vertices.
    // If we don't have the target vertices indices, we can't fully populate MNESourceEstimate.vertices.
    // HOWEVER, for now, we can create a dummy vector 0..N-1 or leave it empty if we just want data.
    // But MNESourceEstimate requires vertices.
    //
    // Ideally, `morphSourceEstimate` should take `subject_to`'s source space as input?
    // Or we assume 0..N-1.
    // Let's populate vertices with 0..N-1 for now, as that's often implicit in "morphed" data (it fills the whole space).

    Eigen::VectorXi vertices(n_out);
    for (int i = 0; i < n_out; ++i)
        vertices[i] = i;

    return MNESourceEstimate(res_data, vertices, stc.tmin, stc.tstep);
}

//=============================================================================================================

Eigen::SparseMatrix<double> MneMorphMap::toEigen() const
{
    if (this->map)
    {
        return this->map->toEigenSparse();
    }
    return Eigen::SparseMatrix<double>();
}
