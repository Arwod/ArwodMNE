#include <QtTest/QtTest>
#include <mne/c/mne_morph_map.h>
#include <mne/mne_sourceestimate.h>
#include <fiff/c/fiff_sparse_matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using namespace MNELIB;
using namespace FIFFLIB;
using namespace Eigen;

class TestMorphing : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testToEigenSparse();
    void testMorphSourceEstimateManual();
};

void TestMorphing::initTestCase()
{
}

void TestMorphing::cleanupTestCase()
{
}

void TestMorphing::testToEigenSparse()
{
    // Manually create a FiffSparseMatrix (RCS)
    // 3x3 identity matrix
    int nrow = 3;
    int ncol = 3;
    int nnz[] = {1, 1, 1};

    int *colindex[3];
    float *vals[3];

    for (int i = 0; i < 3; ++i)
    {
        colindex[i] = new int[1];
        vals[i] = new float[1];
        colindex[i][0] = i;
        vals[i][0] = 1.0f;
    }

    FiffSparseMatrix *fiffMat = FiffSparseMatrix::create_sparse_rcs(nrow, ncol, nnz, colindex, vals);

    QVERIFY(fiffMat != nullptr);
    QCOMPARE(fiffMat->m, 3);
    QCOMPARE(fiffMat->n, 3);
    QCOMPARE(fiffMat->nz, 3);
    QCOMPARE(fiffMat->coding, FIFFTS_MC_RCS);

    // Convert to Eigen
    SparseMatrix<double> eigenMat = fiffMat->toEigenSparse();

    QCOMPARE(eigenMat.rows(), 3);
    QCOMPARE(eigenMat.cols(), 3);
    QCOMPARE(eigenMat.nonZeros(), 3);

    // Verify values
    MatrixXd dense = MatrixXd(eigenMat);
    QCOMPARE(dense(0, 0), 1.0);
    QCOMPARE(dense(1, 1), 1.0);
    QCOMPARE(dense(2, 2), 1.0);
    QCOMPARE(dense(0, 1), 0.0);

    // Cleanup manually allocated arrays
    for (int i = 0; i < 3; ++i)
    {
        delete[] colindex[i];
        delete[] vals[i];
    }
    delete fiffMat;
}

void TestMorphing::testMorphSourceEstimateManual()
{
    // Simulate a Morphing process
    // STC: 4 vertices (2 LH, 2 RH).
    // Morph Map LH: 2x2 Identity (No change)
    // Morph Map RH: 2x2 permutation (Swap)

    // 1. Create SourceEstimate
    MatrixXd data(4, 5); // 4 sources, 5 time points
    data.setZero();
    data.row(0).setConstant(1.0); // LH 0
    data.row(1).setConstant(2.0); // LH 1
    data.row(2).setConstant(3.0); // RH 0
    data.row(3).setConstant(4.0); // RH 1

    VectorXi vertices(4);
    vertices << 0, 1, 0, 1; // 0,1 for LH; 0,1 for RH (indices reset)
    // Note: MNESourceEstimate vertices vector usually implies splitting.
    // Our heuristic looks for decreasing index.

    MNESourceEstimate stc(data, vertices, 0.0f, 1.0f);

    // 2. Create Morph Maps (Manual FiffSparseMatrix -> MneMorphMap)
    // Map LH: Identity 2x2
    int nrow = 2, ncol = 2;
    int nnz_lh[] = {1, 1};
    int *col_lh[2];
    float *val_lh[2];
    col_lh[0] = new int[1];
    col_lh[0][0] = 0;
    val_lh[0] = new float[1];
    val_lh[0][0] = 1.0f;
    col_lh[1] = new int[1];
    col_lh[1][0] = 1;
    val_lh[1] = new float[1];
    val_lh[1][0] = 1.0f;

    FiffSparseMatrix *fiffMatLH = FiffSparseMatrix::create_sparse_rcs(nrow, ncol, nnz_lh, col_lh, val_lh);

    // Map RH: Swap 2x2 (Row 0 takes Col 1, Row 1 takes Col 0)
    int nnz_rh[] = {1, 1};
    int *col_rh[2];
    float *val_rh[2];
    col_rh[0] = new int[1];
    col_rh[0][0] = 1;
    val_rh[0] = new float[1];
    val_rh[0][0] = 1.0f;
    col_rh[1] = new int[1];
    col_rh[1][0] = 0;
    val_rh[1] = new float[1];
    val_rh[1][0] = 1.0f;

    FiffSparseMatrix *fiffMatRH = FiffSparseMatrix::create_sparse_rcs(nrow, ncol, nnz_rh, col_rh, val_rh);

    // Wrap in MneMorphMap
    MneMorphMap::SPtr mapLH(new MneMorphMap());
    mapLH->map = fiffMatLH;
    MneMorphMap::SPtr mapRH(new MneMorphMap());
    mapRH->map = fiffMatRH;

    // 3. Apply Morphing manually (simulating morphSourceEstimate logic)
    // We can't call morphSourceEstimate easily because it tries to read from file.
    // We will verify the core logic: split -> toEigen -> multiply -> merge.

    Eigen::SparseMatrix<double> matLH = mapLH->toEigen();
    Eigen::SparseMatrix<double> matRH = mapRH->toEigen();

    // Split STC
    int n_lh = 2;
    int n_rh = 2;
    MatrixXd data_lh = stc.data.topRows(n_lh);
    MatrixXd data_rh = stc.data.bottomRows(n_rh);

    // Multiply
    MatrixXd res_lh = matLH * data_lh;
    MatrixXd res_rh = matRH * data_rh;

    // Verify LH (Identity)
    QCOMPARE(res_lh(0, 0), 1.0);
    QCOMPARE(res_lh(1, 0), 2.0);

    // Verify RH (Swap)
    // res_rh row 0 = 1.0 * data_rh row 1 (val 4.0) = 4.0
    // res_rh row 1 = 1.0 * data_rh row 0 (val 3.0) = 3.0
    QCOMPARE(res_rh(0, 0), 4.0);
    QCOMPARE(res_rh(1, 0), 3.0);

    // Cleanup
    for (int i = 0; i < 2; ++i)
    {
        delete[] col_lh[i];
        delete[] val_lh[i];
        delete[] col_rh[i];
        delete[] val_rh[i];
    }
}

QTEST_GUILESS_MAIN(TestMorphing)
#include "test_morphing.moc"
