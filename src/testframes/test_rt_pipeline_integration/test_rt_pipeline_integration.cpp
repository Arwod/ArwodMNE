#include <QtTest/QtTest>
#include <QtTest/QSignalSpy>
#include <QtCore/QDebug>

#include <rtprocessing/rtcov.h>
#include <rtprocessing/rtinvop.h>
#include <fiff/fiff_info.h>
#include <mne/mne_forwardsolution.h>
#include <mne/mne_inverse_operator.h>
#include <utils/ioutils.h>

using namespace RTPROCESSINGLIB;
using namespace FIFFLIB;
using namespace MNELIB;
using namespace UTILSLIB;
using namespace Eigen;

class TestRtPipelineIntegration : public QObject
{
    Q_OBJECT

public:
    TestRtPipelineIntegration();
    ~TestRtPipelineIntegration();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCovarianceToInverse();
    
private:
    QString dataPath;
};

TestRtPipelineIntegration::TestRtPipelineIntegration()
{
}

TestRtPipelineIntegration::~TestRtPipelineIntegration()
{
}

void TestRtPipelineIntegration::initTestCase()
{
    qRegisterMetaType<MNELIB::MNEInverseOperator>("MNELIB::MNEInverseOperator");
    qRegisterMetaType<MNELIB::MNEInverseOperator>("MNEInverseOperator"); // just in case
    
    // Use data from previous phases
    dataPath = QCoreApplication::applicationDirPath() + "/../../../data";
    qDebug() << "Data Path:" << dataPath;
}

void TestRtPipelineIntegration::cleanupTestCase()
{
}

void TestRtPipelineIntegration::testCovarianceToInverse()
{
    // 1. Load FiffInfo (from Phase 6 or 8 data)
    // We can use "df_evoked-ave.fif" to get the info
    QString evokedPath = dataPath + "/df_evoked-ave.fif";
    QFile file(evokedPath);
    QVERIFY(file.exists());
    
    FiffEvoked evoked(file);
    QSharedPointer<FiffInfo> pInfo = QSharedPointer<FiffInfo>::create(evoked.info);
    
    // 2. Setup RtCov
    RtCov rtCov(pInfo);
    
    // 3. Setup RtInvOp
    // We need a Forward Solution. We can use "df_evoked-ave.fif" if we had a fwd...
    // But in Phase 8 we didn't save the FWD to a file, we generated it in Python.
    // Wait, Phase 6 (Minimum Norm) might have saved something?
    // "mn_inv.fif" exists. We can read fwd from there or just read the inv op and extract fwd?
    // MNEInverseOperator contains forward solution? No, it contains source space and other things.
    // We need a MNEForwardSolution.
    
    // Let's assume we can generate a dummy Fwd or try to read one.
    // Ideally we should have "sample_audvis-meg-eeg-oct-6-fwd.fif" style file.
    // But we are using simulated data.
    
    // Hack: We can construct a simple ForwardSolution or Mock it.
    // Or we can rely on the fact that we can't fully run RtInvOp without a real Fwd.
    // Let's try to load "mn_inv.fif" and use its info to construct a dummy fwd?
    // Or better, let's update gen_test_data.py to save the forward solution used in Phase 8.
    
    // Let's assume for now we skip the FWD part until we fix gen_test_data.
    // Or we can create a placeholder.
    
    // Actually, let's update gen_test_data.py to save the forward solution in Phase 8.
    // This is safer.
    
    // For now, I will write the code assuming "df_fwd.fif" exists.
    
    // 4. Load Forward Solution
    QFile fwdFile(dataPath + "/df_fwd.fif");
    QVERIFY(fwdFile.exists());
    
    QSharedPointer<MNEForwardSolution> pFwd = QSharedPointer<MNEForwardSolution>::create(fwdFile);
    QVERIFY(!pFwd->isEmpty());
    
    // 5. Setup RtInvOp
    RtInvOp rtInvOp(pInfo, pFwd);
    QSignalSpy spyInvOp(&rtInvOp, &RtInvOp::invOperatorCalculated);
    
    // 6. Feed Data to RtCov
    // Create random noise data matching channels
    int nChan = pInfo->chs.size();
    int blockSize = 100;
    int nBlocks = 20; // 2000 samples
    
    FiffCov noiseCov;
    
    for(int i = 0; i < nBlocks; ++i) {
        MatrixXd matData = MatrixXd::Random(nChan, blockSize); // Gaussian noise
        noiseCov = rtCov.estimateCovariance(matData, 1000); // Want 1000 samples
        
        if(!noiseCov.isEmpty()) {
            qDebug() << "Covariance computed at block" << i;
            
            // Pass to RtInvOp
            rtInvOp.append(noiseCov);
            break;
        }
    }
    
    QVERIFY(!noiseCov.isEmpty());
    QVERIFY(noiseCov.data.rows() == nChan);
    
    // 7. Verify RtInvOp Result
    // RtInvOp runs in a thread, so we wait for the signal
    QVERIFY(spyInvOp.wait(5000)); // Wait up to 5s
    
    QList<QVariant> arguments = spyInvOp.takeFirst();
    MNEInverseOperator invOp = arguments.at(0).value<MNEInverseOperator>();
    
    QVERIFY(!invOp.eigen_leads->isEmpty());
    qDebug() << "Inverse Operator computed with" << invOp.eigen_leads->nrow << "sources";
    QVERIFY(invOp.eigen_leads->nrow > 0);
}

QTEST_GUILESS_MAIN(TestRtPipelineIntegration)
#include "test_rt_pipeline_integration.moc"
