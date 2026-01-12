#include <QtTest/QtTest>
#include <connectivity/metrics/coherence.h>
#include <connectivity/metrics/phaselagindex.h>
#include <connectivity/connectivitysettings.h>
#include <connectivity/network/network.h>
#include <connectivity/network/networknode.h>
#include <connectivity/network/networkedge.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace CONNECTIVITYLIB;
using namespace Eigen;

class TestConnectivity : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testCoherence();
    void testPLI();
};

void TestConnectivity::initTestCase()
{
}

void TestConnectivity::cleanupTestCase()
{
}

void TestConnectivity::testCoherence()
{
    // Setup Data
    int n_channels = 3;
    int n_samples = 1000;
    int n_trials = 50;
    double sfreq = 1000.0;
    
    ConnectivitySettings settings;
    settings.setSamplingFrequency(sfreq);
    settings.setWindowType("hanning"); // Single taper for simplicity
    settings.setFFTSize(n_samples); // No padding
    
    // Generate Trials
    for(int t=0; t<n_trials; ++t) {
        MatrixXd matData(n_channels, n_samples);
        for(int i=0; i<n_samples; ++i) {
            double time = i / sfreq;
            double signal = std::sin(2.0 * M_PI * 10.0 * time); // 10 Hz
            
            matData(0, i) = signal + 0.1 * ((double)rand()/RAND_MAX - 0.5);
            matData(1, i) = signal + 0.1 * ((double)rand()/RAND_MAX - 0.5); // Correlated with 0
            matData(2, i) = ((double)rand()/RAND_MAX - 0.5); // Noise
        }
        
        ConnectivitySettings::IntermediateTrialData trial;
        trial.matData = matData;
        settings.append(trial);
    }
    
    // Calculate Coherence
    Network network = Coherence::calculate(settings);
    
    // Check results
    // Network contains nodes and edges.
    // Edge weights are MatrixXd (1 x n_freqs) usually? Or averaged?
    // Coherence returns full spectrum.
    
    // Find edge 0-1
    // Network has nodes. Each node has edges.
    // Let's iterate edges.
    
    QSharedPointer<NetworkNode> node0 = network.getNodeAt(0);
    QList<QSharedPointer<NetworkEdge>> edges = node0->getFullEdges();
    
    bool found01 = false;
    bool found02 = false;
    
    for(auto edge : edges) {
        int other = (edge->getStartNodeID() == 0) ? edge->getEndNodeID() : edge->getStartNodeID();
        
        MatrixXd weights = edge->getMatrixWeight();
        // Weights: (n_freqs x 1)? Or (1 x n_freqs)?
        // Coherency::computePSDCSDAbs: matWeight = matCohy.row(j).cwiseAbs().transpose(); 
        // matCohy is 1 x Freqs. Transpose is Freqs x 1.
        
        // Find 10 Hz bin
        int bin10Hz = int(10.0 * n_samples / sfreq);
        
        if (other == 1) {
            found01 = true;
            // Check if bin10Hz is within bounds
            if(bin10Hz < weights.rows()) {
                double coh = weights(bin10Hz, 0);
                qDebug() << "Coh 0-1 at 10Hz:" << coh;
                QVERIFY(coh > 0.9);
            }
        } else if (other == 2) {
            found02 = true;
            if(bin10Hz < weights.rows()) {
                double coh = weights(bin10Hz, 0);
                qDebug() << "Coh 0-2 at 10Hz:" << coh;
                QVERIFY(coh < 0.5);
            }
        }
    }
    
    QVERIFY(found01);
    QVERIFY(found02);
}

void TestConnectivity::testPLI()
{
    // Setup Data
    int n_channels = 2;
    int n_samples = 1000;
    int n_trials = 20; // Need trials for PLI consistency
    double sfreq = 1000.0;
    
    ConnectivitySettings settings;
    settings.setSamplingFrequency(sfreq);
    settings.setWindowType("hanning");
    settings.setFFTSize(n_samples);
    
    // Generate Trials
    // 0 and 1 have constant phase difference (90 deg)
    for(int t=0; t<n_trials; ++t) {
        MatrixXd matData(n_channels, n_samples);
        for(int i=0; i<n_samples; ++i) {
            double time = i / sfreq;
            double signal = std::sin(2.0 * M_PI * 10.0 * time);
            double signal_lag = std::sin(2.0 * M_PI * 10.0 * time + M_PI/2.0);
            
            matData(0, i) = signal + 0.1 * ((double)rand()/RAND_MAX - 0.5);
            matData(1, i) = signal_lag + 0.1 * ((double)rand()/RAND_MAX - 0.5);
        }
        
        ConnectivitySettings::IntermediateTrialData trial;
        trial.matData = matData;
        settings.append(trial);
    }
    
    // Calculate PLI
    Network network = PhaseLagIndex::calculate(settings);
    
    QSharedPointer<NetworkNode> node0 = network.getNodeAt(0);
    QList<QSharedPointer<NetworkEdge>> edges = node0->getFullEdges();
    
    bool found01 = false;
    for(auto edge : edges) {
        // Assuming edge connects 0 and 1
        MatrixXd weights = edge->getMatrixWeight();
        int bin10Hz = int(10.0 * n_samples / sfreq);
        
        if(bin10Hz < weights.rows()) {
            double pli = weights(bin10Hz, 0);
            
            qDebug() << "PLI 0-1 at 10Hz:" << pli;
            
            // PLI should be close to 1 because phase diff is constant (+90 deg)
            QVERIFY(pli > 0.9);
            found01 = true;
        }
    }
    QVERIFY(found01);
}

QTEST_GUILESS_MAIN(TestConnectivity)
#include "test_connectivity.moc"
