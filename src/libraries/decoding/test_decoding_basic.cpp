#include "csp.h"
#include "temporal_decoding.h"
#include "spatial_filtering.h"
#include <iostream>
#include <vector>
#include <random>

using namespace DECODINGLIB;

int main()
{
    std::cout << "Testing MNE-CPP Decoding Module..." << std::endl;
    
    // Generate synthetic data
    int n_epochs = 100;
    int n_channels = 32;
    int n_times = 200;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    
    std::vector<Eigen::MatrixXd> epochs;
    std::vector<int> labels;
    
    for (int i = 0; i < n_epochs; ++i) {
        Eigen::MatrixXd epoch(n_channels, n_times);
        for (int c = 0; c < n_channels; ++c) {
            for (int t = 0; t < n_times; ++t) {
                epoch(c, t) = dis(gen);
            }
        }
        
        // Add some class-specific signal
        int label = i % 2;
        if (label == 1) {
            // Add signal to first few channels for class 1
            for (int c = 0; c < 5; ++c) {
                for (int t = 50; t < 150; ++t) {
                    epoch(c, t) += 2.0 * dis(gen);
                }
            }
        }
        
        epochs.push_back(epoch);
        labels.push_back(label);
    }
    
    std::cout << "Generated " << n_epochs << " epochs with " << n_channels 
              << " channels and " << n_times << " time points." << std::endl;
    
    // Test CSP
    std::cout << "\nTesting CSP..." << std::endl;
    CSP csp(4, false, true);
    
    if (csp.fit(epochs, labels)) {
        std::cout << "CSP fit successful!" << std::endl;
        
        Eigen::MatrixXd features = csp.transform(epochs);
        std::cout << "CSP features shape: " << features.rows() << " x " << features.cols() << std::endl;
        
        Eigen::MatrixXd filters = csp.getFilters();
        std::cout << "CSP filters shape: " << filters.rows() << " x " << filters.cols() << std::endl;
    } else {
        std::cout << "CSP fit failed!" << std::endl;
    }
    
    // Test SPoC
    std::cout << "\nTesting SPoC..." << std::endl;
    SPoC spoc(4, false, true);
    
    // Generate continuous target
    Eigen::VectorXd target(n_epochs);
    for (int i = 0; i < n_epochs; ++i) {
        target(i) = dis(gen);
    }
    
    if (spoc.fit(epochs, target)) {
        std::cout << "SPoC fit successful!" << std::endl;
        
        Eigen::MatrixXd features = spoc.transform(epochs);
        std::cout << "SPoC features shape: " << features.rows() << " x " << features.cols() << std::endl;
    } else {
        std::cout << "SPoC fit failed!" << std::endl;
    }
    
    // Test Sliding Estimator
    std::cout << "\nTesting SlidingEstimator..." << std::endl;
    auto base_estimator = std::make_shared<LinearDiscriminantAnalysis>(0.01);
    SlidingEstimator sliding_est(base_estimator);
    
    Eigen::VectorXi labels_eigen = Eigen::VectorXi::Map(labels.data(), labels.size());
    
    if (sliding_est.fit(epochs, labels_eigen)) {
        std::cout << "SlidingEstimator fit successful!" << std::endl;
        
        Eigen::VectorXd scores = sliding_est.score(epochs, labels_eigen);
        std::cout << "SlidingEstimator scores shape: " << scores.size() << std::endl;
        std::cout << "Mean accuracy: " << scores.mean() << std::endl;
    } else {
        std::cout << "SlidingEstimator fit failed!" << std::endl;
    }
    
    // Test Spatial Filtering
    std::cout << "\nTesting UnsupervisedSpatialFilter (PCA)..." << std::endl;
    UnsupervisedSpatialFilter pca_filter(UnsupervisedSpatialFilter::Method::PCA, 10);
    
    if (pca_filter.fit(epochs)) {
        std::cout << "PCA filter fit successful!" << std::endl;
        
        std::vector<Eigen::MatrixXd> filtered_epochs = pca_filter.transform(epochs);
        std::cout << "Filtered epochs count: " << filtered_epochs.size() << std::endl;
        if (!filtered_epochs.empty()) {
            std::cout << "Filtered epoch shape: " << filtered_epochs[0].rows() 
                      << " x " << filtered_epochs[0].cols() << std::endl;
        }
        
        Eigen::VectorXd explained_var = pca_filter.getExplainedVarianceRatio();
        std::cout << "Explained variance (first 5 components): ";
        for (int i = 0; i < std::min(5, (int)explained_var.size()); ++i) {
            std::cout << explained_var(i) << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "PCA filter fit failed!" << std::endl;
    }
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}