#ifndef SPATIAL_FILTERING_H
#define SPATIAL_FILTERING_H

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "decoding_global.h"
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>

//=============================================================================================================
// DEFINE NAMESPACE DECODINGLIB
//=============================================================================================================

namespace DECODINGLIB
{

//=============================================================================================================
/**
 * Base class for spatial filtering methods
 */
class DECODINGSHARED_EXPORT BaseSpatialFilter
{
public:
    typedef std::shared_ptr<BaseSpatialFilter> SPtr;
    typedef std::shared_ptr<const BaseSpatialFilter> ConstSPtr;

    virtual ~BaseSpatialFilter() = default;

    /**
     * Fit the spatial filter
     */
    virtual bool fit(const std::vector<Eigen::MatrixXd>& epochs) = 0;

    /**
     * Apply the spatial filter
     */
    virtual std::vector<Eigen::MatrixXd> transform(const std::vector<Eigen::MatrixXd>& epochs) const = 0;

    /**
     * Fit and transform in one step
     */
    virtual std::vector<Eigen::MatrixXd> fitTransform(const std::vector<Eigen::MatrixXd>& epochs);

    /**
     * Get the spatial filters
     */
    virtual Eigen::MatrixXd getFilters() const = 0;
};

//=============================================================================================================
/**
 * Supervised Spatial Filter
 * 
 * Base class for supervised spatial filtering methods that use labels.
 */
class DECODINGSHARED_EXPORT SupervisedSpatialFilter : public BaseSpatialFilter
{
public:
    virtual ~SupervisedSpatialFilter() = default;

    /**
     * Fit the spatial filter with labels
     */
    virtual bool fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels) = 0;

    /**
     * Fit without labels (calls supervised version with empty labels)
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs) override;

    /**
     * Fit and transform with labels
     */
    virtual std::vector<Eigen::MatrixXd> fitTransform(const std::vector<Eigen::MatrixXd>& epochs, 
                                                     const std::vector<int>& labels);
};

//=============================================================================================================
/**
 * Unsupervised Spatial Filter
 * 
 * Implements various unsupervised spatial filtering methods.
 */
class DECODINGSHARED_EXPORT UnsupervisedSpatialFilter : public BaseSpatialFilter
{
public:
    typedef std::shared_ptr<UnsupervisedSpatialFilter> SPtr;
    typedef std::shared_ptr<const UnsupervisedSpatialFilter> ConstSPtr;

    /**
     * @brief Spatial filtering methods
     */
    enum class Method {
        PCA,            /**< Principal Component Analysis */
        ICA,            /**< Independent Component Analysis */
        Whitening,      /**< Whitening transformation */
        Laplacian       /**< Laplacian spatial filter */
    };

    //=========================================================================================================
    /**
     * Constructs an UnsupervisedSpatialFilter.
     *
     * @param[in] method        Filtering method to use.
     * @param[in] n_components  Number of components to keep.
     * @param[in] reg_param     Regularization parameter.
     */
    explicit UnsupervisedSpatialFilter(Method method = Method::PCA,
                                      int n_components = 10,
                                      double reg_param = 0.01);

    //=========================================================================================================
    /**
     * Destroys the UnsupervisedSpatialFilter.
     */
    ~UnsupervisedSpatialFilter() = default;

    //=========================================================================================================
    /**
     * Fit the spatial filter.
     * 
     * @param[in] epochs Vector of epochs data (Channels x Time).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs) override;

    //=========================================================================================================
    /**
     * Apply the spatial filter.
     * 
     * @param[in] epochs Vector of epochs data.
     * @return Filtered epochs.
     */
    std::vector<Eigen::MatrixXd> transform(const std::vector<Eigen::MatrixXd>& epochs) const override;

    //=========================================================================================================
    /**
     * Get the spatial filters.
     */
    Eigen::MatrixXd getFilters() const override { return m_matFilters; }

    //=========================================================================================================
    /**
     * Get the explained variance ratio (for PCA).
     */
    Eigen::VectorXd getExplainedVarianceRatio() const { return m_vecExplainedVariance; }

    //=========================================================================================================
    /**
     * Set the number of components.
     */
    void setNComponents(int n_components) { m_iNComponents = n_components; }

private:
    //=========================================================================================================
    /**
     * Fit PCA spatial filter.
     */
    bool fitPCA(const std::vector<Eigen::MatrixXd>& epochs);

    //=========================================================================================================
    /**
     * Fit whitening spatial filter.
     */
    bool fitWhitening(const std::vector<Eigen::MatrixXd>& epochs);

    //=========================================================================================================
    /**
     * Fit Laplacian spatial filter.
     */
    bool fitLaplacian(const std::vector<Eigen::MatrixXd>& epochs);

    //=========================================================================================================
    /**
     * Compute covariance matrix from epochs.
     */
    Eigen::MatrixXd computeCovarianceMatrix(const std::vector<Eigen::MatrixXd>& epochs) const;

    Method m_method;
    int m_iNComponents;
    double m_dRegParam;
    
    Eigen::MatrixXd m_matFilters;
    Eigen::VectorXd m_vecExplainedVariance;
    Eigen::VectorXd m_vecMean;
    bool m_bFitted;
};

//=============================================================================================================
/**
 * Xdawn Spatial Filter
 * 
 * Implements the Xdawn spatial filtering method for ERP enhancement.
 */
class DECODINGSHARED_EXPORT XdawnSpatialFilter : public SupervisedSpatialFilter
{
public:
    typedef std::shared_ptr<XdawnSpatialFilter> SPtr;
    typedef std::shared_ptr<const XdawnSpatialFilter> ConstSPtr;

    //=========================================================================================================
    /**
     * Constructs an XdawnSpatialFilter.
     *
     * @param[in] n_components  Number of components to extract.
     * @param[in] reg_param     Regularization parameter.
     */
    explicit XdawnSpatialFilter(int n_components = 4, double reg_param = 0.01);

    //=========================================================================================================
    /**
     * Destroys the XdawnSpatialFilter.
     */
    ~XdawnSpatialFilter() = default;

    //=========================================================================================================
    /**
     * Fit the Xdawn spatial filter.
     * 
     * @param[in] epochs Vector of epochs data (Channels x Time).
     * @param[in] labels Vector of labels (0 for non-target, 1 for target).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs, const std::vector<int>& labels) override;

    //=========================================================================================================
    /**
     * Apply the Xdawn spatial filter.
     * 
     * @param[in] epochs Vector of epochs data.
     * @return Filtered epochs.
     */
    std::vector<Eigen::MatrixXd> transform(const std::vector<Eigen::MatrixXd>& epochs) const override;

    //=========================================================================================================
    /**
     * Get the spatial filters.
     */
    Eigen::MatrixXd getFilters() const override { return m_matFilters; }

    //=========================================================================================================
    /**
     * Get the spatial patterns.
     */
    Eigen::MatrixXd getPatterns() const { return m_matPatterns; }

    //=========================================================================================================
    /**
     * Get the evoked response.
     */
    Eigen::MatrixXd getEvokedResponse() const { return m_matEvoked; }

private:
    int m_iNComponents;
    double m_dRegParam;
    
    Eigen::MatrixXd m_matFilters;
    Eigen::MatrixXd m_matPatterns;
    Eigen::MatrixXd m_matEvoked;
    bool m_bFitted;
};

//=============================================================================================================
/**
 * Surface Laplacian Spatial Filter
 * 
 * Implements surface Laplacian spatial filtering for EEG data.
 */
class DECODINGSHARED_EXPORT SurfaceLaplacianFilter : public BaseSpatialFilter
{
public:
    typedef std::shared_ptr<SurfaceLaplacianFilter> SPtr;
    typedef std::shared_ptr<const SurfaceLaplacianFilter> ConstSPtr;

    //=========================================================================================================
    /**
     * Constructs a SurfaceLaplacianFilter.
     *
     * @param[in] m_order       Order of the Laplacian (typically 4).
     * @param[in] lambda        Regularization parameter.
     */
    explicit SurfaceLaplacianFilter(int m_order = 4, double lambda = 1e-5);

    //=========================================================================================================
    /**
     * Destroys the SurfaceLaplacianFilter.
     */
    ~SurfaceLaplacianFilter() = default;

    //=========================================================================================================
    /**
     * Fit the surface Laplacian filter.
     * 
     * @param[in] epochs Vector of epochs data (not used, filter is data-independent).
     * @return true if successful.
     */
    bool fit(const std::vector<Eigen::MatrixXd>& epochs) override;

    //=========================================================================================================
    /**
     * Apply the surface Laplacian filter.
     * 
     * @param[in] epochs Vector of epochs data.
     * @return Filtered epochs.
     */
    std::vector<Eigen::MatrixXd> transform(const std::vector<Eigen::MatrixXd>& epochs) const override;

    //=========================================================================================================
    /**
     * Get the spatial filters (Laplacian matrix).
     */
    Eigen::MatrixXd getFilters() const override { return m_matLaplacian; }

    //=========================================================================================================
    /**
     * Set electrode positions for computing Laplacian.
     * 
     * @param[in] positions Electrode positions (n_channels x 3).
     */
    void setElectrodePositions(const Eigen::MatrixXd& positions);

private:
    //=========================================================================================================
    /**
     * Compute surface Laplacian matrix from electrode positions.
     */
    void computeLaplacianMatrix();

    //=========================================================================================================
    /**
     * Compute spherical spline interpolation matrix.
     */
    Eigen::MatrixXd computeSplineMatrix(const Eigen::MatrixXd& positions) const;

    int m_iOrder;
    double m_dLambda;
    
    Eigen::MatrixXd m_matPositions;
    Eigen::MatrixXd m_matLaplacian;
    bool m_bFitted;
};

} // namespace DECODINGLIB

#endif // SPATIAL_FILTERING_H