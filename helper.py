#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:14:27 2022

@author: adugna
"""

import tensorflow as tf
import ee
import math
import json
from google.cloud import storage

###########################################
# S2 PREPROCESSING
###########################################
#These helper functions are adopted from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

CLOUD_FILTER = 100
CLD_PRB_THRESH = 30
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))



def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)



def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def NBRaddTimeline(image):
    nir = image.select('B8') 
    swir2 = image.select('B12')
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR')
    return image.addBands(nbr)

###########################################
# S1 PREPROCESSING
###########################################

#The Sentinel-1 helper files are adopted from Mullissa, A.; Vollrath, A.; Odongo-Braun, C.; Slagter, B.; Balling, J.; Gou, Y.; Gorelick, N.; Reiche, J. 
#Sentinel-1 SAR Backscatter Analysis Ready Data Preparation in Google Earth Engine. Remote Sens. 2021, 13, 1954. https://doi.org/10.3390/rs13101954

#The original codes are found in https://github.com/adugnag/gee_s1_ard
#-------------------------//
# convert linear to db and vice versa
#-------------------------//

def lin_to_db(image):
  """ Convert backscatter from linear to dB. """
  bandNames = image.bandNames().remove('angle')
  db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
  return image.addBands(db, None, True)

def lin_to_db2(image):
    """ Convert backscatter from linear to dB by removing the ratio band. """
    db = ee.Image(10).multiply(image.select(['VV', 'VH']).log10()).rename(['VV', 'VH'])
    return image.addBands(db)

#-------------------------//
# Add ratio bands
#-------------------------//

def add_ratio_lin(image):
    """ Adding ratio band for visualization """
    ratio = image.addBands(image.select('VV').divide(image.select('VH')).rename('VVVH_ratio'))
    return ratio.set('system:time_start', image.get('system:time_start'))

#---------------------------------------------------------------------------//
# Additional Border Noise Removal
#---------------------------------------------------------------------------//


def maskAngLT452(image):
   """ (mask out angles >= 45.23993) """
   ang = image.select(['angle'])
   return image.updateMask(ang.lt(45.23993)).set('system:time_start', image.get('system:time_start'))



def maskAngGT30(image):
   """ Function to mask out edges of images using angle.
    (mask out angles <= 30.63993) """
   ang = image.select(['angle'])
   return image.updateMask(ang.gt(30.63993)).set('system:time_start', image.get('system:time_start'))


def maskEdge(image):
   """ Remove edges.
   Source: Andreas Vollrath """
   mask = image.select(0).unitScale(-25, 5).multiply(255).toByte().connectedComponents(ee.Kernel.rectangle(1,1), 100)
   return image.updateMask(mask.select(0)).set('system:time_start', image.get('system:time_start')) 



def f_mask_edges(image):
  """ Mask edges. This function requires that the input image has one VH or VV band, and an 'angle' bands. """
  output = maskAngGT30(image)
  output = maskAngLT452(output)
  #output = maskEdge(output)
  return output.set('system:time_start', image.get('system:time_start'))


#---------------------------------------------------------------------------//
# 3.SPECKLE FILTERS
#---------------------------------------------------------------------------//

def leefilter(image,KERNEL_SIZE):
    """Lee Filter applied to one image. It is implemented as described in 
    J. S. Lee, “Digital image enhancement and noise filtering by use of local statistics,” 
    IEEE Pattern Anal. Machine Intell., vol. PAMI-2, pp. 165–168, Mar. 1980."""
        
    bandNames = image.bandNames().remove('angle')
  
    #S1-GRD images are multilooked 5 times in range
    enl = 5
    #Compute the speckle standard deviation
    eta = 1.0/math.sqrt(enl) 
    eta = ee.Image.constant(eta)

    #MMSE estimator
    #Neighbourhood mean and variance
    oneImg = ee.Image.constant(1)
    #Estimate stats
    reducers = ee.Reducer.mean().combine( \
                      reducer2= ee.Reducer.variance(), \
                      sharedInputs= True
                      )
    stats = image.select(bandNames).reduceNeighborhood( \
                      reducer= reducers, \
                          kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), \
                              optimization= 'window')
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_variance'))
        
    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    #Estimate weight 
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
  
    #if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    output = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(image.select(bandNames)))
    output = output.rename(bandNames)
    return image.addBands(output, None, True)


def boxcar(image, KERNEL_SIZE):
    """
    Apply boxcar filter on every image in the collection.
    Parameters
    ----------
    image : ee.Image
        Image to be filtered
    KERNEL_SIZE : positive odd integer
        Neighbourhood window size
    Returns
    -------
    ee.Image
        Filtered Image
    """
    bandNames = image.bandNames().remove('angle')
      #Define a boxcar kernel
    #kernel = ee.Kernel.square((KERNEL_SIZE/2), units='pixels', normalize=True)


    weights = ee.List.repeat(ee.List.repeat(1, KERNEL_SIZE),KERNEL_SIZE);
    kernel = ee.Kernel.fixed(KERNEL_SIZE,KERNEL_SIZE, weights, 1, 1, False);
     #Apply boxcar
    output = image.select(bandNames).convolve(kernel).rename(bandNames)
    return image.addBands(output, None, True)

def gammamap(image,KERNEL_SIZE): 
    
    """
    Gamma Maximum a-posterior Filter applied to one image. It is implemented as described in 
    Lopes A., Nezry, E., Touzi, R., and Laur, H., 1990.  
    Maximum A Posteriori Speckle Filtering and First Order texture Models in SAR Images.  
    International  Geoscience  and  Remote  Sensing  Symposium (IGARSS).
    Parameters
    ----------
    image : ee.Image
        Image to be filtered
    KERNEL_SIZE : positive odd integer
        Neighbourhood window size
    Returns
    -------
    ee.Image
        Filtered Image
    """
    enl = 5
    bandNames = image.bandNames().remove('angle')
    #local mean
    reducers = ee.Reducer.mean().combine( \
                      reducer2= ee.Reducer.stdDev(), \
                      sharedInputs= True
                      )
    stats = (image.select(bandNames).reduceNeighborhood( \
                      reducer= reducers, \
                          kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), \
                              optimization= 'window'))
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    stdDevBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_stdDev'))
        
    z = stats.select(meanBand)
    sigz = stats.select(stdDevBand)
    
    #local observed coefficient of variation
    ci = sigz.divide(z)
    #noise coefficient of variation (or noise sigma)
    cu = 1.0/math.sqrt(enl)
    #threshold for the observed coefficient of variation
    cmax = math.sqrt(2.0) * cu
    cu = ee.Image.constant(cu)
    cmax = ee.Image.constant(cmax)
    enlImg = ee.Image.constant(enl)
    oneImg = ee.Image.constant(1)
    twoImg = ee.Image.constant(2)

    alpha = oneImg.add(cu.pow(2)).divide(ci.pow(2).subtract(cu.pow(2)))

    #Implements the Gamma MAP filter described in equation 11 in Lopez et al. 1990
    q = image.select(bandNames).expression('z**2 * (z * alpha - enl - 1)**2 + 4 * alpha * enl * b() * z', { 'z': z,  'alpha':alpha,'enl': enl})
    rHat = z.multiply(alpha.subtract(enlImg).subtract(oneImg)).add(q.sqrt()).divide(twoImg.multiply(alpha))
  
    #if ci <= cu then its a homogenous region ->> boxcar filter
    zHat = (z.updateMask(ci.lte(cu))).rename(bandNames)
    #if cmax > ci > cu then its a textured medium ->> apply Gamma MAP filter
    rHat = (rHat.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))).rename(bandNames)
    #ci>cmax then its strong signal ->> retain
    x = image.select(bandNames).updateMask(ci.gte(cmax)).rename(bandNames)  
    #Merge
    output = ee.ImageCollection([zHat,rHat,x]).sum()
    return image.addBands(output, None, True)

def RefinedLee(image):
    """
    This filter is modified from the implementation by Guido Lemoine 
    Source: Lemoine et al. https://code.earthengine.google.com/5d1ed0a0f0417f098fdfd2fa137c3d0c
    Parameters
    ----------
    image: ee.Image
        Image to be filtered
    Returns
    -------
    result: ee.Image
        Filtered Image
    """

    bandNames = image.bandNames().remove('angle')

    def inner(b):

        img = image.select([b]);
    
        # img must be linear, i.e. not in dB!
        # Set up 3x3 kernels 
        weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
        kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False);
  
        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);
  
        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
  
        sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False);
  
        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel); 
        sample_var = variance3.neighborhoodToBands(sample_kernel);
  
        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
  
        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max());
  
        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient);
  
        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask);
  
        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5));
        directions = directions.addBands(directions.select(1).Not().multiply(6));
        directions = directions.addBands(directions.select(2).Not().multiply(7));
        directions = directions.addBands(directions.select(3).Not().multiply(8));
  
        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask);
  
        # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum());  
  
        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
  
        #Calculate localNoiseVariance
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);
  
        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));
  
        diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);
  
        rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False);
        diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False);
  
        # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
  
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
  
        # and add the bands for rotated kernels
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

  
        # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
        dir_mean = dir_mean.reduce(ee.Reducer.sum());
        dir_var = dir_var.reduce(ee.Reducer.sum());
  
        # A finally generate the filtered value
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
  
        b = varX.divide(dir_var)
        result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
  
        return result.arrayProject([0]).arrayFlatten([['sum']]).float()
    
    result = ee.ImageCollection(bandNames.map(inner)).toBands().rename(bandNames).copyProperties(image)
    
    return image.addBands(result, None, True)

def leesigma(image,KERNEL_SIZE):
    """
    Implements the improved lee sigma filter to one image. 
    It is implemented as described in, Lee, J.-S. Wen, J.-H. Ainsworth, T.L. Chen, K.-S. Chen, A.J. 
    Improved sigma filter for speckle filtering of SAR imagery. 
    IEEE Trans. Geosci. Remote Sens. 2009, 47, 202–213.
    Parameters
    ----------
    image : ee.Image
        Image to be filtered
    KERNEL_SIZE : positive odd integer
        Neighbourhood window size
    Returns
    -------
    ee.Image
        Filtered Image
    """

    #parameters
    Tk = ee.Image.constant(7) #number of bright pixels in a 3x3 window
    sigma = 0.9
    enl = 4
    target_kernel = 3
    bandNames = image.bandNames().remove('angle')
  
    #compute the 98 percentile intensity 
    z98 = ee.Dictionary(image.select(bandNames).reduceRegion(
                reducer= ee.Reducer.percentile([98]),
                geometry= image.geometry(),
                scale=10,
                maxPixels=1e13
            )).toImage()
  

    #select the strong scatterers to retain
    brightPixel = image.select(bandNames).gte(z98)
    K = brightPixel.reduceNeighborhood(ee.Reducer.countDistinctNonNull()
            ,ee.Kernel.square(target_kernel/2)) 
    retainPixel = K.gte(Tk)
  
  
    #compute the a-priori mean within a 3x3 local window
    #original noise standard deviation since the data is 5 look
    eta = 1.0/math.sqrt(enl) 
    eta = ee.Image.constant(eta)
    #MMSE applied to estimate the apriori mean
    reducers = ee.Reducer.mean().combine( \
                      reducer2= ee.Reducer.variance(), \
                      sharedInputs= True
                      )
    stats = image.select(bandNames).reduceNeighborhood( \
                      reducer= reducers, \
                          kernel= ee.Kernel.square(target_kernel/2,'pixels'), \
                              optimization= 'window')
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_variance'))
        
    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    
    oneImg = ee.Image.constant(1)
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
    xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)))
  
    #step 3: compute the sigma range
    #Lookup table (J.S.Lee et al 2009) for range and eta values for intensity (only 4 look is shown here)
    LUT = ee.Dictionary({0.5: ee.Dictionary({'I1': 0.694,'I2': 1.385,'eta': 0.1921}),
                                 0.6: ee.Dictionary({'I1': 0.630,'I2': 1.495,'eta': 0.2348}),
                                 0.7: ee.Dictionary({'I1': 0.560,'I2': 1.627,'eta': 0.2825}),
                                 0.8: ee.Dictionary({'I1': 0.480,'I2': 1.804,'eta': 0.3354}),
                                 0.9: ee.Dictionary({'I1': 0.378,'I2': 2.094,'eta': 0.3991}),
                                 0.95: ee.Dictionary({'I1': 0.302,'I2': 2.360,'eta': 0.4391})});
  
    #extract data from lookup
    sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
    I1 = sigmaImage.select('I1')
    I2 = sigmaImage.select('I2')
    #new speckle sigma
    nEta = sigmaImage.select('eta')
    #establish the sigma ranges
    I1 = I1.multiply(xTilde)
    I2 = I2.multiply(xTilde)
  
    #step 3: apply MMSE filter for pixels in the sigma range
    #MMSE estimator
    mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
    z = image.select(bandNames).updateMask(mask)
  
    stats = z.reduceNeighborhood( \
                      reducer= reducers, \
                          kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), \
                              optimization= 'window')
        
    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    
    
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)))
    b = varx.divide(varz)
    #if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))
  
    #remove the applied masks and merge the retained pixels and the filtered pixels
    xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
    output = ee.Image(xHat).rename(bandNames)
    return image.addBands(output, None, True)


def MonoTemporal_Filter(coll,KERNEL_SIZE, SPECKLE_FILTER) :
    """
    A wrapper function for monotemporal filter
    Parameters
    ----------
    coll : ee Image collection
        the image collection to be filtered
    KERNEL_SIZE : odd integer
        Spatial Neighbourhood window
    SPECKLE_FILTER : String
        Type of speckle filter
    Returns
    -------
    ee.ImageCollection
        An image collection where a mono-temporal filter is applied to each 
        image individually
    """
    def _filter(image):    
       if (SPECKLE_FILTER=='BOXCAR'):
          _filtered = boxcar(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='LEE'):
          _filtered = leefilter(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='GAMMA MAP'):
          _filtered = gammamap(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='REFINED LEE'):
          _filtered = RefinedLee(image)
       elif (SPECKLE_FILTER=='LEE SIGMA'):
          _filtered = leesigma(image, KERNEL_SIZE)
       return _filtered
    return coll.map(_filter)



def MultiTemporal_Filter(coll,KERNEL_SIZE, SPECKLE_FILTER,NR_OF_IMAGES):
    """ The following Multi-temporal speckle filters are implemented as described in
    S. Quegan and J. J. Yu, “Filtering of multichannel SAR images,” 
    IEEE Trans Geosci. Remote Sensing, vol. 39, Nov. 2001."""
  
    def Quegan(image) :
  
        """ this function will filter the collection used for the multi-temporal part
        it takes care of:
        - same image geometry (i.e relative orbit)
        - full overlap of image
        - amount of images taken for filtering 
            -- all before
           -- if not enough, images taken after the image to filter are added """
    
        def get_filtered_collection(image):
  
            #filter collection over are and by relative orbit
            s1_coll = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filterBounds(image.geometry()) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.Or(ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_stop')), \
                                     ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_start'))
                ))
      
            #a function that takes the image and checks for the overlap
            def check_overlap(_image):
                # get all S1 frames from this date intersecting with the image bounds                                 
                s1 = s1_coll.filterDate(_image.date(), _image.date().advance(1, 'day'))
                # intersect those images with the image to filter
                intersect = image.geometry().intersection(s1.geometry().dissolve(), 10)
                # check if intersect is sufficient
                valid_date = ee.Algorithms.If(intersect.area(10).divide(image.geometry().area(10)).gt(0.95), \
                                              _image.date().format('YYYY-MM-dd')
                                              )
                return ee.Feature(None, {'date': valid_date})
      
      
            # this function will pick up the acq dates for fully overlapping acquisitions before the image acquistion
            dates_before = s1_coll.filterDate('2014-01-01', image.date().advance(1, 'day')) \
                                    .sort('system:time_start', False).limit(5*NR_OF_IMAGES) \
                                    .map(check_overlap).distinct('date').aggregate_array('date')
    
            # if the images before are not enough, we add images from after the image acquisition 
            # this will only be the case at the beginning of S1 mission
            dates = (ee.List(ee.Algorithms.If( \
                                             dates_before.size().gte(NR_OF_IMAGES), \
                                                 dates_before.slice(0, NR_OF_IMAGES), \
                                                     s1_coll \
                                                         .filterDate(image.date(), '2100-01-01') \
                                                             .sort('system:time_start', True).limit(5*NR_OF_IMAGES) \
                                                                 .map(check_overlap) \
                                                                     .distinct('date') \
                                                                         .aggregate_array('date') \
                                                                             .cat(dates_before).distinct().sort().slice(0, NR_OF_IMAGES))))
    
            #now we re-filter the collection to get the right acquisitions for multi-temporal filtering
            return ee.ImageCollection(dates.map(lambda date: s1_coll.filterDate(date, ee.Date(date).advance(1,'day')).toList(s1_coll.size())).flatten())
      
          
  
        #we get our dedicated image collection for that image
        s1 = get_filtered_collection(image)
  
        bands = image.bandNames().remove('angle')
        s1 = s1.select(bands)
        meanBands = bands.map(lambda bandName: ee.String(bandName).cat('_mean'))
        ratioBands = bands.map(lambda bandName: ee.String(bandName).cat('_ratio'))
        count_img = s1.reduce(ee.Reducer.count())

        def inner(image):
            if (SPECKLE_FILTER=='BOXCAR'):
                _filtered = boxcar(image, KERNEL_SIZE).select(bands).rename(meanBands) 
            elif (SPECKLE_FILTER=='LEE'):
                _filtered = leefilter(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='GAMMA MAP'):
                _filtered = gammamap(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='REFINED LEE'):
                _filtered = RefinedLee(image).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='LEE SIGMA'):
                _filtered = leesigma(image, KERNEL_SIZE).select(bands).rename(meanBands)
    
            _ratio = image.select(bands).divide(_filtered).rename(ratioBands) 
            return _filtered.addBands(_ratio)

        isum = s1.map(inner).select(ratioBands).reduce(ee.Reducer.sum())
        filtered = inner(image).select(meanBands)
        divide = filtered.divide(count_img)
        output = divide.multiply(isum).rename(bands)

        return image.addBands(output, None, True)
    return coll.map(Quegan)


#---------------------------------------------------------------------------//
# Terrain Flattening
#---------------------------------------------------------------------------//

def slope_correction(collection, TERRAIN_FLATTENING_MODEL \
                                              ,DEM \
                                              ,TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER):
  
  ninetyRad = ee.Image.constant(90).multiply(math.pi/180)

  def _volumetric_model_SCF(theta_iRad, alpha_rRad):

      # Volume model
      nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
      denominator = (ninetyRad.subtract(theta_iRad)).tan()
      return nominator.divide(denominator)

  def _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
      # Surface model
      nominator = (ninetyRad.subtract(theta_iRad)).cos()
      denominator = alpha_azRad.cos() \
        .multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos())
      return nominator.divide(denominator)
  
  def _erode(image, distance):
      #buffer function (thanks Noel)
      
      d = (image.Not().unmask(1)
          .fastDistanceTransform(30).sqrt()
          .multiply(ee.Image.pixelArea().sqrt()))
    
      return image.updateMask(d.gt(distance))    
  
  def _masking(alpha_rRad, theta_iRad, buffer):
      # calculate masks
      # layover, where slope > radar viewing angle
      layover = alpha_rRad.lt(theta_iRad).rename('layover')
      # shadow
      shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
      # combine layover and shadow
      mask = layover.And(shadow)
      # add buffer to final mask
      if (buffer > 0):
          mask = _erode(mask, buffer)
      return mask.rename('no_data_mask')

  def _correct(image):
        
      bandNames = image.bandNames()
      #get the image geometry and projection
      #geom = image.geometry()
      #proj = image.select(1).projection()
        
      #calculate the look direction
      heading = (ee.Terrain.aspect(image.select('angle'))
                                     .reduceRegion(ee.Reducer.mean(),image.geometry(),1000)
                                     .get('aspect'))
        
      #the numbering follows the article chapters
      #2.1.1 Radar geometry 
      theta_iRad = image.select('angle').multiply(math.pi/180)
      phi_iRad = ee.Image.constant(heading).multiply(math.pi/180)
        
      #2.1.2 Terrain geometry
      alpha_sRad = ee.Terrain.slope(DEM).select('slope').multiply(math.pi/180)#.reproject(proj).clip(geom)
      phi_sRad = ee.Terrain.aspect(DEM).select('aspect').multiply(math.pi/180)#.reproject(proj).clip(geom)
        
      #2.1.3 Model geometry
      #reduce to 3 angle
      phi_rRad = phi_iRad.subtract(phi_sRad)

      #slope steepness in range (eq. 2)
      alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

      #slope steepness in azimuth (eq 3)
      alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

      #2.2 
      #Gamma_nought
      gamma0 = image.divide(theta_iRad.cos())

      if (TERRAIN_FLATTENING_MODEL == 'VOLUME'):
            #Volumetric Model
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)
        
      if (TERRAIN_FLATTENING_MODEL == 'DIRECT'):
            scf = _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)
        
       #apply model for Gamm0
      gamma0_flat = gamma0.divide(scf)

      #get Layover/Shadow mask
      mask = _masking(alpha_rRad, theta_iRad, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER)
      output = gamma0_flat.updateMask(mask).rename(bandNames).copyProperties(image)       
      output = ee.Image(output).addBands(image.select('angle'))
        
      return output.set('system:time_start', image.get('system:time_start'))  
  return collection.map(_correct)


###########################################
# S1 PREPROCESSING
###########################################

def s1_preproc(params):
    """
    Applies preprocessing to a collection of S1 images to return an analysis ready sentinel-1 data. 

    """
    
    APPLY_BORDER_NOISE_CORRECTION = params['APPLY_BORDER_NOISE_CORRECTION']
    APPLY_TERRAIN_FLATTENING = params['APPLY_TERRAIN_FLATTENING']
    APPLY_SPECKLE_FILTERING = params['APPLY_SPECKLE_FILTERING']
    POLARIZATION = params['POLARIZATION']
    SPECKLE_FILTER_FRAMEWORK = params['SPECKLE_FILTER_FRAMEWORK']
    SPECKLE_FILTER = params['SPECKLE_FILTER']
    SPECKLE_FILTER_KERNEL_SIZE = params['SPECKLE_FILTER_KERNEL_SIZE']
    NR_OF_IMAGES = params['NR_OF_IMAGES']
    TERRAIN_FLATTENING_MODEL = params['TERRAIN_FLATTENING_MODEL']
    DEM = params['DEM']
    SATELLITE = params['SATELLITE']
    TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER = params['TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER']
    FORMAT = params['FORMAT']
    START_DATE = params['START_DATE']
    STOP_DATE = params['STOP_DATE']
    ORBIT = params['ORBIT']
    RELATIVE_ORBIT_NUMBER = params['RELATIVE_ORBIT_NUMBER']
    ROI = params['ROI']
    CLIP_TO_ROI = params['CLIP_TO_ROI']
    
    if APPLY_BORDER_NOISE_CORRECTION is None: APPLY_BORDER_NOISE_CORRECTION = True    
    if APPLY_TERRAIN_FLATTENING is None: APPLY_TERRAIN_FLATTENING = True       
    if APPLY_SPECKLE_FILTERING is None: APPLY_SPECKLE_FILTERING = True         
    if POLARIZATION is None: POLARIZATION = 'VVVH'
    if SPECKLE_FILTER_FRAMEWORK is None: SPECKLE_FILTER_FRAMEWORK = 'MULTI BOXCAR' 
    if SPECKLE_FILTER is None: SPECKLE_FILTER = 'GAMMA MAP' 
    if SPECKLE_FILTER_KERNEL_SIZE is None: SPECKLE_FILTER_KERNEL_SIZE = 7
    if NR_OF_IMAGES is None: NR_OF_IMAGES = 10
    if TERRAIN_FLATTENING_MODEL is None: TERRAIN_FLATTENING_MODEL = 'VOLUME' 
    if TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER is None: TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER = 0 
    if FORMAT is None: FORMAT = 'DB' 
    if ORBIT is None: ORBIT = 'DESCENDING' 
    
    
    pol_required = ['VV', 'VH', 'VVVH']
    if (POLARIZATION not in pol_required):
        raise ValueError("ERROR!!! Parameter POLARIZATION not correctly defined")

    
    orbit_required = ['ASCENDING', 'DESCENDING', 'BOTH']
    if (ORBIT not in orbit_required):
        raise ValueError("ERROR!!! Parameter ORBIT not correctly defined")

    model_required = ['DIRECT', 'VOLUME']
    if (TERRAIN_FLATTENING_MODEL not in model_required):
        raise ValueError("ERROR!!! Parameter TERRAIN_FLATTENING_MODEL not correctly defined")

    format_required = ['LINEAR', 'DB']
    if (FORMAT not in format_required):
        raise ValueError("ERROR!!! FORMAT not correctly defined")
        
    frame_needed = ['MONO', 'MULTI']
    if (SPECKLE_FILTER_FRAMEWORK not in frame_needed):
        raise ValueError("ERROR!!! SPECKLE_FILTER_FRAMEWORK not correctly defined")

    format_sfilter = ['BOXCAR', 'LEE', 'GAMMA MAP' \
              ,'REFINED LEE', 'LEE SIGMA']
    if (SPECKLE_FILTER not in format_sfilter):
        raise ValueError("ERROR!!! SPECKLE_FILTER not correctly defined")

    if (TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER < 0):
        raise ValueError("ERROR!!! TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER not correctly defined")

    if (SPECKLE_FILTER_KERNEL_SIZE <= 0):
        raise ValueError("ERROR!!! SPECKLE_FILTER_KERNEL_SIZE not correctly defined")
    
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                .filter(ee.Filter.eq('resolution_meters', 10)) \
                .filterDate(START_DATE, STOP_DATE) \
                .filterBounds(ROI)
    

        # select orbit
    if (ORBIT != 'BOTH'):
      s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', ORBIT))
   
    if (RELATIVE_ORBIT_NUMBER != 'ANY'): 
      s1 =  s1.filter(ee.Filter.eq('relativeOrbitNumber_start', RELATIVE_ORBIT_NUMBER)) 

   
    if (SATELLITE=='A' or SATELLITE=='B' ):
        s1 = s1.filter(ee.Filter.eq('platform_number', SATELLITE))
     
    
    if (POLARIZATION == 'VV'):
      s1 = s1.select(['VV','angle'])
    elif (POLARIZATION == 'VH'):
      s1 = s1.select(['VH','angle'])
    elif (POLARIZATION == 'VVVH'):
      s1 = s1.select(['VV','VH','angle'])  
      
    
    # clip image to roi
    if (CLIP_TO_ROI):
        s1 = s1.map(lambda image: image.clip(ROI))
    
    
    if (APPLY_BORDER_NOISE_CORRECTION == True):
      s1_1 = s1.map(f_mask_edges)
    else : 
      s1_1 = s1
          
    
    if (APPLY_SPECKLE_FILTERING) :
        if (SPECKLE_FILTER_FRAMEWORK == 'MONO') :
            s1_1 = ee.ImageCollection(MonoTemporal_Filter(s1_1, SPECKLE_FILTER_KERNEL_SIZE, SPECKLE_FILTER ))
        else :
            s1_1 = ee.ImageCollection(MultiTemporal_Filter(s1_1, SPECKLE_FILTER_KERNEL_SIZE, SPECKLE_FILTER,NR_OF_IMAGES ));  

    
    if (APPLY_TERRAIN_FLATTENING == True):
        s1_1 = slope_correction(s1_1 \
                              , TERRAIN_FLATTENING_MODEL \
                              ,DEM \
                              ,TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER)
              
    
    if (FORMAT == 'DB'):
        s1_1 = s1_1.map(lin_to_db)
        
        
    return s1_1

###########################################
# DEEP LEARNING HELPERS
###########################################

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        #attention takes three inputs: queries, keys, and values,
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)
    def attention(self, query, key, value):
        #use the product between the queries and the keys 
        #to know "how much" each element is the sequence is important with the rest
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        #resulting vector, score is divided by a scaling factor based on the size of the embedding
        #scaling fcator is square root of the embeding dimension
        scaled_score = score / tf.math.sqrt(dim_key)
        #the attention scaled_score is then softmaxed
        weights = tf.nn.softmax(scaled_score, axis=-1)
        #Attention(Q, K, V ) = softmax[(QK)/√dim_key]V
        output = tf.matmul(weights, value)
        return output, weights 
        
    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, inputs):

        batch_size = tf.shape(inputs)[0]
        #MSA takes the queries, keys, and values  as input from the   
        #previous layer and projects them using the 3 linear layers.
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        #self attention of different heads are concatenated  
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.5,  **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        # Transformer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="gelu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
    def call(self, inputs, training):
        out1 = self.layernorm1(inputs)       
        attention_output = self.multiheadselfattention(out1)
        attention_output = self.dropout1(attention_output, training=training)       
        out2 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask



def get_dataset(TRAIN_FILE_PATH, TEST_FILE_PATH, VALIDATION_FILE_PATH, FEATURES_DICT1, FEATURES_DICT2, batch_size):
    
  def parse_tfrecord(example_proto):
    parsed_features1 = tf.io.parse_single_example(example_proto, FEATURES_DICT1)
    parsed_features2 = tf.io.parse_single_example(example_proto, FEATURES_DICT2)

    labels = parsed_features1.pop('Label')
    return parsed_features1, parsed_features2, tf.cast(labels, tf.int32)


  def to_tuple(inputs1, inputs2, label):
    data1 = tf.transpose(list(inputs1.values()))
    data2 = tf.transpose(list(inputs2.values()))
 
    return (data1,data2), label
    
  # Load datasets from the files. whole
  train_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type='GZIP')
  test_dataset = tf.data.TFRecordDataset(TEST_FILE_PATH, compression_type='GZIP')
  validation_dataset = tf.data.TFRecordDataset(VALIDATION_FILE_PATH, compression_type='GZIP')

  # Compute the size of the shuffle buffer.  We can get away with this
  # because it's a small dataset, but watch out with larger datasets.
  train_size = 0
  for _ in iter(train_dataset):
    train_size+=1

  # Map the functions over the datasets to parse and convert to tuples.
  train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=4)
  train_dataset = train_dataset.map(to_tuple, num_parallel_calls=4)
  train_dataset = train_dataset.shuffle(train_size).batch(batch_size)

  test_dataset = test_dataset.map(parse_tfrecord, num_parallel_calls=4)
  test_dataset = test_dataset.map(to_tuple, num_parallel_calls=4)
  test_dataset = test_dataset.batch(batch_size)

  validation_dataset = validation_dataset.map(parse_tfrecord, num_parallel_calls=4)
  validation_dataset = validation_dataset.map(to_tuple, num_parallel_calls=4)
  validation_dataset = validation_dataset.batch(batch_size)
    
  return train_dataset, validation_dataset, test_dataset

    
def model_builder(hp):

  
  hp_units = hp.Int('filters', min_value=32, max_value=1024, step=32)
  lstm_units = hp.Int('lstm units', min_value=32, max_value=128, step=32)

  input_shapes1 = [None, 2]
  inputs1 = tf.keras.layers.Input(shape=input_shapes1)
  x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_units))(inputs1)
  x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x1)
  x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu'))(x1)
  x1 = PositionalEmbedding(sequence_length=37, output_dim=hp_units)(x1)
  x1 = TransformerBlock(embed_dim=hp_units, ff_dim=lstm_units, num_heads = 4)(x1) 
  x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
  x1 = tf.keras.layers.Dense(hp_units*8)(x1)
  x1 = tf.keras.layers.BatchNormalization()(x1)
  x1 = tf.keras.layers.Activation('relu')(x1)
  x1 = tf.keras.layers.Dropout(0.5)(x1)
  x1 = tf.keras.layers.Dense(hp_units*4)(x1)
  x1 = tf.keras.layers.BatchNormalization()(x1)
  x1 = tf.keras.layers.Activation('relu')(x1)
  x1 = tf.keras.layers.Dropout(0.5)(x1)
  outputs1 = tf.keras.layers.Dense(1, activation='sigmoid')(x1)

  input_shapes2 = [None, 1]
  inputs2 = tf.keras.layers.Input(shape=input_shapes2)
  x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hp_units))(inputs2)
  x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x2)
  x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu'))(x2)
  x2 = PositionalEmbedding(sequence_length=25, output_dim=hp_units)(x2)
  x2 = TransformerBlock(embed_dim=hp_units, ff_dim=lstm_units, num_heads = 4)(x2) 
  x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
  x2 = tf.keras.layers.Dense(hp_units*8)(x2)
  x2 = tf.keras.layers.BatchNormalization()(x2)
  x2 = tf.keras.layers.Activation('relu')(x2)
  x2 = tf.keras.layers.Dropout(0.5)(x2)
  x2 = tf.keras.layers.Dense(hp_units*4)(x2)
  x2 = tf.keras.layers.BatchNormalization()(x2)
  x2 = tf.keras.layers.Activation('relu')(x2)
  x2 = tf.keras.layers.Dropout(0.5)(x2)
  outputs2 = tf.keras.layers.Dense(1, activation='sigmoid')(x2)

  outputs = tf.keras.layers.multiply([outputs1,outputs2])

  model = tf.keras.models.Model(inputs=[inputs1,inputs2], outputs=[outputs])
  #model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy']) #'sparse_categorical_focal_loss'


  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

def export_test_image(fmap,image,params):
  image_export_options2 = {
    'patchDimensions': [64, 64],
    'maxFileSize': 104857600,
    'compressed': True,
    'tensorDepths':fmap
  }

  # Setup the task.
  image_task2 = ee.batch.Export.image.toCloudStorage(
    image=image,
    description='Image Export',
    fileNamePrefix=params['IMAGE_FILE_PREFIX'],
    bucket=params['OUTPUT_BUCKET'],
    scale=10,
    fileFormat='TFRecord',
    region=params['ROI'],
    maxPixels=1e13,
    formatOptions=image_export_options2,
  )

  image_task2.start()


def prepTestData(mixer,BANDS,image_files_list,NUM_IMAGES):
  # Get relevant info from the JSON mixer file.
  patch_width = mixer['patchDimensions'][0]
  patch_height = mixer['patchDimensions'][1]
  #patches = mixer['totalPatches']


  patch_dimensions_flat = [NUM_IMAGES, patch_width * patch_height]

  # Note that the tensors are in the shape of a patch, one patch for each band.
  image_columns = [tf.io.FixedLenFeature(shape=patch_dimensions_flat, dtype=tf.float32) for k in BANDS]

  # Parsing dictionary.
  image_features_dict = dict(zip(BANDS, image_columns))

  print(image_features_dict)

  # Note that you can make one dataset from many files by specifying a list.
  image_dataset = tf.data.TFRecordDataset(image_files_list, compression_type='GZIP')

  # Parsing function.
  def parse_image1(example_proto):
    return tf.io.parse_single_example(example_proto, image_features_dict)

  # Parse the data into tensors, one long tensor per patch.
  image_dataset = image_dataset.map(parse_image1, num_parallel_calls=5)
  print(image_dataset)

  # transpose function.
  def tp_image(tf_Data):
    for k, v in tf_Data.items():
      tf_Data[k] = tf.transpose(v)
    return tf_Data

  # tranpose the input tensors
  image_dataset = image_dataset.map(tp_image)
  print(image_dataset)

  # Break our long tensors into many little ones.
  image_dataset = image_dataset.flat_map(
    lambda features: tf.data.Dataset.from_tensor_slices(features)
  )
  print(image_dataset)

  # Turn the dictionary in each record into a tuple without a label.
  image_dataset = image_dataset.map(
    #lambda data_dict:tf.reshape(tensor=(tf.transpose(list(data_dict.values())), ), shape=[22,1])
    lambda data_dict: tf.transpose(list(data_dict.values()),)
    )
  print(image_dataset)

  # Turn each patch into a batch.
  image_dataset = image_dataset.batch(patch_width * patch_height)

  print(image_dataset)

  return image_dataset
  
def writeTfrecord(OUTPUT_IMAGE_FILE,OUTPUT_ASSET_ID,predictions, mixer):
# Instantiate the tfrecrod writer.
  writer = tf.io.TFRecordWriter(OUTPUT_IMAGE_FILE)

  patch = [[]]
  cur_patch = 1
  for prediction in predictions:
    patch[0].append(tf.argmax(prediction))
  # Once we've seen a patches-worth of class_ids...
    if (len(patch[0]) == mixer['patchDimensions'][0] * mixer['patchDimensions'][1]):
      print('Done with patch ' + str(cur_patch) + ' of ' + str(mixer['totalPatches']) + '...')
      # Create an example
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
              'fnf': tf.train.Feature(int64_list=tf.train.Int64List(value=patch[0]))
              }
          )
      )
      # Write the example to the file and clear our patch array so it's ready for
      # another batch of class ids
      writer.write(example.SerializeToString())
      patch = [[]]
      cur_patch += 1

  writer.close()
