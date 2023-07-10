#coding:utf8
import os
import nibabel as nib
import numpy as np
import time
import torch as th
import random
from sklearn.metrics import f1_score
import SimpleITK as sitk
import nibabel as nib 
import scipy

###
def get_JS(SR,GT):
    # JS : Jaccard similarity

    Inter = np.sum((SR+GT)==2)
    Union = np.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
###
def do(testFilename, resultFilename):
    """Main function"""
    testImage, resultImage = getImages(testFilename, resultFilename)
    
    dsc = getDSC(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    avd = getAVD(testImage, resultImage)    
    recall, f1 = getLesionDetection(testImage, resultImage)    

    return dsc,h95,avd,recall,f1
###
def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)
    
    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)
    
    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage,  0.5,  1.5, 1, 0) # WMH == 1    
    nonWMHImage     = sitk.BinaryThreshold(testImage,  1.5,  2.0, 0, 1) # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)
    
    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
        
    return maskedTestImage, bResultImage
    #return testImage,resultImage
###
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
###
def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
        
            
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
###
def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)
    
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    
    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return recall, f1    
###
def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
        
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
###
###
def generate_f1_and_f2(pred_3d,tru_3d):
    pre = pred_3d
    tru = tru_3d
    pree = np.zeros((pred_3d.shape[0], pred_3d.shape[1], pred_3d.shape[2]))
    truu = np.zeros((tru_3d.shape[0], tru_3d.shape[1], tru_3d.shape[2]))
    pree[pre==0] = 1
    truu[tru==0] = 1
    TP = TN = FP = FN = 0

    TP = np.sum(np.multiply(pre, tru) == 1)
    TN = np.sum(np.multiply(pree, truu) == 1)
    FP = np.sum(np.multiply(pre, truu) == 1)
    FN = np.sum(np.multiply(pree, tru) == 1)

    sensitivity=TP/(TP+FN)
    specificity=TN/(FP+TN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    #f1score=2*((precision*recall)/(precision+recall))
    f1score=f1_score(pred_3d.reshape(-1,1),tru_3d.reshape(-1,1),average='binary')
    f2score = ((1+4)*((precision*recall)+1e-15)/(4*precision+recall)+1e-15)
    return f1score,recall,precision,f2score
###
###
def iou_score(output, target):
    smooth = 1e-5

    intersection = np.sum(np.multiply(output, target) == 1)
    union = np.sum(output)+np.sum(target)-np.sum(np.multiply(output, target) == 1)
 
    return (intersection + smooth) / (union + smooth)
###

def dice_score(pred, targs):
    pred = (pred>0.5)
    return 2. * (pred*targs).sum() / (pred+targs+0.0000001).sum()
