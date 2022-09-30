import torch
import numpy as np # linear algebra
import pandas as pd
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

# Load the scans in given folder path
def load_scan(path):
    print(f'Load scans from {path}...')
    #import pdb;pdb.set_trace()
    if os.listdir(path) == []:
        return None
    slices = [dicom.dcmread(os.path.join(path,i)) for i in os.listdir(path) if '.dcm' in i]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    if slice_thickness == 0:
        slice_thickness = np.abs(slices[1].ImagePositionPatient[2] - slices[2].ImagePositionPatient[2])
        if slice_thickness == 0:
            raise NotImplementedError
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    print('Transform to HU unit...')
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    print('Resampling...')
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def crop_image(img,tol=0):
    mask = img > tol
    img =  img[np.ix_([True]*img.shape[0],mask.any(0).any(1),mask.any(0).any(0))]
    return(img)
    
def lung_box(original, seg, return_coord = False):
    seg_temp = seg.copy()
    contours, hierarchy = cv2.findContours(seg_temp,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    all_boxes_y = []
    all_boxes_yh = []
    all_boxes_x = []
    all_boxes_xw = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        all_boxes_y.append(y)
        all_boxes_yh.append(y+h)
        all_boxes_x.append(x)
        all_boxes_xw.append(x+w)
    
    try:
        y = min(all_boxes_y)
        yh = max(all_boxes_yh)
        x = min(all_boxes_x)
        xw = max(all_boxes_xw)        
        seg_temp[y-5:yh+5, x-5:xw+5] = 1

        lung_bb = original.copy()
        lung_bb[seg_temp==0] = -1000
        if return_coord:
            return y, yh, x,xw
        else:
            return lung_bb, seg_temp
    except:
        lung_bb = original.copy()
        lung_bb[seg==0] = -1000
        if return_coord:
            return None,None,None,None
        else:
            return lung_bb, seg_temp
        

def largest_lung_box(pix_resampled,segmentation):
    b_y = float('Inf')
    b_yh = -float('Inf')
    b_x = float('Inf')
    b_xw = - float('Inf')
    for i in range(len(pix_resampled)):
        y,yh,x,xw = lung_box(pix_resampled[i], segmentation[i],True)
        try:
            if y < b_y:
                b_y = y
            if x < b_x:
                b_x = x
            if yh > b_yh:
                b_yh = yh
            if xw > b_xw:
                b_xw = xw
        except:
            pass
    return pix_resampled[:,b_y-1:b_yh+1, b_x-1:b_xw+1]


def pad_image(img, size):
    print('Pad images to size',size)
    c, old_image_height, old_image_width = img.shape

    new_image_width = size
    new_image_height = size
    
    if old_image_height > size or old_image_width > size:
        raise NotImplementedError
    
    result = np.full((c, new_image_height,new_image_width), -1000)

    ## compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[0:c,y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = img
    
    return result

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces,_,_ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def normalize(image, MIN_BOUND,MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image,PIXEL_MEAN):
    image = image - PIXEL_MEAN
    return image