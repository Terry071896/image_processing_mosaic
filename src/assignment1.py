# %% [markdown]
# # Project 1 – Images, intensities, and histograms
# Author: Terry Cox
# 
# ### Instructions:
# 
# **Goals:**  This project will introduce everyone to the basics of manipulating and processing images with python, such as image I/O, display, etc.  It will also cover basic histogram building and analysis, thresholding, region-based analysis, and histogram-based processing of images. 
# 
# The project must be done in python.  Students may use the matplotlib, numpy, and skimage image libraries - EXCEPT where otherwise noted.  Students will submit all of their code (either as executable python files or Jupyter notebooks).  
# 
# 1. **Build a histogram:**  Use the built-in numpy function to compute the histogram of an image.  Display the resulting histogram using a bar chart from matplotlib.   Display histograms for a couple of different images and describe how they relate to what you see in the image (e.g. what regions/objects are what part of the histogram).  Thresholding (below) can help with this.   Show examples with different ranges and bin sizes. 
# 
# 2. **Regions and components:**  Define a function that performs double-sided (high and low) thresholding on images to define regions, visualize results (and histograms) on several images.   Perform flood fill and connected component on these thresholded images.  Remove connected components (change them to the background label) that are smaller than a certain size (you specify).  Visualize the results as a color image (different colors for different regions).  
# 
# 3. **Otsu thresholding:** Write (from scratch, using general numpy functions – no using built-in skimage or other toolkits) a function that performs Otsu thresholding on a grey-scale image.   Show results for different images and comment on this in relation to the histogram.  EXTRA CREDIT (10pts): Write (from scratch) a function that does Otsu three-class thresholding on a greyscale image.  Show and discuss results - you can find a good example/test image here: three-class-test.png.
# 
# 4. **Histogram equalization:**  Perform histogram equalization (using the built-in routines in skimage) on a selection of images, show the histograms before and after equalization, and comment on the visual results.   Perform adaptive and/or local histogram equalization on a variety of images.  Identify all important parameters, experiment with (vary) those parameters, and report results on selection of images of different types (photos, medical, etc.).
# Some example images are given here Download here, but you should also go and find some others to experiment with and report on.

# %%
import numpy as np
import matplotlib.pyplot as plt
import glob
import skimage
from collections import Counter
import matplotlib
from itertools import combinations
import os

image_dir = os.path.join(os.path.abspath(""), 'images/*')
print(image_dir)
image_paths = glob.glob(image_dir)
print(image_paths)

# %%
def rgb_to_grey(file, weights=np.array([0.2989, 0.5870, 0.1140])):
    """Loads image and transforms to grey scale 2-D image.

    Args:
        file (str): Image file (png, tiff, jpeg, or whatever file matplotlib.pyplot.imread can read)
        weights (np.array): A vector of length 3 convert 3D image to 2D. Default is standard weights for grayscale conversion np.array([0.2989, 0.5870, 0.1140])

    Returns:
        np.array, np.array: 2D numpy array
    """
    img = plt.imread(file) # reads image to narray
    img = (img-np.min(img))
    img = img/np.max(img)
    #print(file, img.shape) # prints file and shape
    if len(img.shape) == 3: # checks to see if there image is 3D 
        if img.shape[-1] > 3: # if there are more bands than rgb, choose just the rgb matricies
            img = img[:,:,:3]
        return img @ weights # dot product weights
    else:
        return img # return image if only 2D

# %% [markdown]
# ### Part 1: 
# 
# Below I show an image next to its histograms with bins 10, 50, and 100.  When comparing the different binned histograms, it quickly becomes apparent that the more bins there are the more "regions" appear.  For example, in the 10 bin histogram below, there seems to be only one region.  But in the 100 bin histogram, there seems to be distinctly 2 and possibly a third.

# %%
def image_histogram(image, n_bins=10, size=(15,5), title='Image Histogram', threshold=None):
    if isinstance(image, str):
        image = rgb_to_grey(image)
    else:
        image = (image-np.min(image))
        image = image/np.max(image)
        
    bins = np.array(list(range(n_bins)))/(n_bins)
    hist, bins = np.histogram(image.reshape(-1,1), bins=bins)
    
    x = bins[:-1]+np.diff(bins)/2

    shape = (1,2)
    f, axarr = plt.subplots(1,2, figsize=size)
    axarr[0].imshow(image, cmap='gray')
    axarr[1].bar(x,hist/np.sum(hist),width = 1/n_bins)
    if threshold is not None:
        colors = ['red', 'blue', 'green', 'black', 'yellow', 'orange']
        try:
            for i, x in enumerate(threshold):
                for t in x:
                    axarr[1].vlines(t, 0, np.max(hist/np.sum(hist)), colors=colors[i])
        except:
            try:
                for t in threshold:
                    axarr[1].vlines(t, 0, np.max(hist/np.sum(hist)), colors='red')
            except:
                axarr[1].vlines(threshold, 0, np.max(hist/np.sum(hist)), colors='red')
    axarr[1].set_xlabel('intensity')
    axarr[1].set_ylabel('probability')
    plt.suptitle(title)
    plt.show()


# %%
print('PART 1:')
for n_bins in [10,50,100]:
    image_histogram(image=image_paths[0], n_bins=n_bins, title='Image Histogram: %s bins'%n_bins)

# %% [markdown]
# Below looks at different images with different histograms, some of which have several regions and others that do not.  Some might have a large spike in the histogram for a certain bin and others are a bit more spread out depending on the contrast of the image.

# %%
for img_path in glob.glob(image_dir):
    image_histogram(img_path, n_bins=50, title='Image Histogram: %s'%img_path)

# %% [markdown]
# ## Part 2:
# 
# Regions and components:  Define a function that performs double-sided (high and low) thresholding on images to define regions, visualize results (and histograms) on several images.   Perform flood fill and connected component on these thresholded images – you may use functions from skimage, e.g. skimage.measure.label().  Remove connected components (change them to the background label) that are smaller than a certain size (you specify).  Visualize the results as a color image (different colors for different regions).  

# %%
def high_low_thresholding(img, threshold=0.5):
    if np.max(img) > 1:
        img = (img-np.min(img))
        img = img/np.max(img)
    img2 = img.copy()
    try:
        last_t = 0
        for i, t in enumerate(threshold):
            img2[(img >= last_t) & (img < t)] = i
            last_t = t
        img2[(img >= t)] = i+1
            
    except:
        img2[img > threshold] = 1
        img2[img <= threshold] = 0
    return img2

def connected_components(img, connectivity=2, background=0, threshold=0.05):
    connections = skimage.measure.label(img, connectivity=connectivity, background=background)
    groups = dict(Counter(connections.reshape(-1)))
    total_pixs = np.sum(list(groups.values()))
    if threshold < 1:
        counter = 1
        for k, v in groups.items():
            if v/total_pixs < threshold:
                connections[connections == k] = background
            else:
                connections[connections == k] = counter
                counter +=1
    elif threshold > 1:
        counter = 1
        for k, v in groups.items():
            if v < threshold:
                connections[connections == k] = background
            else:
                connections[connections == k] = counter
                counter +=1
    # groups = dict(Counter(connections.reshape(-1)))
    return connections

def connections_to_color(img, background=0):
    shape = img.shape
    groups = dict(Counter(connections.reshape(-1)))
    max = np.max(list(groups.keys()))
    color_img = np.zeros((shape[0], shape[1], 3))
    for i in range(len(img)):
        for j in range(len(img[0])):
            k = img[i][j]
            p = k/max
            if k == background:
                rgb = [0,0,0]
            else:
                # https://stackoverflow.com/questions/35425476/module-with-extensive-list-of-rgb-colors-to-create-dictionary
                rgb = matplotlib.colors.to_rgb(list(matplotlib.colors.cnames.values())[::-1][k%len(list(matplotlib.colors.cnames.values()))])
            #print(rgb)
            for a in range(3):
                color_img[i][j][a] = rgb[a]
    return color_img

print('PART 2:')
threshold = 0.1
n_bins = 50
cc_thresh = 100
background = 1
connectivity = 2

img = rgb_to_grey(image_paths[5])


image_histogram(img, n_bins=n_bins, threshold=threshold)
print('thresholding...')
cats = high_low_thresholding(img, threshold=threshold)

#connections = skimage.measure.label(cats, connectivity=2, background=-1)
print('connecting...')
connections = connected_components(cats, connectivity=connectivity, threshold=cc_thresh, background=background)
print('coloring...')
color = connections_to_color(connections, background=background)


shape = (1,2)
f, axarr = plt.subplots(1,2, figsize=(15,15))
axarr[0].imshow(cats, cmap='gray')
axarr[1].imshow(color)
axarr[0].set_title('Thresholded: %s'%threshold)
axarr[1].set_title('Connected Component Groups')
#plt.suptitle('Thresholded and Connected Component Groups')
plt.show()


# %% [markdown]
# ## Part 3
# 
# Otsu thresholding: Write (from scratch, using general numpy functions – no using built-in skimage or other toolkits) a function that performs Otsu thresholding on a grey-scale image.   Show results for different images and comment on this in relation to the histogram.  EXTRA CREDIT (10pts): Write (from scratch) a function that does Otsu three-class thresholding on a greyscale image.  Show and discuss results - you can find a good example/test image here: three-class-test.png.  

# %%
def otsu_step(img, threshold=0.5):
    if isinstance(threshold, float):
        threshold = [threshold]
    else:
        threshold = list(threshold)
        threshold.sort()

    sig_b = 0
    img_mean = np.mean(img)
    img_size = img.shape[0]*img.shape[1]
    last_t = np.min(img)
    for t in threshold:
        img_t = img[(img > last_t) & (img <= t)]
        sig_b += (len(img_t)/img_size)*((np.mean(img_t)-img_mean)**2)
        last_t = t
    img_t = img[(img > threshold[-1])]
    sig_b += (len(img_t)/img_size)*((np.mean(img_t)-img_mean)**2)
    return np.sqrt(sig_b)

def otsu(img, n_thresh=1, step=0.02):
    if isinstance(n_thresh, int):
        thresholds = []
        counter = 0
        for i in range(n_thresh):
            counter = counter+(i+1)/(n_thresh+1)
            thresholds.append(counter)
    elif isinstance(n_thresh, list):
        thresholds = n_thresh
        thresholds.sort()
    else:
        print('n_thresh must be int or list. Returning None.')

    img = (img-np.min(img))
    img = img/np.max(img)
    thresholds = list(set([tuple(np.sort(x)) for x in combinations(np.arange(step,1-step,step=step), n_thresh)]))
    scores = [otsu_step(img, threshold=list(x)) for x in thresholds]
    return thresholds[np.argmax(scores)]

print('PART 3:')


# %%
for image_path in image_paths:
    print("--------------------------------", image_path, "--------------------------------")
    img = rgb_to_grey(image_path)
    thresholds_2 = otsu(img, n_thresh=2, step=0.02)
    thresholds_1 = otsu(img, n_thresh=1, step=0.02)
    image_histogram(img, n_bins=50, threshold=(thresholds_1, thresholds_2))
    groups_2 = high_low_thresholding(img=img, threshold=thresholds_2)
    groups_1 = high_low_thresholding(img=img, threshold=thresholds_1)

    shape = (1,2)
    f, axarr = plt.subplots(1,2, figsize=(10,10))
    axarr[0].imshow(groups_1, cmap='gray')
    axarr[1].imshow(groups_2, cmap='gray')
    axarr[0].set_title('2 Groups (red line)')
    axarr[1].set_title('3 Groups (blue line)')
    plt.show()

# %% [markdown]
# The above code and images show the Otsu algorithm for both 2 and 3 groupings over many images.  Based on the results from the images, typically we can see that the thresholds gravitate towards the "vallies" of the histograms.  If there are no vallies between peaks, it attemts to split the distribution in half.  It is in this case that it appears to have the "wrong" number of groups to decypher.  Ideally, we would want to choose the appropriate number of group for the approrate image.  I would propose to use the least number of groups possible that also has a threshold in at least one valley.  The picture of the houndog is a good example of why we might think there is on 2 groups with 2 peaks, but the single threshold is actually missing the valley between the two peaks because the distribution closest to 1 is skewed to the right and therefore moving the threshold to the right.  The correct number of thresholds seems to be 2 allowing a 3 group to split the skew and the 2 distributions.

# %% [markdown]
# ## Part 4
# 
# Histogram equalization:  Perform histogram equalization (using the built-in routines in skimage) on a selection of images, show the histograms before and after equalization, and comment on the visual results.   Perform adaptive and/or local histogram equalization on a variety of images.  Identify all important parameters, experiment with (vary) those parameters, and report results on selection of images of different types (photos, medical, etc.).

# %%
def equalization_comparision(img_path, kernel_size=None, clip_limit=0.1):
    img = rgb_to_grey(img_path)
    # Next couple lines taken from: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
    # Equalization
    img_eq = skimage.exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = skimage.exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)

    print("--------------------------------", img_path, "--------------------------------")
    image_histogram(img, n_bins=50, threshold=None, title='Original Grayscale Image: %s'%img_path)

    image_histogram(img_eq, n_bins=50, threshold=None, title='Histogram Equalization: %s'%img_path)

    image_histogram(img_adapteq, n_bins=50, threshold=None, title='Adaptive Histogram Equalization: %s'%img_path)

print('PART 4:')
for img_path in image_paths:
    equalization_comparision(img_path=img_path)

# %% [markdown]
# Looking at the histograms of the orignial, equalization, and adaptive equalization there are many differences between the images and the histograms themselves.  The histogram equalization transformation does a pretty good job on images that have close to normal distributions.  It basically will take the distribution and "equalize" it to where the distribution is now flat.  It seems to struggle with images that contain a high contrast or have a short range as it appears to wash out details.  For example, figures three-class-example.png and xray.png (medical images) have high contrast with small ranges.  Therefore the equalization washes out details.  This is not the same when it comes to the adaptive histogram, the localization of the equalization helps significantly when it comes to inhanceing local details.  These medical images have been transformed to where it is much easier to see details (especially in the xray of the hand).  As for images with a more normal distibuition, we can still see details, but background also appears to show more details as well.  For example, the image me_and_blu.png has a nice histogram equalization, but the adaptive histogram equalization show too much details around clouds and inhances details of the sun rays.  This makes the image a bit strange.  To help improve the adaptive histogram equalization method for images like this, it is work playing around with the parameters to get the best image.  Below shows the "help" function of the parameters of skimage for adaptive histogram equalizaton:
# ```{json}
# kernel_size : int or array_like, optional
#     Defines the shape of contextual regions used in the algorithm. If iterable is passed, it must have the same number of elements as image.ndim (without color channel). If integer, it is broadcasted to each image dimension. By default, kernel_size is 1/8 of image height by 1/8 of its width.
# clip_limit : float, optional
#     Clipping limit, normalized between 0 and 1 (higher values give more contrast).
# nbins : int, optional
#     Number of gray bins for histogram ("data range").
# ```
# The larger the kernal_size the closer it is to a normal histogram equalization as it is the filter that is doing the equalization over the space.  Therefore, to imporve the adaptive histogram equalization for the me_and_blu.png image, it would be worth increasing the kernel_size so not to focus too much on the background in the sky.  The clip_limit take the top to histogram and flattens it out over all the pixels.  So the higher the number the more contrast is created.  Too high and once again we would get a regular histogram equalization. 

# %%
# For fun I wrote my own group based equalization...it was okay.

# groups = high_low_thresholding(img=img, threshold=thresholds)
# the_key = {}
# for group in Counter(groups.reshape(1,-1)[0]).keys():
#     img_local_eq = skimage.exposure.equalize_hist(img[groups == group])
#     the_key = {**the_key, **{k: v for k, v in zip(img[groups == group],img_local_eq)}}

# img_my_eq = img.copy()
# for i, x in enumerate(img_my_eq):
#     for j, a in enumerate(x):
#         img_my_eq[i][j] = the_key[a]

# image_histogram(img_my_eq, n_bins=50, threshold=None)


