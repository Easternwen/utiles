import numpy as np
from sklearn.utils import check_array,check_random_state
from sklearn import feature_extraction
import numbers
from itertools import product


def _compute_n_patches(image_shape, patch_size,step=1, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Parameters
    ----------
    image_shape : tuple of ints (image_height, image_width, image_depth)
        the dimensions of the image. For color images, the channel dimension is excluded.
        
    patch_size : tuple of ints (patch_height, patch_width,patch_depth)
        the dimensions of one patch.
        
    step : integer or tuple of length arr.ndim, optional default is 1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
        
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
 
    patch_indices_shape = ((np.array(image_shape) - np.array(patch_size)) //
                           np.array(step)) + 1
    
    all_patches = np.prod(patch_indices_shape)


    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches
    
def extract_patches_3d(image, patch_size,extraction_step=1, max_patches=None,random_state=None):
    """
    Reshape a 3D image into a collection of patches
    The resulting patches are allocated in a dedicated array.
    
    Parameters
    ----------
    image : array, shape = (image_height, image_width, image_depth) or
        (image_height, image_width,image_depth, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.
        
    patch_size : tuple of ints (patch_height, patch_width,patch_depth)
        the dimensions of one patch.
        
    extraction_step : integer or tuple of length arr.ndim, optional default is 1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions. 
        
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width,patch_depth) or
         (n_patches, patch_height, patch_width,patch_depth, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.    
    """

    i_h, i_w,i_d = image.shape[:3]
    p_h, p_w,p_d = patch_size
    
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")  
    if p_d > i_d:
        raise ValueError("Depth of the patch should be less than the depth"
                         " of the image.") 
        
    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w,i_d, -1))
    n_colors = image.shape[-1]

    extracted_patches = feature_extraction.image.extract_patches(image,
                                        patch_shape=(p_h, p_w,p_d, n_colors),
                                        extraction_step=extraction_step)

    n_patches = _compute_n_patches(image.shape[:3], patch_size,step=extraction_step, max_patches=max_patches)
    
    if max_patches:
        rng = check_random_state(random_state)
        if isinstance(extraction_step, numbers.Number):
            extraction_step = tuple([extraction_step] * (image.ndim-1))
        
        i_s = rng.randint((i_h - p_h)//extraction_step[0] + 1, size=n_patches)
        j_s = rng.randint((i_w - p_w)//extraction_step[1] + 1, size=n_patches)
        k_s = rng.randint((i_d - p_d)//extraction_step[2] + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s,k_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w,p_d, n_colors)
    # remove the color dimension if useless
#     if patches.shape[-1] == 1:
#         return patches.reshape((n_patches, p_h, p_w))
#     else:
    return patches

def reconstruct_from_patches_3d(patches, image_size, extraction_step=1):
    """Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, front to back, averaging the overlapping
    regions.
    
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width,patch_depth) or
        (n_patches, patch_height, patch_width,patch_depth, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension.
        
    image_size : tuple of ints (image_height, image_width,image_depth) or
        (image_height, image_width,image_depth, n_channels)
        the size of the image that will be reconstructed
    
    extraction_step : integer or tuple of length arr.ndim, optional default is 1
        Indicates step size at which extraction was performed.
        If integer is given, then the step is uniform in all dimensions. 
        
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image
    """   
    if patches.ndim-1 != len(image_size):
        raise ValueError("The dimension of the patch %r does not match the image dimension %r" % (patches.ndim-1,len(image_size)))   
        
    eps = 1e-5

    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * len(image_size))
        
    i_h, i_w, i_d = image_size[:3]
    p_h, p_w, p_d = patches.shape[1:4]
    
    
    img = np.zeros(image_size)
    used = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = (i_h - p_h)//extraction_step[0] + 1
    n_w = (i_w - p_w)//extraction_step[1] + 1
    n_d = (i_d - p_d)//extraction_step[2] + 1
    
    for p, (i, j, z) in zip(patches, product(range(n_h), range(n_w),range(n_d))):
        img[i*extraction_step[0]:i*extraction_step[0] + p_h, j*extraction_step[1]:j*extraction_step[1] + p_w,z*extraction_step[2]:z*extraction_step[2]+p_d] += p
        used[i*extraction_step[0]:i*extraction_step[0] + p_h, j*extraction_step[1]:j*extraction_step[1] + p_w,z*extraction_step[2]:z*extraction_step[2]+p_d] += 1

        
#     for i in range(i_h):
#         for j in range(i_w):
#             for k in range(i_d):
#             # divide by the amount of overlap
#             # XXX: is this the most efficient way? memory-wise yes, cpu wise?
#                 img[i, j, k] /= float(min(i + 1, p_h, i_h - i) *
#                                    min(j + 1, p_w, i_w - j)*
#                                      min(k+1,p_d,i_d-k))
    

    
    return img/used    