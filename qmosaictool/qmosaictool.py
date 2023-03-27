"""Main module."""
import numpy as np
from astropy.io import fits, ascii
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS


defaultCoord = SkyCoord(120.0592192 * u.deg,-10.79151878 * u.deg)

def find_images(paths='*',coord=defaultCoord):
    fileList = np.sort(glob.glob(paths))
    inImg = []
    for one_file in fileList:
        inImg.append(check_coordinates_in_fits(one_file))
        
    for ind,one_file in enumerate(fileList):
        print(one_file,inImg[ind])
    
def check_coordinates_in_fits(fits_filename,coord=defaultCoord):
    """
    Checks if a given set of coordinates is inside a FITS image.
    
    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        A set of coordinates in the form of a SkyCoord object.
    fits_filename : str
        The filename of the FITS image file.
    
    Returns
    -------
    bool
        True if the coordinates are inside the image, False otherwise.
    """
    
    # Open the FITS file and extract the image dimensions
    with fits.open(fits_filename) as HDUList:
        image_data = HDUList['SCI'].data
        image_shape = image_data.shape
    
        # Convert the coordinates to pixel coordinates in the image
        wcs_res = WCS(HDUList['SCI'].header)
        coord_pix = coord.to_pixel(wcs=wcs_res)
    
        # Check if the pixel coordinates are inside the image
        x, y = coord_pix
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            return True
        else:
            return False
    