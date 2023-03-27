"""Main module."""
import numpy as np
from astropy.io import fits, ascii
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import Table
from photutils.centroids import centroid_com, centroid_sources
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
import tqdm
import astropy.units as u
import crds
import pdb

defaultCoord = SkyCoord(120.0592192 * u.deg,-10.79151878 * u.deg)

class photObj(object):
    """
    Do photometry on a list of images
    """
    def __init__(self,paths='*',coord=defaultCoord,
                 EECalc=0.878,descrip='test',
                 manualPlateScale=None):
        """
        manualPlateScale: None or float
            If None, will look up the plate scale
            If manually selected, it is the length of a pixel in 
            milli-arcseconds
        """
        self.path_input = paths
        self.fileList = search_for_images(paths)
        self.centroidBox= 13
        self.coord = defaultCoord
        self.src_radius = 10
        self.backg_radii = [12,20]
        self.ee_calc = EECalc ## Needs an update for each filter!!!
        self.descrip = descrip
        self.manualPlateScale = manualPlateScale

    def get_centroid(self,xguess,yguess,image):
        """
        Find the centroid
        """
        x, y = centroid_sources(image, xguess, yguess,
                                box_size=self.centroidBox,
                                centroid_func=centroid_com)

        return x,y

    def do_phot(self,xc,yc,image,head,error):
        """ 
        Do aperture photometry
        """
        srcAperture = CircularAperture((xc[0],yc[0]),r=self.src_radius)
        bkgAperture = CircularAnnulus((xc[0],yc[0]),r_in=self.backg_radii[0],
                                      r_out=self.backg_radii[1])

        srcPhot = aperture_photometry(image,srcAperture,error=error)
        bkgPhot = aperture_photometry(image,bkgAperture,error=error)
        
        bkgArea = bkgAperture.area
        srcArea = srcAperture.area
        bkgEst = bkgPhot['aperture_sum'] / bkgArea * srcArea
        
        bkgSubPhot = srcPhot['aperture_sum'] - bkgEst
        bkgErr = (bkgPhot['aperture_sum_err'] / bkgArea * srcArea)
        bkgSubPhot_err = np.sqrt(srcPhot['aperture_sum_err']**2 + bkgErr**2)

        ## Get the pixel area
        areaFile = crds.getreferences(head,reftypes=['area'])
        
        if self.manualPlateScale is None:
            with fits.open(areaFile['area']) as HDUList:
                avgArea = HDUList[0].header['PIXAR_SR']
                nearbyArea = HDUList['SCI'].data[int(yc),int(xc)]
                useArea = avgArea * nearbyArea
        else:
            useArea = (self.manualPlateScale * 1e-3 / 206265.)**2
        
        photJy = (bkgSubPhot * useArea / self.ee_calc * u.MJy).to(u.uJy)
        photJy_err = (bkgSubPhot_err * useArea/self.ee_calc * u.MJy).to(u.uJy)
        photRes = {}
        photRes['pxSum'] = bkgSubPhot[0]
        photRes['phot (uJy)'] = photJy[0]
        photRes['phot (uJy) err'] = photJy_err[0]
        return photRes


    def process_one_file(self,fits_filename):
        with fits.open(fits_filename) as HDUList:
            image_data = HDUList['SCI'].data
            error = HDUList['ERR'].data
            head = HDUList[0].header
            image_shape = image_data.shape
    
            # Convert the coordinates to pixel coordinates in the image
            wcs_res = WCS(HDUList['SCI'].header)
            coord_pix = self.coord.to_pixel(wcs=wcs_res)
        
            # Check if the pixel coordinates are inside the image
            xguess, yguess = coord_pix
            if 0 <= xguess < image_shape[1] and 0 <= yguess < image_shape[0]:
                xc, yc = self.get_centroid(xguess,yguess,image_data)
                newPos = pixel_to_skycoord(xc,yc,wcs_res)
                separation = self.coord.separation(newPos)
                if separation > 0.8 * u.arcsec:
                    phot_res = None
                else:
                    
                    phot_res = self.do_phot(xc,yc,image_data,head,error)

                    phot_res['coord'] = newPos
            else:
                ## no source
                phot_res = None
        return phot_res
            
    def process_all_files(self):
        t = Table()
        nFile = len(self.fileList)
        firstRow = True
        for ind in tqdm.tqdm(np.arange(nFile)):
            oneFile = self.fileList[ind]
            phot_res = self.process_one_file(oneFile)
            if phot_res is not None:
                if firstRow == True:
                    useKeys = phot_res.keys()
                    allRes = {}
                    for oneKey in useKeys:
                        allRes[oneKey] = [phot_res[oneKey]]
                    firstRow = False ## now the first row is set
                else:
                    for oneKey in useKeys:
                        allRes[oneKey].append(phot_res[oneKey])
            
        for oneKey in useKeys:
            t[oneKey] = allRes[oneKey]
        t.write('all_phot_{}.ecsv'.format(self.descrip),overwrite=True)
    

def search_for_images(paths):
    """
    Search for a set of images
    """
    fileList = np.sort(glob.glob(paths))
    return fileList

def find_images(paths='*',coord=defaultCoord):
    """ 
    find the images with a given source coordinate inside 
    """
    fileList = search_for_images(paths)
    inImg = []
    for one_file in fileList:
        inImg.append(check_coordinates_in_fits(one_file))
        
    # for ind,one_file in enumerate(fileList):
    #     print(one_file,inImg[ind])
    table = Table()
    table['path'] = fileList
    table['in image'] = inImg
    return table
    
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
    