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
import os
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

defaultCoord = SkyCoord(120.0592192 * u.deg,-10.79151878 * u.deg)

class photObj(object):
    """
    Do photometry on a list of images
    """
    def __init__(self,paths='*',coord=defaultCoord,
                 EECalc=0.878,descrip='test',
                 manualPlateScale=None,src_radius=10,
                 bkg_radii=[12,20],
                 directPaths=None,filterName=None,
                 interpolate=False):
        """
        manualPlateScale: None or float
            If None, will look up the plate scale
            If manually selected, it is the length of a pixel in 
            milli-arcseconds
        """
        if directPaths is None:
            self.path_input = paths
            self.fileList = search_for_images(paths)
        else:
            self.path_input = 'direct'
            self.fileList = directPaths
        
        self.centroidBox= 13
        self.coord = coord
        self.src_radius = src_radius
        self.backg_radii = bkg_radii
        self.ee_calc = EECalc ## Needs an update for each filter!!!
        self.descrip = descrip
        self.manualPlateScale = manualPlateScale
        self.filterName = filterName
        self.interpolate = interpolate

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
        photRes['bkgEst'] = bkgEst[0]
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
                    
                    if self.interpolate == True:
                        ## interpolate where the photometry is happening
                        kernel = Gaussian2DKernel(x_stddev=1)
                        margin = 5
                        x_st = np.max([int(xc - self.backg_radii[1] - margin),0])
                        x_end = np.min([int(xc + self.backg_radii[1] + margin),image_shape[1]])
                        y_st = np.max([int(yc - self.backg_radii[1] - margin),0])
                        y_end = np.min([int(yc + self.backg_radii[1] + margin),image_shape[0]])
                        cutout = image_data[y_st:y_end,x_st:x_end]
                        cutout_error = error[y_st:y_end,x_st:x_end]
                        
                        fixed_image = interpolate_replace_nans(cutout, kernel)
                        fixed_error = interpolate_replace_nans(cutout_error,kernel)
                        image_data[y_st:y_end,x_st:x_end] = fixed_image
                        error[y_st:y_end,x_st:x_end] = fixed_error
                        
                    
                    phot_res = self.do_phot(xc,yc,image_data,head,error)

                    phot_res['coord'] = newPos
                    phot_res['filename'] = os.path.basename(fits_filename)
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
        
        t.meta['FILTER'] = str(self.filterName)
        t.write('all_phot_{}.ecsv'.format(self.descrip),overwrite=True)
    

class manyCals(object):
    def __init__(self,pathSearch,srcDescrip='_test',
                 fixApSizes=None,
                 interpolate=False,manualPlateScale=None):
        """
        object to organize and do photometry on many files

        Parameters
        ----------
        fixApSize: list of 3 floats or None
            Fixed aperture size for source, background start and
            background end. Used for mosaics on a 
            shared common pixel plate scale
        """
        self.path_input = pathSearch
        self.fileList = search_for_images(pathSearch)
        self.srcDescrip = srcDescrip
        self.interpolate = interpolate
        self.fixApSizes=fixApSizes
        self.manualPlateScale = manualPlateScale

    def gather_filters(self):
        t = Table()
        t['path'] = self.fileList
        filterList = []
        pupilList = []
        for oneFile in self.fileList:
            head = fits.getheader(oneFile)
            filterList.append(head['FILTER'])
            pupilList.append(head['PUPIL'])
        t['Filter'] = filterList
        t['Pupil'] = pupilList
        self.t = t
        self.filters = np.unique(t['Filter'])

    def do_all_filt(self):
        self.gather_filters()
        for oneFilt in self.filters:
            pts = self.t['Filter'] == oneFilt
            fileList = self.t['path'][pts]
            if oneFilt in apCorEstimate:
                EECalc = apCorEstimate[oneFilt]
            else:
                print("No filter {} found in apcor table".format(oneFilt))
                pdb.set_trace()
            if self.fixApSizes is not None:
                srcap, bkgStart, bkgEnd = self.fixApSizes
            elif oneFilt in ap_px_to_use:
                srcap, bkgStart, bkgEnd = ap_px_to_use[oneFilt]
            else:
                print("No filter {} found in apsize table".format(oneFilt))
                pdb.set_trace()

            oneDescrip = "{}{}".format(oneFilt,self.srcDescrip)
            po = photObj(directPaths=fileList,EECalc=EECalc,
                         descrip=oneDescrip,src_radius=srcap,
                         bkg_radii=[bkgStart,bkgEnd],
                         filterName=oneFilt,
                         manualPlateScale=self.manualPlateScale,
                         interpolate=self.interpolate)
            po.process_all_files()

    def combine_phot(self):
        photFiles = np.sort(glob.glob('all_phot_*{}.ecsv'.format(self.srcDescrip)))
        
        filt_list = []
        pxSum, apPhot, apPhot_err, coordMed = [], [], [], []
        for oneFile in photFiles:
            dat = ascii.read(oneFile)
            filt = dat.meta['FILTER']
            filt_list.append(filt)
            pxSum.append(np.nanmedian(dat['pxSum']))
            apPhot.append(np.nanmedian(dat['phot (uJy)']))
            apPhot_err.append(np.nanmedian(dat['phot (uJy) err']))
            coordMed.append(SkyCoord(np.nanmedian(dat['coord'].ra),
                                     np.nanmedian(dat['coord'].dec)))
        res = Table()
        res['Filter'] = filt_list
        res['coord'] = coordMed
        res['pxSum'] = pxSum
        res['phot (uJy)'] = apPhot
        res['phot (uJy) err'] = apPhot_err
        res['file'] = photFiles
        res.write('combined_phot_{}.ecsv'.format(self.srcDescrip),overwrite=True)

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


def lookup_flux(catalog,coord=defaultCoord,
                distThresh=0.8 * u.arcsec):
    """
    Loook up flux from a catalog

    Parameters
    -----------
    catalog: str
        catalog name or wildcard search pattern
    coord: astropy.coordinates.SkyCoord object
        Coordinates for the star to look up
    distThresh: astropy.units.quantity.Quantity
        Distance threshold, inside which the a match is considered good
    """
    fileList = np.sort(glob.glob(catalog))
    for onePath in fileList:
        print(onePath)
        all_dat = ascii.read(onePath)
        goodpt = (np.isfinite(all_dat['sky_centroid'].ra) & 
                  np.isfinite(all_dat['sky_centroid'].dec))
        dat = all_dat[goodpt]
        res = coord.match_to_catalog_sky(dat['sky_centroid'])

        distance = res[1][0].to(u.arcsec)
        
        if distance < distThresh:
            fl = dat['aper_total_flux'][res[0]].to(u.uJy)
            flerr = dat['aper_total_flux_err'][res[0]].to(u.uJy)
            print('Flux: ',fl,'+/-',flerr)
        else:
            print("No source found in catalog closer than {}".format(distThresh))

## 0.3 arcsec for SW and LW
## https://jwst-docs.stsci.edu/files/97978351/182257576/1/1669655270449/Encircled_Energy_LW_ETCv2.txt

apCorEstimate = {'F070W': 0.895,
                'F090W': 0.889,
                'F115W': 0.882,
                'F140M': 0.878,
                'F150W2': 0.878,
                'F150W': 0.878,
                'F162M': 0.877,
                'F164N': 0.877,
                'F182M': 0.877,
                'F187N': 0.876,
                'F200W': 0.875,
                'F210M': 0.873,
                'F212N': 0.873,
                'F250M': 0.853,
                'F277W': 0.85,
                'F300M': 0.849,
                'F322W2': 0.847,
                'F323N': 0.846,
                'F335M': 0.845,
                'F356W': 0.843,
                'F360M': 0.841,
                'F405N': 0.836,
                'F410M': 0.835,
                'F430M': 0.829,
                'F444W': 0.822,
                'F460M': 0.81,
                'F466N': 0.809,
                'F470N': 0.805,
                'F480M': 0.796}

ap_px_to_use = {'F070W' : [10, 12, 20],
                'F090W' : [10, 12, 20],
                'F115W' : [10, 12, 20],
                'F140M' : [10, 12, 20],
                'F150W2': [10, 12, 20],
                'F150W' : [10, 12, 20],
                'F162M' : [10, 12, 20],
                'F164N' : [10, 12, 20],
                'F182M' : [10, 12, 20],
                'F187N' : [10, 12, 20],
                'F200W' : [10, 12, 20],
                'F210M' : [10, 12, 20],
                'F212N' : [10, 12, 20],
                'F250M' : [ 5,  6, 10],
                'F277W' : [ 5,  6, 10],
                'F300M' : [ 5,  6, 10],
                'F322W2': [ 5,  6, 10],
                'F323N' : [ 5,  6, 10],
                'F335M' : [ 5,  6, 10],
                'F356W' : [ 5,  6, 10],
                'F360M' : [ 5,  6, 10],
                'F405N' : [ 5,  6, 10],
                'F410M' : [ 5,  6, 10],
                'F430M' : [ 5,  6, 10],
                'F444W' : [ 5,  6, 10],
                'F460M' : [ 5,  6, 10],
                'F466N' : [ 5,  6, 10],
                'F470N' : [ 5,  6, 10],
                'F480M' : [ 5,  6, 10]}
