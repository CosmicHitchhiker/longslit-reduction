#!/home/astrolander/Documents/study/python/mamont/bin/python3
#coding=utf-8

import numpy as np
import numpy.ma as ma
from astropy.io import fits
import astropy.stats as astats
import scipy.signal as sig
from matplotlib import pyplot as plt
from astroscrappy import *
import json
import pandas as pd
from scipy.ndimage.interpolation import shift
from astropy.table import Table
import datetime
from astropy.modeling import models, fitting
import warnings
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import scipy.optimize as opt
from itertools import *
from longslit import geometry
import matplotlib.patches as patches
import sys



def open_fits_array_data(names, margins=[0, None, 0, None], header=False):
    '''Give an array of data of named fits files.

    Read data from all mentioned files and return an 3D ndarray of images.
    If margins are mentioned, evry image is cropped.
    If it is necessary also return a list of headers.

    Parameters
    ----------
    names : list of strings
        List of files to be opened.
    margins : list of inegers, optional
        Coordinates of image rectangle to be returned: xmin, xmax, ymin, ymax.
    header : bool, optional
        If True, also return list of headers.

    Returns
    -------
    fitses : 3D ndarray
        An array of opened fits-images.
    headers : list of fits.header
        A list of heders of mentioned files (returned when header is True).
    '''
    xmin, xmax, ymin, ymax = margins
    if len(np.shape(fits.getdata(names[0]))) == 2:
	    fitses = np.array([fits.getdata(names[0])[ymin:ymax, xmin:xmax]])
	    if header is True:
	        headers = [fits.getheader(names[0])]
	    for name in names[1:]:
	        fitses = np.append(fitses, [fits.getdata(name)[ymin:ymax, xmin:xmax]],
	                           axis=0)
	        if header is True:
	            headers.append(fits.getheader(name))
    elif len(np.shape(fits.getdata(names[0]))) == 3:
	    fitses = np.array([fits.getdata(names[0])[0, ymin:ymax, xmin:xmax]])
	    if header is True:
	        headers = [fits.getheader(names[0])]
	    for name in names[1:]:
	        fitses = np.append(fitses, [fits.getdata(name)[0, ymin:ymax, xmin:xmax]],
	                           axis=0)
	        if header is True:
	            headers.append(fits.getheader(name))
    if header is True:
        return fitses, headers
    else:
        return fitses


def get_bias(bias, gain=1):
    '''Calculate superbias and readnoise.

    Apply sigma-clipping to all given bias images.
    Calculate readnoise (median robust standard deviation multiplied by gain)
    Get superbias by averaging all bias images.

    Parameters
    ----------
    bias : 3D ndarray
        Array of bias images.
    gain : float, optional
        Electrons per ADU in given bias images (default is 1).

    Returns
    -------
    suber_bias : 2D ndarray
        Superbias image.
    read_noise : float
        Read noise in the current observations
    '''
    bias_clean = astats.sigma_clip(bias, sigma=5)
    read_noise = np.median(astats.mad_std(bias, axis=(1, 2))) * gain
    superbias = np.average(bias_clean, axis=0)
    superbias = superbias.filled(superbias.mean())
    return (superbias, read_noise)


def clear_CH(images):
    '''Clear images of cosmic hits.

    Remove cosmic hits using astroscrappy:
    https://github.com/astropy/astroscrappy

    Parameters
    ----------
    images : 3D ndarray
        Array of images.

    Returns
    -------
    images : 3D ndarrray
        Array of images without cosmic hits.
    '''
    return np.array(list(map(lambda x: detect_cosmics(x)[1], images)))


def get_flat(files, bias=0, N_str=40):
    '''Get sensitivity of each pixel.

    Substract bias. Average fits images (with cosmic hits removed).
    Get reference string by averaging N_str middle strings.
    Divide every string by reference string.

    Parameters
    ----------
    files : 3D ndarray
        Array of fits images of flat field.
    bias : 2D ndarray, optional
        Superbias image to be substracted.
    N_str : int, optional
        Nember of middle strings to be used in calculating
        reserence string (default is 40)

    Returns
    -------
    flat : 2D ndarray
        Image of sensitivity of each pixel.
    '''
    files = files - bias
    flat = np.mean(clear_CH(files), axis=0)
    mid = int(np.shape(flat)[0] / 2)
    mid_str = flat[mid - 20: mid + 20]
    reference_str = np.mean(mid_str, axis=0)
    flat = flat / reference_str
    print('FLAT IS CALCULATED')
    return flat


def linearize(data, WL_map=None, Y_corr=None, lam=None):
    '''Transform x-y image to wavelength-y image.

    Parameters
    ----------
    data : 2D ndarray
        Fits image (in x-y space)
    WL_map : 2D ndarray

    Returns
    -------
    F_interp : 2D ndarray

    '''
    x = np.arange(len(data[0]))
    y = np.arange(len(data))
    if WL_map is not None:
        data = np.array(list(map(lambda val, coord: np.interp(coord, x, val),
                                 data, WL_map)))
    if Y_corr is not None:
        data = np.array(list(map(lambda val, coord: np.interp(coord, y, val),
                                 data.T, Y_corr.T))).T
    if lam is not None:
        wl = np.linspace(lam.min(), lam.max(), len(data.T))
        data = np.array(list(map(lambda val: np.interp(wl, lam, val), data)))
    return data


def extract_sky(data, sky):
    '''Remove sky spectrum from the image.

    Read sky spectrum from the mentioned area.
    Fit sky spectrum changing by the second-order polinomial.
    Substract sky from the object area.

    Parameters
    ----------
    data : 2D ndarray
        Fits image
    sky : array of  4 integers
        Numbers of strings to be used as borders of sky area

    Returns
    -------
    data : 2D ndarray
        Image of object with sky spectrum substracted
    '''
    y_sky = np.arange(sky[0], sky[1])
    for i in range(2, len(sky), 2):
        if sky[i] > len(data):
            break
        if sky[i+1] > len(data):
            sky[i+1] = len(data) - 1
        y_sky = np.append(y_sky, np.arange(sky[i], sky[i+1]))
    tdata = data[y_sky].T
    sky_poly = np.array(list(map(lambda x: np.polyfit(y_sky, x, 2), tdata)))
    real_sky = np.array(list(map(lambda x: np.polyval(x, np.arange(len(data))), sky_poly))).T
    return data - real_sky, real_sky


def correct_dispersion(data, errors=None):
    '''Correct non-straightness of spectra.

    Just move evry row to make the brightest pixel of every raw
    lie on the same y-coordinate.

    Parameters
    ----------
    data : 2D ndarray
        Fits image of galaxy
    erros : 2D ndarray, optional
        Fits image of errors

    Returns
    -------
    corrected : 2D ndarray
        Image with despersion corrected
    errors : 2D ndarray
        Errors with dispersion corrected exactly like in Image!
        (return if errors are given)
    '''
    data = data.T
    m = np.array(list(map(np.argmax, data)), dtype=int)
    x = np.arange(len(m))
    km = np.polyfit(x, m, 2)
    m = np.polyval(km, x)
    mv = np.min(m) - m
    corrected = np.array(list(map(lambda x, y: shift(x, y), data, mv))).T
    if errors is not None:
        errors = errors.T
        errors_corr = np.array(list(map(lambda x, y: shift(x, y), errors, mv))).T
        return corrected, errors_corr
    print('DISPERSION IS CORRECTED')
    return(corrected)


def get_clear_data(data, mode, WL_map, Y_correction, sky_reg, galaxy_region=[0, -1],
                   gain=1, bias=0, flat=1, read_noise=0, sensitivity=1,
                   star_wl=None, lam=None):
    '''Perform full reduction for data and calculate errors.

    Substract bias.
    Correct sensitivity (divide by the coefficients of sensitivity).
    Remove cosmic hits.
    Transform x into wavelenght.
    Substract sky.
    Correct dispersion.
    Summarize all images to one.

    Parameters
    ----------
    data : 3D ndarray
        Array of object images
    WL_map : 2D ndarray
        Image, where value in pixel is a wavelenght in this pixel
    sky_reg : array of  4 integers
        Numbers of strings to be used as borders of sky area
    galaxy_region : array of 2 integers, optional
        y-borders of galaxy
    gain : float
        electrons per ADU (from object header)
    bias : 2D ndarray
        Image of bias
    flat : 2D ndarray
        Image of sensitivity of each pixel
    read_noise : float
        Read noise
    neon : 2D ndarray
        Neon image to check linerization

    Returns
    -------
    data : 2D ndarray
        Result image
    errors : 2D ndarray
        Result errors image
    '''
    medium_results = {}
    medium_results['initial'] = np.sum(data, axis=0)
    if mode['bias'] != 'No':
        data = data - bias
        data[data < 0] = 0
        medium_results['bias'] = np.sum(data, axis=0)
    errors = np.sqrt(np.abs(data) * gain + read_noise**2) / gain

    if mode['cosmics'] == 'Yes':
        data = np.array(list(map(lambda x: detect_cosmics(x)[1], data)))
        medium_results['cosmics'] = np.sum(data, axis=0)


    errors = np.sqrt(np.sum(errors**2, axis=0))
    data = np.sum(data, axis=0)

    # data = np.array(list(map(lambda x: linearize(x, WL_map, Y_correction, lam), data)))
    # errors = np.array(list(map(lambda x: linearize(x, WL_map, Y_correction, lam), errors)))
    data = linearize(data, WL_map, Y_correction, lam)
    errors = linearize(errors, WL_map, Y_correction, lam)
    medium_results['linerization'] = data

    print('DATA IS LINEARIZED')

    if mode['flat'] != 'No':
        data = data / flat
        errors = errors / flat
        medium_results['flat'] = data

    # if mode['sky'] != 'No':
    #     # sum_x_data = np.sum(data, axis=(0,2))
    #     sum_x_data = np.sum(data, axis = 1)
    #     plt.plot(sum_x_data)
    #     for i in range(0, len(sky_reg), 2):
    #         plt.plot([sky_reg[i], sky_reg[i+1]], [sum_x_data[sky_reg[i]], sum_x_data[sky_reg[i+1]]])
    #     plt.show()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(data)
    #     for i in range(0, len(sky_reg), 2):
    #         rect = patches.Rectangle((0, sky_reg[i]), len(data[0]),
    #                                  sky_reg[i+1]-sky_reg[i], color=(1, 1, 0, 0.4))
    #         ax.add_patch(rect)
    #     plt.show()

    if mode['star'] != 'No':
        sensitivity = np.interp(lam, star_wl, sensitivity)
        data = data * sensitivity
        errors = errors * sensitivity
        print("SENSITIVITY CORRECTION IS DONE")
        medium_results['star'] = data

    if mode['sky'] != 'No':
        # data = np.array(list(map(lambda x: extract_sky(x, sky_reg), data)))
        data, sky = extract_sky(data, sky_reg)
        print('SKY SPECTRUM IS EXTRACTED')
    # result = [[]] * len(data)
    # err_result = [[]] * len(errors)
    # for i in range(len(data)):
    #     result[i], err_result[i] = correct_dispersion(data[i], errors=errors[i])
    # errors = np.sum(errors, axis=0)
    # data = np.sum(data, axis=0)
    # plt.imshow(data)
    # plt.title('sum')
    # plt.show()
    # data[data < 0] = 0
    # errors = errors[sky_reg[1]:sky_reg[2]]
    # errors = errors[galaxy_region[0]:galaxy_region[1]]
    # data = data[galaxy_region[0]:galaxy_region[1]]
    return data, errors, medium_results, sky


def snr_bin(data, err, rng=None, SNR_min=3, SNR_des=15):
    if rng is None:
        rng = [0, np.shape(data)[1]]
    signal = np.sum(data[:, rng[0]:rng[1]], axis=1)
    signal[signal < 0] = 0
    noise = np.sqrt(np.sum(err[:, rng[0]:rng[1]]**2, axis=1))
    snr = signal/noise
    ymask = (snr > SNR_min)

    y = np.arange(len(data))[ymask]
    x = np.zeros(len(data))[ymask]
    signal = signal[ymask]
    noise = noise[ymask]
    good_data = data[ymask]
    good_err = err[ymask]

    binNum, xNode, yNode, xBar, yBar, sn, nPixels,  \
        scale = voronoi_2d_binning(x, y, signal, noise, SNR_des)

    plt.show()

    res = np.zeros((len(yBar), np.shape(data)[1]))
    errors = np.zeros((len(yBar), np.shape(data)[1]))
    for i in range(len(res)):
        res[i] = np.sum(good_data[binNum == i], axis=0)
        errors[i] = np.sqrt(np.sum(good_err[binNum == i]**2, axis=0))
    res = res[np.argsort(yBar)]
    errors = errors[np.argsort(yBar)]
    sep = yBar[np.argsort(yBar)]
    return res, errors, sep

def read_new_config(parameters):
    print('CONFIGTYPE = NEW')
    mode = {'bias': 'Yes', 'flat': 'Yes', 'x_lin': 'Yes', 'y_lin': 'Yes',
            'cosmics': 'Yes', 'sky': 'Yes', 'sum': 'Yes', 'bin': 'Yes',
            'star': 'Yes', 'WL': 'Yes', 'margins': 'Yes'}
    for i in mode.keys():
        mode[i] = parameters[i]['perform']

    # Margins of image (excludind overscan)
    if mode['margins'] == 'Yes':
        margins = parameters['margins']['margins']
    else:
        margins = [None, None, None, None]

    # Region to use for sky extraction
    sky_reg = parameters['sky']['margins']

    # Region with useful galaxy data
    # We have margins y-shift
    gal_reg = parameters['crop']['margins']
    # Directory to write in
    wd = parameters['wd'] + '/'
    
    bias_names = parameters['bias']['files']

    flat_names = parameters['flat']['files']

    neon_names = parameters['x_lin']['files']

    dots_names = parameters['y_lin']['files']

    snr = parameters['bin']['snr']

    obj_names = parameters['object']['files']

    star_names = parameters['star']['files']
    ref_star = parameters['star']['reference']

    header = parameters['header']['update']
    
    if mode['WL'] != 'No':
        file = parameters['WL']['reference'][0]
        try:
            ref_neon = np.loadtxt(file).T
        except OSError:
            ref_neon = None
        base_neon_name = parameters['WL']['files'][0]
    else:
        ref_neon = base_neon_name = None

    return (mode, margins, wd, gal_reg, obj_names, sky_reg, bias_names,
            flat_names, neon_names, dots_names, star_names, snr, ref_neon,
            base_neon_name, ref_star, header)


def read_config(configname):
    param_file = open(configname, 'r')
    parameters = json.load(param_file)

    try:
        if parameters['configtype'] == 'new':
            return read_new_config(parameters)
    except KeyError:
        mode = {'bias': 'Yes', 'flat': 'Yes', 'x_lin': 'Yes', 'y_lin': 'Yes',
                'cosmics': 'Yes', 'sky': 'Yes', 'sum': 'Yes', 'bin': 'Yes',
                'star': 'Yes', 'WL': 'Yes'}

        sky_reg = bias_names = flat_names = neon_names = dots_names = snr = \
            ref_neon = None

        for i in mode.keys():
            mode[i] = parameters[i]['perform']

        # Margins of image (excludind overscan)
        margins = parameters['regions']['margins']

        # print('margins: ', margins)

        # Region to use for sky extraction
        # It is set in raw fits coordinates, but we have margins, so we need to
        # move it according to y-shift
        if mode['sky'] == 'Yes':
            sky_reg = [x - margins[2] for x in parameters['sky']['regions']]
        else:
            sky_reg = [0, 0]

        # Region with useful galaxy data
        # We have margins y-shift and sky y-shift
        gal_reg = [x - margins[2] for x in parameters['regions']['galaxy']]
        # Directory to write in
        wd = parameters['paths']['wd']
        # Directory to read from
        rd = parameters['paths']['rd']

        if mode['bias'] != 'No':
            bias_names = [rd + x + '.fts' for x in parameters['bias']['files']]

        if mode['flat'] != 'No':
            flat_names = [rd + x + '.fts' for x in parameters['flat']['files']]

        if mode['x_lin'] != 'No':
            neon_names = [rd + x + '.fts' for x in parameters['x_lin']['files']]

        if mode['y_lin'] != 'No':
            dots_names = [rd + x + '.fts' for x in parameters['y_lin']['files']]

        if mode['bin'] == 'Yes':
            snr = parameters['bin']['snr']

        obj_names = [rd + x + '.fts' for x in parameters['paths']['galaxy']]

        if mode['WL'] == 'Yes':
            ref_neon = np.loadtxt(parameters['WL']['file']).T

        return (mode, margins, wd, gal_reg, obj_names, sky_reg, bias_names,
                flat_names, neon_names, dots_names, snr, ref_neon)


def matrix_sensitivity(star, refererence):
    data = open_fits_array_data(star)[0]
    header = fits.getheader(star[0])
    lmin = header['CRVAL1']
    lmax = lmin + header['CDELT1']*header['NAXIS1']
    max_star = np.argmax(np.sum(data, axis=1))
    wl = np.linspace(lmin, lmax, len(data[0]))
    refererence[1] = refererence[1] / np.max(refererence[1])
    obstar = np.sum(data[max_star - 5: max_star + 6], axis=0)
    obstar = obstar / np.max(obstar)

    obstar_i = np.interp(refererence[0], wl, obstar)
    k = refererence[1] / obstar_i
    k = np.interp(wl, refererence[0], sig.medfilt(k, 5))
    k = k / np.median(k)
    return k, wl, data


def fill_header(obj, old_header, lamrange, readnoise, offset, fwhm):
    header = old_header
    neon_header = fits.PrimaryHDU().header

    header['DATE'] = neon_header['DATE'] = str(datetime.date.today())

    header['CDELT1'] = neon_header['CDELT1'] = \
        (float(format((lamrange[1] - lamrange[0]) / np.shape(obj)[1], '.3f')),
         'DISPERSION, ANGSTROM/PX')
    header['CDELT2'] = (float(header['IMSCALE'].split('x')[1]),
                        'SCALE, ARCSEC/PX')
    neon_header['CDELT2'] = 1

    header['CRPIX1'] = neon_header['CRPIX1'] = (1, 'REFERENCE PIX')
    header['CRPIX2'] = (-offset, 'REFERENCE PIX')
    neon_header['CRPIX2'] = 1

    header['CTYPE1'] = neon_header['CTYPE1'] = 'AWAV'
    header['CRVAL1'] = neon_header['CRVAL1'] = (float(format(lamrange[0], '.3f')), 'WAVELENGTH 1-st ELEMENT')
    header['CUNIT1'] = neon_header['CUNIT1'] = 'Angstrom'

    header['RNOISE'] = neon_header['RNOISE'] = readnoise
    header['FWHM'] = neon_header['FWHM'] = fwhm
    return header, neon_header


def basic_preparation(mode, margins=None, bias_names=None, flat_names=None,
                      neon_names=None, dots_names=None, ref_neon=None, base_neon_name=None):
    if mode['bias'] == 'Yes':
        bias_data, bias_headers = open_fits_array_data(bias_names, margins,
                                                       header=True)
        print('calculating bias')
        superbias, read_noise = get_bias(bias_data, (bias_headers[0])['GAIN'])
        print('SUPERBIAS IS CALCULATED')
        print("Read Noise = ", read_noise)
    elif mode['bias'] == 'File':
        bias_data, bias_header = open_fits_array_data(bias_names, header=True)
        superbias = bias_data[0]
        read_noise = bias_header[0]['RNOISE']
    elif mode['bias'] == 'No':
        superbias = 0
        read_noise = 0

    if mode['x_lin'] == 'Yes':
        neon_data = open_fits_array_data(neon_names, margins)
        neon = np.average(np.clip(neon_data - superbias, 0, None), axis=0)
        print('calculating x correction')
        WL_map = geometry.get_correction_map(neon)
        print('X CORRECTION MAP IS PERFORMED')
    if mode['x_lin'] == 'File':
        WL_map = open_fits_array_data(neon_names)[0]
        neon = None
    if mode['x_lin'] == 'No':
        WL_map = None
        neon = None

    if mode['y_lin'] == 'Yes':
        dots_data = open_fits_array_data(dots_names, margins)
        dots = np.average(np.clip(dots_data - superbias, 0, None), axis=0)
        print('calculating y correction')
        Y_correction = geometry.get_correction_map(dots.T, ref='center').T
        print('Y CORRECTION MAP IS PERFORMED')
    elif mode['y_lin'] == 'File':
        Y_correction = open_fits_array_data(dots_names)[0]
        dots = None
    elif mode['y_lin'] == 'No':
        Y_correction = None
        dots = None

    if mode['WL'] == 'Yes':
        if base_neon_name is None:
            neon_pre = linearize(neon, WL_map=WL_map)
        else:
            neon_pre = fits.getdata(base_neon_name)
        lam = geometry.fit_neon(np.sum(neon_pre, axis=0), ref_neon[0], ref_neon[1])
        lam2 = np.linspace(lam[0], lam[-1], len(lam))
        if WL_map is None:
            Xsize = margins[1] - margins[0]
            Ysize = margins[3] - margins[2]
            WL_map_temp = np.tile(np.arange(Xsize), Ysize).reshape(Ysize, Xsize)
            WL_map = np.array(list(map(lambda x: np.interp(lam2, lam, x), WL_map_temp)))
        else:
            WL_map = np.array(list(map(lambda x: np.interp(lam2, lam, x), WL_map)))
        lam = lam2
    else:
        lam = None

    if mode['x_lin'] == 'Yes':
        neon = linearize(neon, WL_map=WL_map, Y_corr=Y_correction,
                         lam=lam)
    if mode['y_lin'] == 'Yes':
        dots = linearize(dots, WL_map=WL_map, Y_corr=Y_correction,
                         lam=lam)

    if mode['flat'] == 'Yes':
        flat_data = open_fits_array_data(flat_names, margins)
        flat_data = np.array([linearize(np.mean(flat_data - superbias, axis=0),
                                        WL_map, Y_correction)])
        flat_coeff = get_flat(flat_data)
    elif mode['flat'] == 'File':
        flat_coeff = open_fits_array_data(flat_names)[0]
    elif mode['flat'] == 'No':
        flat_coeff = 1

    return (read_noise, superbias, flat_coeff, WL_map, Y_correction, neon,
           dots, lam)


def save_results(wd, mode, header, neon_header, superbias, flat_coeff, WL_map,
                 Y_correction, reduct_star, star_wl, sensitivity, obj, err, neon,
                 dots, obj_bin, err_bin, sep, medium_results, sky):
    fits.PrimaryHDU(sky.astype('float32')).writeto(wd + 'sky.fits', overwrite=True)
    fits.PrimaryHDU(medium_results['initial'].astype('float32')).writeto(wd + 'm_initial.fits', overwrite=True)

    if mode['bias'] == 'Yes':
        fits.PrimaryHDU(superbias.astype('float32'), header=header).writeto(wd + 'sbias.fits', overwrite=True)
    if mode['bias'] != 'No':
        fits.PrimaryHDU(medium_results['bias'].astype('float32')).writeto(wd + 'm_bias.fits', overwrite=True)

    if mode['flat'] == 'Yes':
        hdu = fits.PrimaryHDU(flat_coeff.astype('float32'))
        hdu.writeto(wd + 'flatcoef.fits', overwrite=True)
    if mode['flat'] != 'No':
        fits.PrimaryHDU(medium_results['flat'].astype('float32')).writeto(wd + 'm_flat.fits', overwrite=True)

    if mode['x_lin'] == 'Yes':
        fits.PrimaryHDU(WL_map.astype('float32')).writeto(wd + 'WL_map.fits', overwrite=True)
        fits.PrimaryHDU(neon.astype('float32'), header=neon_header).writeto(wd + 'neon.fits', overwrite=True)
    if mode['y_lin'] == 'Yes':
        fits.PrimaryHDU(Y_correction.astype('float32')).writeto(wd + 'Y_correction.fits', overwrite=True)
        fits.PrimaryHDU(dots.astype('float32')).writeto(wd + 'dots.fits', overwrite=True)

    if mode['x_lin'] != 'No' or mode['y_lin'] != 'No' or mode['WL'] != 'No':
        fits.PrimaryHDU(medium_results['linerization'].astype('float32')).writeto(wd + 'm_linerization.fits', overwrite=True)

    if mode['star'] == 'Yes':
        star_image = fits.PrimaryHDU(reduct_star.astype('float32'))
        star_k = fits.BinTableHDU.from_columns(
            [fits.Column(name='wavelenght', format='E', array=star_wl),
             fits.Column(name='sensitivity', format='E', array=sensitivity)])
        fits.HDUList(hdus=[star_image, star_k]).writeto(wd + 'star.fits', overwrite=True)
    if mode['star'] != 'No':
        fits.PrimaryHDU(medium_results['star'].astype('float32')).writeto(wd + 'm_star.fits', overwrite=True)

    if mode['cosmics'] != 'No':
        fits.PrimaryHDU(medium_results['cosmics'].astype('float32')).writeto(wd + 'm_cosmics.fits', overwrite=True)

    fits.PrimaryHDU(obj.astype('float32'), header=header).writeto(wd + 'result.fits', overwrite=True)
    fits.PrimaryHDU(err.astype('float32'), header=header).writeto(wd + 'errors.fits', overwrite=True)
    SNR = obj / err
    fits.PrimaryHDU(SNR.astype('float32')).writeto(wd + 'SNR.fits', overwrite=True)

    if mode['bin'] == 'Yes':
        image = fits.PrimaryHDU(obj_bin.astype('float32'), header=header)
        table = fits.BinTableHDU(data=Table([sep]))
        fits.HDUList(hdus=[image, table]).writeto(wd + 'H_bins.fits', overwrite=True)
        image = fits.PrimaryHDU(err_bin.astype('float32'), header=header)
        fits.HDUList(hdus=[image, table]).writeto(wd + 'H_err_bins.fits', overwrite=True)
        image = fits.PrimaryHDU((obj_bin/err_bin).astype('float32'), header=header)
        fits.HDUList(hdus=[image, table]).writeto(wd + 'H_SNR_bins.fits', overwrite=True)


def main(args=None):
	# mode - str словарь, каждый элемент - указание, выполнять ли данный этап:
	#   Yes - выполнять, No - пропустить, File - готовый результат из файла
	# margins - list со значениями [Xmin, Xmax, Ymin, Ymax]
	# wd - str путь к папке, куда будут сохранены результаты
	# gal_reg - list [Ymin, Ymax] для результирующего файла (с учётом margins)
    # obj_names - list str имена файлов с сырыми спектрами объекта
    # bias_names - list str имена файлов для калибровки bias
    # flat_names - list str имена файлов для калибровки flat
    # neon_names - list str имена файлов для коррекции геометрии по X
    # dots_names - list str имена файлов для коррекции геометрии по Y
    # star_names - list str имена файлов для калибровки спектральной чувств.
    # des_snr - float желаемый сигнал/шум в строке при бинировании
    # ref_neon - 2D ndarray c опорным спектром для калибровки:
    #   [0] - длины волн линий, [1] - относительная высота пиков
    # ref_star - list str имя файла с реальным спектром эталонной звезды
    # rcfg - list со всем вышеперечисленным
    if args is None:
        args = sys.argv
    rcfg = read_config(args[1])
    mode, margins, wd, gal_reg, obj_names, sky_reg, bias_names, flat_names, \
        neon_names, dots_names, star_names, des_snr, ref_neon, base_neon_name, \
        ref_star, header_upd = rcfg

    # read_noise - float средний шум считывания
    # superbias - 2D ndarray средний кадр смещения
    # flat_coeff - 2D ndarray кадр, характеризующий вариации чувствительности
    # WL_map - 2D ndarray карта интерполяции для геом. коррекции по Х
    # Y_correction - 2D ndarray карта интерполяции для геом. коррекции по Y
    # neon - 2D ndarray исправленный калибровочный кадр геом. коррекции по X
    # dots - 2D ndarray исправленный калибровочный кадр геом. коррекции по Y
    # lam - массив длин волн, соответствующих каждому пикселю
    read_noise, superbias, flat_coeff, WL_map, Y_correction, neon, dots, lam \
        = basic_preparation(mode, margins, bias_names, flat_names, neon_names,
                            dots_names, ref_neon, base_neon_name)

    if mode['WL'] != 'No':
        lmin = lam.min()
        lmax = lam.max()
        print('wavelenghts range: ', lmin, lmax)
    if mode['y_lin'] != 'No':
        ymin = int(np.max(Y_correction.T[:, 0]))
        ymax = int(np.min(Y_correction.T[:, -1]))
        # if mode['sky'] != 'No':
        #     sky_reg = [x - ymin for x in sky_reg]
        print(ymin, ymax)

    if mode['star'] == 'Yes':
        ref_star_spectrum = np.loadtxt(ref_star[0]).T

        sensitivity, star_wl, reduct_star = matrix_sensitivity(star_names, ref_star_spectrum)
    else:
        sensitivity = star_wl = reduct_star = None

    obj, obj_headers = open_fits_array_data(obj_names, margins, header=True)
    obj, err, medium_results, sky = get_clear_data(obj, mode, WL_map, Y_correction, sky_reg,
                              gal_reg, (obj_headers[0])['GAIN'],
                              superbias, flat_coeff, read_noise,
                              sensitivity, star_wl, lam=lam)
    plt.imshow(obj)
    plt.show()

    if header_upd:
        if neon is not None:
            fwhm = geometry.calc_fwhm(np.sum(neon, axis=0), lam) #np.linspace(lmin, lmax, len(neon[0])))
        else:
            fwhm = geometry.calc_fwhm(np.sum(fits.getdata(base_neon_name), axis=0), lam)
        offset = margins[2] + ymin + gal_reg[0] + sky_reg[1]
        print('Offset ', offset)
        print(margins[2], ' ', ymin, ' ', gal_reg[0], ' ', sky_reg[1])
        header, neon_header = fill_header(obj, obj_headers[0], (lmin, lmax),
                                          read_noise, offset, fwhm)
    else:
        header = neon_header = obj_headers[0]

    if mode['bin'] == 'Yes':
        wl = np.arange(header['NAXIS1'])*header['CDELT1']+header['CRVAL1']
        fwhm_pix = int(header['FWHM']/header['CDELT1'])
        iHa = np.argmin(np.abs(wl - 6563))
        rng = [iHa - 3*fwhm_pix, iHa + 3*fwhm_pix]
        print('binning region ', rng)
        print('desired SNR ', des_snr)
        obj_bin, err_bin, sep = snr_bin(obj, err, rng=rng, SNR_des=des_snr)
    else:
        obj_bin = None
        err_bin = None
        sep = None

    # obj = np.array(list(map(lambda x: x / np.mean(x), obj)))
    # image = fits.PrimaryHDU(obj[1:-1].astype('float32'), header=header)
    # fits.HDUList(hdus=[image, table]).writeto(wd + 'normalized_bins.fits', overwrite=True)
    save_results(wd, mode, header, neon_header, superbias, flat_coeff, WL_map,
                 Y_correction, reduct_star, star_wl, sensitivity, obj, err, neon,
                 dots, obj_bin, err_bin, sep, medium_results, sky)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
