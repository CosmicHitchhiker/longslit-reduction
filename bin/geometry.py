#!/usr/bin/env python3

import numpy as np
import scipy.signal as sig
import numpy.ma as ma
from pipeline import *
import scipy.optimize as opt
from matplotlib import pyplot as plt
from itertools import *


def all_together_now(neon, ref_neon):
    '''На выходе - "карта интерполяции" на линейную по длинам волн шкалу,
    значение первого элемента и шага.
    '''
    wl_map = get_correction_map(neon)
    lam = geometry.fit_neon(np.sum(neon, axis=0), ref_neon[0], ref_neon[1])
    lam2 = np.linspace(lam[0], lam[-1], len(lam))
    new_wl = np.array(list(map(lambda x: np.interp(lam2, lam, x), WL_map)))
    return lam2, new_wl

def get_correction_map(neon, verbose=False, ref='mean'):
    '''Считает карту интерполяции.
    В каждой строчке - те координаты, на которые нужно
    интерполировать исходное изображение, чтобы исправить
    геометрические искажения вдоль оси Х.
    (Опорные кадры - линейчатые спектры газов)
    ref = 'mean' - приводить к средним значениям
    ref = 'center' - приводить к значению в центре кадра
    '''
    y, x = np.shape(neon)
    y = np.arange(y)
    x = np.arange(x)

    # За fwhm считаем fwhm (в пикселях) средней (по Y) строки
    fwhm = calc_fwhm(neon[int(len(neon) / 2)])
    print(('fwhm = ', fwhm, 'pix\n') if verbose else '', end='')

    # Пики в каждой строчке (list из ndarray разной длины)
    peaks = list(map(lambda row: find_peaks(row, fwhm, 20), neon))
    print('***all peaks are found***' if verbose else '')
    # Пики, отсортированные по линиям (2D masked array)
    peaks = find_lines(peaks, fwhm, y, verbose)
    print('***lines are found***' if verbose else '')

    # k - 2D коэффициенты полинома
    # mask - какие строчки (по Y) использовать
    k, mask = my_polyfit(y, peaks, 2, 2)
    if verbose:
        plt.imshow(neon)
        plt.plot(peaks, y, '.')
        plt.plot(my_poly(k, y), y)
        plt.show()
    if ref == 'mean':
        mean_peaks = ma.mean(peaks[mask], axis=0)
    elif ref == 'center':
        center = int(np.median(y))
        mean_peaks = peaks[center]
        i = 1
        while (np.sum(mean_peaks.mask) != 0):
            mean_peaks = np.median(peaks[center-i:center+i], axis=0)
            i+=1
    corr = np.polyfit(mean_peaks, k.T, 3)
    corr_map = my_poly(my_poly(corr, x).T, y)

    good_columns = (np.min(corr_map, axis=0) > 0)
    # Умножение для bool - это and!
    good_columns *= (np.max(corr_map, axis=0) < x[-1])

    new_x = x[good_columns].astype('int')
    corr_map = corr_map[:, new_x]
    return corr_map


def calc_fwhm(spec, wl=None, n=3, guess=10):
    if wl is None:
        wl = np.arange(len(spec))
    peaks = sig.find_peaks(spec)[0]
    amps = spec[peaks]
    peaks = peaks[np.argsort(amps)][-n:]
    amps = amps[np.argsort(amps)][-n:]
    fwhm = np.average(list(map(lambda x, A: one_peak_fwhm(x, A, wl, spec, guess),
                               wl[peaks], amps)))
    return fwhm


def one_peak_fwhm(x, A, wl, spec, guess=1):
    rng = (wl > x - guess) & (wl < x + guess)
    return 2.355 * np.abs(opt.curve_fit(gauss, wl[rng], spec[rng],
                                        p0=[guess, x, A])[0][0])


def find_peaks(spec, fwhm=0, h=1, acc=True):
    '''Ищет пики выше заданного уровня h относительно медианы.
    Затем удаляет из списка пики, у которых есть соседи ближе fwhm'''
    #spec = spec-np.min(spec)
    spec = spec / np.median(spec)
    # plt.plot(spec)
    # plt.plot(np.ones(len(spec))*np.median(spec))
    # plt.show()
    pks = sig.find_peaks(spec, height=h)[0]
    if acc:
        pks = np.array(list(map(lambda x: xmax(spec, x, fwhm=fwhm), pks)))
        pks = pks[pks > 0]
    mask = np.append(np.diff(pks) < fwhm, False)
    mask = mask + np.append([False], np.diff(pks) < fwhm)
    try:
        pks = pks[np.logical_not(mask)]
        return(pks[::])
    except IndexError:
        return []


def find_lines(peaks, fwhm, y=None, verbose=False):
    if y is None:
        y = np.arange(len(peaks))
    # Делаем все строки одинаковой длины (по наидленнейшей)
    peaks = np.array(list(zip_longest(*peaks)), dtype='float')
    # if verbose:
    #     plt.plot(peaks.T, y, 'o')
    #     plt.show()
    msk = np.isnan(peaks)
    peaks = ma.array(peaks, mask=msk)
    col = ['C' + str(j) for j in range(9)]
    #     print(len(peaks))
    #     print()
    for i in range(len(peaks)):
        fuck = peaks[i:]
        line = fuck[0]
    #     msk = np.logical_not(np.isnan(line))
    #     k = ma.polyfit(y, line, 2)
    #     print(k)
        est = np.ones(len(y)) * ma.median(line)
    #     est = np.polyval(k, y)
        err = est - line
        move_right = ma.filled((err > 5 * ma.median(ma.abs(err))), False)
        move_left = ma.filled((err < -5 * ma.median(ma.abs(err))), False)
        not_move = np.logical_not(move_right + move_left)
        # plt.plot(y[not_move], fuck[0][not_move], '.' + col[i % 9])
        # plt.plot(y, est, col[i % 9], ls='--')
        # plt.plot(y[move_right], fuck[0][move_right], 'x' + col[i % 9])
        # plt.plot(y[move_left], fuck[0][move_left], '+' + col[i % 9])
        # plt.show()

    #         print(i)
    #         print(ma.mean(ma.abs(err)))
    #         print(ma.median(line))
    #         print()
        if np.sum(move_right) > 0:  # Те, что меньше медианы (слева)
            nonearray = ma.array([[None] * np.sum(move_right.astype('int'))], mask=[[True] * np.sum(move_right.astype('int'))])
            fuck[:, move_right] = ma.append(fuck[:, move_right][1:, :], nonearray, axis=0)
        if np.sum(move_left) > 0:
            nonearray = ma.array([[None] * np.sum(move_left.astype('int'))], mask=[[True] * np.sum(move_left.astype('int'))])
            fuck[:, move_left] = ma.append(nonearray, fuck[:, move_left][:-1, :], axis=0)
    #     plt.plot(fuck[0], col[i%9])
        peaks[i:] = fuck
    plt.show()
    peaks = peaks.T
    msk = np.isnan(peaks)
    peaks = ma.array(peaks, mask=msk)
    good_lines = (np.sum(np.logical_not(msk), axis=0) > len(y) / 4.)
    peaks = peaks[:, good_lines]
    return peaks


def fit_neon(data, p, a, mode='wl'):
    '''mode = wl - вернуть длины волн
    mode = k - вернуть коэффициенты
    '''
    data = data / np.median(data) # Нормировка
    # plt.plot(data)
    # plt.show()
    fwhm = calc_fwhm(data)
    data_peaks = find_peaks(data, fwhm, h=20) # Поиск основных пиков
    # print(len(data_peaks))
    data_amp = data[data_peaks.astype('int')] # Примерная высота пиков

    n = 4   # Количество пиков, используемых для первого приближения
    # Пики для первого приближения (координата в пикселях)
    ref_pix = np.sort(data_peaks[np.argsort(data_amp)[-n:]])
    # Пики для первого приближения (длина волны)
    ref_lam = np.sort(p[np.argsort(a)[-n:]])
    # print(ref_lam)
    # print(ref_pix)
    # Первое приближение полиномиального преобразования
    k = np.polyfit(ref_pix, ref_lam, n-1)
    # Координаты пикселей после первого приближения
    ref_pix = np.polyval(k, data_peaks)
    # Пики для второго приближения (длина волны)
    ref_lam = p[a * data.max() > data_amp.min()]
    # Первое приближение (длины волн)
    lam = np.polyval(k, np.arange(len(data)))
    # plt.plot(lam - k[-1], data, '--')

    # Избавление от близко стоящих пиков
    fwhm_l = calc_fwhm(data, wl=lam)
    mask = np.append(np.diff(ref_lam) < fwhm_l * 2, False)
    mask = mask + np.append([False], np.diff(ref_lam) < fwhm_l * 2)
    ref_lam = ref_lam[np.logical_not(mask)]

    # Поиск соответствий между пиками
    shape = (len(ref_pix), 1)
    mask = np.argmin(np.abs(np.tile(ref_lam, shape) - ref_pix.reshape(shape)), axis=1)
    ref_lam = ref_lam[mask]
    # Переход от первого приближения ко второму
    k2 = my_polyfit(ref_pix, ref_lam, 1, degatt=1)[0]
    if (mode == 'k'):
        return(k, k2)
    wl = np.polyval(k2, lam)
    # plt.plot(wl - wl.min(), data)
    # plt.show()
    # plt.plot(wl, data / np.max(data))
    # plt.plot(p, a, '.')
    # plt.show()
    return(wl)


def my_polyfit(x, y, deg, degatt=0):
    k = ma.polyfit(x, y, degatt)
    res = my_poly(k, x)
    resid = ma.abs(res - y)
    medresid = ma.median(resid, axis=0)
    if y.ndim != 1:
        mask = ma.logical_not(ma.sum((resid > 3 * medresid), axis=1).astype('bool'))
    else:
        mask = (resid < 3 * medresid)
    y = y[mask]
    x = x[mask]
    k = ma.polyfit(x, y, deg)
    return(k, mask)


def xmax(spec, guess, wl=None, fwhm=5, w=2):
    if wl is None:
        wl = np.arange(len(spec))
    ind = ((wl > guess - fwhm * w) & (wl < guess + fwhm * w))
    try:
        fit = opt.curve_fit(gauss, wl[ind], spec[ind], p0=[fwhm, guess, np.max(spec[ind])])[0][1]
    except RuntimeError:
        fit = 0
    return fit


def my_poly(p, y):
    '''Applying polinomial to an array of values.

    //MORE DETAILED DESCRIPTION IS COMING///

    Parameters
    ----------
    p : ndarray
        Vector or matrix of polinomial coefficients
    y : float or ndarray
        Value or an array of values to which polinomial
        will be applied.

    Returns
    -------
    k : float or array of floats
        Result:
        p - vector, y - float -> float
        p - matrix, y - float -> vector
        p - vector, y - vector -> vector
        p - matrix, y - vector -> matrix
    '''
    n = len(p)
    m = len(y)
    pow_arr = np.arange(n - 1, -1, -1)
    y = np.ones((n, m)) * y
    y_powered = np.power(y.T, pow_arr)
    return np.dot(y_powered, p)


def gauss(x, s, x0, A):
    return A * np.exp(-(x - x0)**2 / (2 * s**2))
