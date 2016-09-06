#!/usr/bin/env python

# for some reason pylint complains about members being undefined :(
# pylint: disable=E1101

# TODO: global palette
# TODO: no auto-sort numerically

import sys
import os
import re
import subprocess
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans

######################################################################

def quantize(image, bits_per_channel=None):

    if bits_per_channel is None:
        bits_per_channel = 6

    assert image.dtype == np.uint8

    shift = 8-bits_per_channel
    halfbin = 1 << (shift - 1)

    return (((image.astype(int) + halfbin) >> shift)
            << shift) + halfbin

######################################################################

def pack_rgb(rgb):

    orig_shape = None

    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3
        orig_shape = rgb.shape[:-1]
    else:
        assert len(rgb) == 3
        rgb = np.array(rgb)

    rgb = rgb.astype(int).reshape((-1, 3))

    packed = (rgb[:, 0] |
              rgb[:, 1] << 8 |
              rgb[:, 2] << 16)

    if orig_shape is None:
        return packed
    else:
        return packed.reshape(orig_shape)

######################################################################

def unpack_rgb(packed):

    orig_shape = None

    if isinstance(packed, np.ndarray):
        assert packed.dtype == int
        orig_shape = packed.shape
        packed = packed.reshape((-1, 1))

    rgb = (packed & 0xff,
           (packed >> 8) & 0xff,
           (packed >> 16) & 0xff)

    if orig_shape is None:
        return rgb
    else:
        return np.hstack(rgb).reshape(orig_shape + (3,))

######################################################################

def get_bg_color(image, bits_per_channel=None):

    assert image.shape[-1] == 3

    quantized = quantize(image, bits_per_channel).astype(int)
    packed = pack_rgb(quantized)

    unique, counts = np.unique(packed, return_counts=True)

    packed_mode = unique[counts.argmax()]

    return unpack_rgb(packed_mode)

######################################################################

def rgb_to_sv(rgb):

    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)

    axis = len(rgb.shape)-1
    cmax = rgb.max(axis=axis).astype(np.float32)
    cmin = rgb.min(axis=axis).astype(np.float32)
    delta = cmax - cmin

    saturation = delta.astype(np.float32) / cmax.astype(np.float32)
    saturation = np.where(cmax == 0, 0, saturation)

    value = cmax/255.0

    return saturation, value

######################################################################

def nearest(pixels, centers):

    pixels = pixels.astype(int)
    centers = centers.astype(int)

    num_pixels = pixels.shape[0]
    num_channels = pixels.shape[1]
    num_centers = centers.shape[0]

    assert centers.shape[1] == num_channels

    dists = np.empty((num_pixels, num_centers), dtype=pixels.dtype)

    for i in range(num_centers):
        dists_i = pixels - centers[i].reshape((1, num_channels))
        dists[:, i] = (dists_i**2).sum(axis=1)

    return dists.argmin(axis=1)

######################################################################

def crush(output_filename, crush_filename):

    spargs = ['pngcrush', '-q',
              output_filename,
              crush_filename]

    print '  pngcrush -> {}...'.format(output_filename),
    sys.stdout.flush()

    result = subprocess.call(spargs)

    if result == 0:

        before = os.stat(output_filename).st_size
        after = os.stat(crush_filename).st_size

        print '{:.1f}% reduction'.format(100*(1.0-float(after)/before))

        return True

    else:

        print 'warning: pngcrush failed!'
        return False

######################################################################

def percent(string):
    return float(string)/100.0

######################################################################

def parse_args():

    parser = ArgumentParser(
        description='convert scanned, hand-written notes to PDF')

    show_default = ' (default %(default)s)'

    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='files to convert')

    parser.add_argument('-b', dest='basename', metavar='BASENAME',
                        default='output_page_',
                        help='output PNG filename base' + show_default)

    parser.add_argument('-o', dest='pdfname', metavar='PDF',
                        default='output.pdf',
                        help='output PDF filename' + show_default)

    parser.add_argument('-v', dest='value_threshold', metavar='PERCENT',
                        type=percent, default='25',
                        help='background value threshold %%'+show_default)

    parser.add_argument('-s', dest='sat_threshold', metavar='PERCENT',
                        type=percent, default='20',
                        help='background saturation '
                        'threshold %%'+show_default)

    parser.add_argument('-n', dest='num_colors', type=int,
                        default='8',
                        help='number of output colors '+show_default)

    parser.add_argument('-p', dest='sample_fraction',
                        metavar='PERCENT',
                        type=percent, default='5',
                        help='%% of pixels to sample' + show_default)

    parser.add_argument('-C', dest='crush', action='store_false',
                        default=True, help='do not run pngcrush')

    parser.add_argument('-S', dest='saturate', action='store_false',
                        default=True, help='do not saturate colors')

    parser.add_argument('-w', dest='white_bg', action='store_true',
                        default=False, help='make background white')

    return parser.parse_args()

######################################################################

def get_filenames(options):

    filenames = []

    for filename in options.filenames:
        basename = os.path.basename(filename)
        root, _ = os.path.splitext(basename)
        matches = re.findall(r'[0-9]+', root)
        if matches:
            num = int(matches[-1])
        else:
            num = -1
        filenames.append((num, filename))

    del options.filenames

    return [fn for (_, fn) in sorted(filenames)]

######################################################################

def load(input_filename):

    pil_img = Image.open(input_filename)

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    if pil_img.info.has_key('dpi'):
        dpi = pil_img.info['dpi']
    else:
        dpi = (300, 300)

    img = np.array(pil_img)

    return img, dpi

######################################################################

def sample_pixels(img, options):

    pixels = img.reshape((-1, 3))
    num_pixels = pixels.shape[0]
    num_samples = int(num_pixels*options.sample_fraction)

    idx = np.arange(num_pixels)
    np.random.shuffle(idx)

    return pixels[idx[:num_samples]]

######################################################################

def get_fg_mask(bg_color, samples, options):

    s_bg, v_bg = rgb_to_sv(bg_color)
    s_samples, v_samples = rgb_to_sv(samples)

    s_diff = np.abs(s_bg - s_samples)
    v_diff = np.abs(v_bg - v_samples)

    return ((v_diff >= options.value_threshold) |
            (s_diff >= options.sat_threshold))

######################################################################

def get_palette(samples, options):

    print '  getting palette...'

    bg_color = get_bg_color(samples, 6)

    fg_mask = get_fg_mask(bg_color, samples, options)

    centers, _ = kmeans(samples[fg_mask].astype(np.float32),
                        options.num_colors-1,
                        iter=40)

    return np.vstack((bg_color, centers)).astype(np.uint8)

######################################################################

def apply_palette(img, palette, options):

    print '  applying palette...'

    bg_color = palette[0]

    fg_mask = get_fg_mask(bg_color, img, options)

    orig_shape = img.shape

    pixels = img.reshape((-1, 3))
    fg_mask = fg_mask.flatten()

    num_pixels = pixels.shape[0]

    labels = np.zeros(num_pixels, dtype=np.uint8)

    labels[fg_mask] = nearest(pixels[fg_mask], palette)

    return labels.reshape(orig_shape[:-1])

######################################################################

def save(output_filename, labels, palette, dpi, options):

    print '  saving {}...'.format(output_filename)

    if options.saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin)/(pmax-pmin)
        palette = palette.astype(np.uint8)

    if options.white_bg:
        palette = palette.copy()
        palette[0] = (255, 255, 255)

    output_img = Image.fromarray(labels, 'P')
    output_img.putpalette(palette.flatten())
    output_img.save(output_filename, dpi=dpi)

######################################################################

def notescan_main():

    options = parse_args()

    filenames = get_filenames(options)

    outputs = []

    do_pngcrush = options.crush
    if do_pngcrush and subprocess.call(['pngcrush', '-q']) != 0:
        print 'warning: no working pngcrush found!'

    for input_filename in filenames:

        try:
            img, dpi = load(input_filename)
        except IOError:
            print 'warning: error opening ' + input_filename
            continue

        output_basename = '{}{:04d}'.format(options.basename, len(outputs))
        output_filename = output_basename + '.png'
        crush_filename = output_basename + '_crush.png'

        print 'opened', input_filename

        samples = sample_pixels(img, options)

        palette = get_palette(samples, options)

        labels = apply_palette(img, palette, options)

        save(output_filename, labels, palette, dpi, options)

        if do_pngcrush:
            if crush(output_filename, crush_filename):
                output_filename = crush_filename
            else:
                do_pngcrush = False

        outputs.append(output_filename)
        print '  done\n'

    if subprocess.call(['convert'] + outputs + [options.pdfname]) == 0:
        print 'wrote', options.pdfname

######################################################################

if __name__ == '__main__':

    notescan_main()
