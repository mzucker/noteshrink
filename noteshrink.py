#!/usr/bin/env python

'''Converts sequence of images to compact PDF while removing speckles,
bleedthrough, etc.

'''

# for some reason pylint complains about members being undefined :(
# pylint: disable=E1101

from __future__ import print_function

import sys
import os
import re
import subprocess
import shlex

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.morphology import binary_opening, square
from scipy.ndimage import median_filter

######################################################################

def quantize_colors(image, bits_per_channel=6):
    """
    Reduces the number of colors in an image by reducing the number of bits per channel.

    Args:
        image (numpy.ndarray): The input image.
        bits_per_channel (int): The number of bits per channel (default: 6).

    Returns:
        numpy.ndarray: The quantized image.
    """

    assert image.dtype == np.uint8

    shift = 8-bits_per_channel
    halfbin = 2**(shift - 1)

    # Truncate last shift bits and add half the clipped bin
    return (np.left_shift(np.right_shift(image, shift), shift) + halfbin).astype('uint8')

######################################################################

def pack_rgb(rgb):
    """
    Packs a 24-bit RGB triple into a single integer.

    Args:
        rgb (numpy.ndarray or tuple): The RGB values.

    Returns:
        int or numpy.ndarray: The packed RGB values.
    """

    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3 # RGB array must have 3 channels
        rgb = rgb.astype(np.uint32)
    else:
        assert len(rgb) == 3 # RGB tuple must have 3 channels
        rgb = np.array(rgb, dtype=np.uint32)

    packed = (rgb[:, 0] << 16 |
              rgb[:, 1] << 8 |
              rgb[:, 2])

    return packed

######################################################################

def unpack_rgb(packed):
    """
    Unpacks a single integer or array of integers into one or more 24-bit RGB values.

    Args:
        packed (int or numpy.ndarray): The packed RGB values.

    Returns:
        numpy.ndarray: The unpacked RGB values.
    """

    return np.column_stack(((packed >> 16) & 0xff,
                            (packed >> 8) & 0xff,
                            packed & 0xff))

######################################################################

def get_background_color(pixels, bits_per_channel=6):
    """
    Estimates the background color from an image or array of RGB colors by finding the most frequent color in the image.

    Args:
        pixels (numpy.ndarray): The RGB input pixels.
        bits_per_channel (int): The number of bits per channel (default: 6).

    Returns:
        numpy.ndarray: An RGB tuple representing the background color.
    """

    assert pixels.shape[-1] == 3

    quantized = quantize_colors(pixels, bits_per_channel).astype(np.uint32)
    packed = pack_rgb(quantized)

    unique, counts = np.unique(packed, return_counts=True)

    packed_mode = unique[counts.argmax()]

    return unpack_rgb(packed_mode)

######################################################################

def rgb_to_sv(rgb):
    """
    Converts an RGB image or array of RGB colors to saturation and value, returning each one as a separate
    32-bit floating point array or value.

    Args:
        rgb (numpy.ndarray): The input RGB values.

    Returns:
        tuple: A tuple containing the saturation and value arrays or values.
    """

    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)

    rgb = rgb.reshape((-1,3))

    cmin = rgb.min(axis=1)
    cmax = rgb.max(axis=1)
    saturation = np.where(cmax == 0, 0, 1 - cmin/cmax)  # Handle division by zero

    value = cmax/255.0

    return saturation, value

######################################################################

def postprocess(output_filename, options):
    """
    Runs a postprocessing command on the provided file.

    Args:
        output_filename (str): The output filename.
        options (argparse.Namespace): The command-line options.

    Returns:
        str or None: The postprocessed filename if successful, None otherwise.
    """


    assert options.postprocess_cmd

    base, _ = os.path.splitext(output_filename)
    post_filename = base + options.postprocess_ext

    cmd = options.postprocess_cmd
    cmd = cmd.replace('%i', output_filename)
    cmd = cmd.replace('%o', post_filename)
    cmd = cmd.replace('%e', options.postprocess_ext)

    subprocess_args = shlex.split(cmd)

    if os.path.exists(post_filename):
        os.unlink(post_filename)

    if not options.quiet:
        print('  running "{}"...'.format(cmd), end=' ')
        sys.stdout.flush()

    try:
        result = subprocess.call(subprocess_args)
        before = os.stat(output_filename).st_size
        after = os.stat(post_filename).st_size
    except OSError:
        result = -1

    if result == 0:

        if not options.quiet:
            print('{:.1f}% reduction'.format(
                100*(1.0-float(after)/before)))

        return post_filename

    else:

        sys.stderr.write('warning: postprocessing failed!\n')
        return None

######################################################################

def percent(string):
    """
    Converts a string (e.g., '85') to a fraction (e.g., 0.85).

    Args:
        string (str): The input string.

    Returns:
        float: The converted fraction.
    """
    return float(string) / 100.0

######################################################################

def get_argument_parser():
    """
    Parses the command-line arguments for the program.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='convert scanned, hand-written notes to PDF')

    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='files to convert')

    parser.add_argument('-q', dest='quiet', action='store_true',
                        default=False,
                        help='reduce program output')

    parser.add_argument('-b', dest='basename', metavar='BASENAME',
                        default='page',
                        help='output PNG filename base')

    parser.add_argument('-o', dest='pdfname', metavar='PDF',
                        default='output.pdf',
                        help='output PDF filename')

    parser.add_argument('-v', dest='value_threshold', metavar='PERCENT',
                        type=percent, default='25',
                        help='background value threshold %%')

    parser.add_argument('-s', dest='sat_threshold', metavar='PERCENT',
                        type=percent, default='15',
                        help='background saturation '
                        'threshold %%')

    parser.add_argument('-n', dest='num_colors', type=int,
                        default='8',
                        help='number of output colors')

    parser.add_argument('-p', dest='sample_fraction',
                        metavar='PERCENT',
                        type=percent, default='5',
                        help='%% of pixels to sample')

    parser.add_argument('-w', dest='white_bg', action='store_true',
                        default=False, help='make background white')

    parser.add_argument('-g', dest='global_palette',
                        action='store_true', default=False,
                        help='use one global palette for all pages')

    parser.add_argument('-S', dest='saturate', action='store_false',
                        default=True, help='do not saturate colors')

    parser.add_argument('-K', dest='sort_numerically',
                        action='store_false', default=True,
                        help='keep filenames ordered as specified; '
                        'use if you *really* want IMG_10.png to '
                        'precede IMG_2.png')

    parser.add_argument('--denoise-median', default=False, action='store_true',
                        help='Median filtering the output image with kernel size --denoise-median-strength.')

    parser.add_argument('--denoise-opening', default=False, action='store_true',
                        help='Perform opening (erosion followed by a dilation) of the binary background mask with kernel size --denoise-opening-strength. This replaces speckles with background.')

    parser.add_argument('--denoise-median-strength', default=3, type=int,
                        help='Denoising strength [1, 2, 3, ...]. Size of the filter kernel used.')

    parser.add_argument('--denoise-opening-strength', default=3, type=int,
                        help='Denoising strength [1, 2, 3, ...]. Size of the filter kernel used.')

    parser.add_argument('-P', dest='postprocess_cmd', default=None,
                        help='set postprocessing command (see -O, -C, -Q)')

    parser.add_argument('-e', dest='postprocess_ext',
                        default='_post.png',
                        help='filename suffix/extension for '
                        'postprocessing command')

    parser.add_argument('-O', dest='postprocess_cmd',
                        action='store_const',
                        const='optipng -silent %i -out %o',
                        help='same as -P "%(const)s"')

    parser.add_argument('-C', dest='postprocess_cmd',
                        action='store_const',
                        const='pngcrush -q %i %o',
                        help='same as -P "%(const)s"')

    parser.add_argument('-Q', dest='postprocess_cmd',
                        action='store_const',
                        const='pngquant --ext %e %i',
                        help='same as -P "%(const)s"')

    parser.add_argument('-c', dest='pdf_cmd', metavar="COMMAND",
                        default='convert %i %o',
                        help='PDF command (default "%(default)s")')

    return parser

######################################################################

def get_filenames(options):
    """
    Gets the filenames from the command line, optionally sorted numerically.
    (..., image10, image9, ...) --> (..., image9, image10, ...)

    Args:
        options (argparse.Namespace): The command-line options.

    Returns:
        list: The sorted filenames.
    """

    if not options.sort_numerically:
        return options.filenames

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

    return [fn for (_, fn) in sorted(filenames)]

######################################################################

def load(input_filename):
    """
    Loads an image with Pillow and converts it to a numpy pixel array.

    Args:
        input_filename (str): The input filename.

    Returns:
        tuple: A tuple containing the pixel array, DPI (x, y), and image shape.
    """

    pil_img = Image.open(input_filename)

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    if 'dpi' in pil_img.info:
        dpi = pil_img.info['dpi']
    else:
        dpi = (300, 300)

    img = np.array(pil_img)

    return img.reshape((-1, img.shape[-1])), dpi, img.shape

######################################################################

def sample_pixels(pixels, options):
    """
    Picks a fixed percentage of pixels in the image, returned in random order.

    Args:
        pixels (numpy.ndarray): The input pixels.
        options (argparse.Namespace): The command-line options.

    Returns:
        numpy.ndarray: The sampled pixels.
    """

    num_pixels = pixels.shape[0]
    num_samples = int(round(num_pixels*options.sample_fraction, 0))

    return shuffle(pixels, random_state=0, n_samples=num_samples)

######################################################################

def get_fg_mask(bg_color, samples, options):
    """
    Determines whether each pixel in a set of samples is foreground by comparing it to the background color.

    Args:
        bg_color (numpy.ndarray): The background color.
        samples (numpy.ndarray): The pixel samples.
        options (argparse.Namespace): The command-line options.

    Returns:
        numpy.ndarray: The foreground mask.
    """

    s_bg, v_bg = rgb_to_sv(bg_color)
    s_samples, v_samples = rgb_to_sv(samples)

    s_diff = np.abs(s_bg - s_samples)
    v_diff = np.abs(v_bg - v_samples)

    return ((v_diff >= options.value_threshold) |
            (s_diff >= options.sat_threshold))

######################################################################

def get_fit(pixels, options, return_mask=False):
    """
    Extracts the palette for the set of sampled RGB values.

    Args:
        pixels (numpy.ndarray): The RGB pixel values to be sampled.
        options (argparse.Namespace): The command-line options.
        return_mask (bool): Whether to return the foreground mask (default: False).

    Returns:
        tuple: A tuple containing the color palette, background color, and optionally the foreground mask.
    """

    if not options.quiet:
        print('  getting palette...')

    samples = sample_pixels(pixels, options)

    bg_color = get_background_color(samples, 6)

    fg_mask = get_fg_mask(bg_color, samples, options)

    # Fit model to data sample
    fit = KMeans(n_clusters=options.num_colors-1,
                     random_state=0,
                     n_init='auto'
                    ).fit(samples[fg_mask])

    if not return_mask:
        return fit, bg_color
    else:
        return fit, bg_color, fg_mask

######################################################################

def apply_quantization(pixels, fit, bg_color, shape, options):
    """
    Applies color quantization and background removal to the image.

    :param pixels (numpy.ndarray): The picture pixels
    :param fit (sklearn.cluster._kmeans.KMeans): Trained prediction model
    :param bg_color (numpy.ndarray): The background color
    :param options (argparse.Namespace): Command line options.
    :return: None
    """

    if not options.quiet:
        print('  applying palette...')

    # get pixel mask with pixels corresp. to bg_color
    fg_mask_full = get_fg_mask(bg_color, pixels, options)

    if options.denoise_opening:
        ker = square(options.denoise_opening_strength)
        fg_mask_full = binary_opening(fg_mask_full.reshape(shape[:-1]), ker).flatten()

    # init color-labels with 0 corresp. to bg_color
    labels = np.zeros(pixels.shape[0], dtype='uint8')

    # predict non-bg_color pixels to color-labels
    # (0 corresp. to bg_color)
    labels[fg_mask_full] = fit.predict(pixels[fg_mask_full]) + 1

    fit.cluster_centers_ = fit.cluster_centers_.round(0)
    palette = np.vstack((bg_color, fit.cluster_centers_.astype('uint8')))

    return labels, palette

######################################################################

def save(output_filename, labels, palette, shape, dpi, options):
    """
    Saves the label/palette pair as an indexed PNG image.

    Args:
        output_filename (str): The output filename.
        labels (numpy.ndarray): The color labels.
        palette (numpy.ndarray): The color palette.
        shape (tuple): The image shape.
        dpi (tuple): The image DPI (x, y).
        options (argparse.Namespace): The command-line options.
    """

    if not options.quiet:
        print('  saving {}...'.format(output_filename))

    if options.saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin)/(pmax-pmin)
        palette = palette.round(0).astype(np.uint8)

    if options.white_bg:
        palette = palette.copy()
        palette[0] = (255, 255, 255)


    if options.denoise_median:
        # Median filtering is per color channel. In RGB space this would lead to color deviations.
        output_img = Image.fromarray(palette[labels].reshape(shape)).convert('HSV')
        output_img = median_filter(output_img, size=(options.denoise_median_strength, options.denoise_median_strength, 1))
        output_img = Image.fromarray(output_img, 'HSV').convert('RGB')
    else:
        output_img = Image.fromarray(palette[labels].reshape(shape), 'RGB')

    output_img.save(output_filename, dpi=dpi)

######################################################################

def get_global_fit(filenames, options):
    """
    Extracts the palette of a list of input files by sampling from all of them.

    Args:
        filesnames (list): List of file names as strings
        options (argparse.Namespace): The command-line options.

    Returns:
        tuple: A tuple containing the color palette, background color, and optionally the foreground mask.
    """

    all_samples = []

    if not options.quiet:
        print('building global palette...')

    for input_filename in filenames:

        pixels, _, _ = load(input_filename)

        if not options.quiet:
            print('  processing {}...'.format(input_filename))

        samples = sample_pixels(pixels, options)
        all_samples.append(samples)

    num_inputs = len(filenames)

    all_samples = [s[:int(round(float(s.shape[0])/num_inputs))]
                   for s in all_samples]

    all_samples = np.vstack(tuple(all_samples)).astype('uint8')

    global_fit, bg_color = get_fit(all_samples, options, return_mask=False)

    if not options.quiet:
        print('  done\n')

    return global_fit, bg_color

######################################################################

def emit_pdf(outputs, options):
    '''
    Runs the PDF conversion command to generate the PDF.

    Args:
        outputs (list): List of file names as strings
        options (argparse.Namespace): The command-line options.
    '''

    cmd = options.pdf_cmd
    cmd = cmd.replace('%o', options.pdfname)
    if len(outputs) > 2:
        cmd_print = cmd.replace('%i', ' '.join(outputs[:2] + ['...']))
    else:
        cmd_print = cmd.replace('%i', ' '.join(outputs))
    cmd = cmd.replace('%i', ' '.join(outputs))

    if not options.quiet:
        print('running PDF command "{}"...'.format(cmd_print))

    try:
        result = subprocess.call(shlex.split(cmd))
    except OSError:
        result = -1

    if result == 0:
        if not options.quiet:
            print('  wrote', options.pdfname)
    else:
        sys.stderr.write('warning: PDF command failed\n')

######################################################################

def notescan_main(options):
    """
    Main function for the notescan program.

    Args:
        options (argparse.Namespace): The command-line options.
    """

    filenames = get_filenames(options)

    do_global = options.global_palette and len(filenames) > 1
    do_postprocess = bool(options.postprocess_cmd)

    outputs = []

    if do_global:
        fit, bg_color = get_global_fit(filenames, options)

    for input_filename in filenames:

        pixels, dpi, shape = load(input_filename)

        output_filename = '{}{:04d}.png'.format(
            options.basename, len(outputs))

        if not options.quiet:
            print('opened', input_filename)

        if not do_global:
            fit, bg_color = get_fit(pixels, options)

        labels, palette = apply_quantization(pixels=pixels, fit=fit, bg_color=bg_color, shape=shape, options=options)

        save(output_filename, labels, palette, shape, dpi, options)

        if do_postprocess:
            post_filename = postprocess(output_filename, options)
            if post_filename:
                output_filename = post_filename
            else:
                do_postprocess = False

        outputs.append(output_filename)

        if not options.quiet:
            print('  done\n')

    emit_pdf(outputs, options)

######################################################################

def main():
    '''Parse args and call notescan_main().'''
    notescan_main(options=get_argument_parser().parse_args())

if __name__ == '__main__':
    main()
