#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_analysis
@file: plotting_utils.py
@time: 11/17/2023 4:13 PM
@description: Various plotting utilities for customizing plots.
"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import colors
import colorsys
from scipy.ndimage import gaussian_filter1d


def remove_top_right_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def remove_bottom_right_frame(ax):
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def color_to_rgba(color_name):
    """
    Converts color name to RGB.
    :param color_name:
    :return:
    """

    return colors.to_rgba(color_name)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def adjust_lightness(color, amount=0.5):
    """
    Same as lighten_color but adjusts brightness to lighter color if amount>1 or darker if amount<1.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def make_cmap_n_from_color_lite2dark(color, n_colors):
    """
    Make ListedColormap from matplotlib color of size N using the lighten_color function.
    :param color: Matplotlib color string.
    :param n_colors: Number of colors to have in cmap.
    :return:
    """
    light_factors = np.linspace(0.2, 1, n_colors)
    cmap = colors.ListedColormap(colors=[lighten_color(color, amount=i) for i in light_factors])
    return cmap


def save_figure_to_files(fig, save_path, file_name, suffix=None, file_types=list, dpi=500):
    """
    Save figure to file.
    :param fig: Figure to save.
    :param save_path: Path to save figure.
    :param file_name: Name of file.
    :param suffix: Suffix to add to file name.
    :param file_types: List of file types to save.
    :param dpi: Resolution of figure.
    :return:
    """

    if file_types is None:
        file_types = ['png', 'eps', 'pdf']

    if suffix is not None:
        file_name = file_name + '_' + suffix

    for file_type in file_types:
        file_format = '.{}'.format(file_type)
        file_path = os.path.join(save_path, file_name + file_format)

        print('Saving in: {}'.format(file_path))
        if file_type == 'eps':
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Render a matplotlib table.
    :param data:
    :param col_width:
    :param row_height:
    :param font_size:
    :param header_color:
    :param row_colors:
    :param edge_color:
    :param bbox:
    :param header_columns:
    :param ax:
    :param kwargs:
    :return:
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax.get_figure(), ax

def save_figure_with_options(figure, file_formats, filename, output_dir='', dark_background=False):
    # Make transparent for dark background
    if dark_background:
        figure.patch.set_alpha(0)
        figure.set_facecolor('#f4f4ec')
        for ax in figure.get_axes():
            ax.set_facecolor('#f4f4ec')
        #plt.rcParams.update({'axes.facecolor': '#f4f4ec',  # very pale beige
        #                        'figure.facecolor': '#f4f4ec'})
        transparent = True
        filename = filename + '_transparent'
    else:
        transparent = False

    # Save the figure in each specified file format
    for file_format in file_formats:
        file_path = os.path.join(output_dir, f"{filename}.{file_format}")
        figure.savefig(file_path, transparent=transparent, bbox_inches='tight', dpi='figure')

    return

def apply_y_ticks_to_all_subplots(axs):
    """

    :param axs:
    :return:
    """
    # Determine if axs is 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, find the global maximum y-value
        max_y = float('-inf')
        min_y = float('inf')
        for row in axs:
            for ax in row:
                y_max = ax.get_ylim()[1]
                y_min = ax.get_ylim()[0]
                max_y = max(max_y, y_max)
                min_y = min(min_y, y_min)
        # Set the same y-axis range for all subplots
        for row in axs:
            for ax in row:
                ax.set_ylim(min_y, max_y+1)
                ax.yaxis.set_visible(True)
                ax.tick_params(axis='y', which='both', labelleft=True)
    else:
        # It's a 1D array-like, find the global maximum y-value
        max_y = float('-inf')
        min_y = float('inf')
        for ax in axs:
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            max_y = max(max_y, y_max+1)
            min_y = min(min_y, y_min)
        # Set the same y-axis range for all subplots
        for ax in axs:
            ax.set_ylim(min_y, max_y)
            ax.yaxis.set_visible(True)
            ax.tick_params(axis='y', which='both', labelleft=True)

    return


def apply_y_ticks_per_row(axs):
    """
    Adjusts y-axis limits for each row of subplots independently.

    :param axs: 1D or 2D array-like of Matplotlib Axes.
    """
    # Determine if axs is a 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, process each row separately
        for row in axs:
            min_y, max_y = float('inf'), float('-inf')
            for ax in row:
                y_min, y_max = ax.get_ylim()
                min_y, max_y = min(min_y, y_min), max(max_y, y_max)
            for ax in row:
                ax.set_ylim(min_y, max_y)
                ax.yaxis.set_visible(True)
                ax.tick_params(axis='y', which='both', labelleft=True)
    else:
        # It's a 1D array, process normally
        min_y, max_y = float('inf'), float('-inf')
        for ax in axs:
            y_min, y_max = ax.get_ylim()
            min_y, max_y = min(min_y, y_min), max(max_y, y_max)
        for ax in axs:
            ax.set_ylim(min_y, max_y)
            ax.yaxis.set_visible(True)
            ax.tick_params(axis='y', which='both', labelleft=True)


def align_y_axis(axs):
    """
    Align the y-axis of all subplots in a given row or column.

    Parameters:
    axs : 1D or 2D array-like of matplotlib.axes.Axes
        A 1D or 2D array or list of matplotlib Axes objects.
    """
    # Check if axs is 2D array-like
    if hasattr(axs[0], '__iter__'):
        # It's a 2D array-like, align y-axis for each row
        for row in axs:
            min_y = float('inf')
            max_y = float('-inf')
            for ax in row:
                y_min, y_max = ax.get_ylim()
                min_y = min(min_y, y_min)
                max_y = max(max_y, y_max)

            for ax in row:
                ax.set_ylim(min_y, max_y)
    else:
        # It's a 1D array-like, align y-axis for the whole array
        min_y = float('inf')
        max_y = float('-inf')
        for ax in axs:
            y_min, y_max = ax.get_ylim()
            min_y = -2
            max_y = max(max_y, y_max) * 3

        for ax in axs:
            ax.set_ylim(min_y, max_y)


def half_gaussian_filter(series, sigma, truncate=3.0):
    """
    Apply a causal half-Gaussian filter using gaussian_filter1d, without future information.

    Args:
        series (pd.Series): Input 1D series.
        sigma (float): Standard deviation for Gaussian kernel (in number of samples).
        truncate (float): Truncate the filter at this many sigmas.

    Returns:
        pd.Series: Smoothed series.
    """
    # Pad the beginning to avoid border effects
    pad_width = int(truncate * sigma)
    padded = np.pad(series.values, (pad_width, 0), mode='edge')

    # Apply Gaussian filter
    smoothed = gaussian_filter1d(padded, sigma=sigma, truncate=truncate)

    # Drop padding
    smoothed = smoothed[pad_width:]

    return pd.Series(smoothed, index=series.index)