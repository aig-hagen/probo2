#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## scatter.py
##
##  Created on: Jun 05, 2015
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
import json
import matplotlib.pyplot as plt
from matplotlib import __version__ as mpl_version
import numpy as np
from src.plotting.Plot import Plot
import six
from six.moves import range
import src.analysis.statistics as Stats
import os
#
#==============================================================================
class ScatterException(Exception):
    pass


#
#==============================================================================
class Scatter(Plot, object):
    """
        Scatter plot class.
    """

    def __init__(self, options):
        """
            Scatter constructor.
        """

        super(Scatter, self).__init__(options)

        # setting up axes limits
        if not self.x_min:
            self.x_min = self.y_min  # self.y_min is supposed to have a default value
        else:
            self.y_min = self.x_min

        if not self.x_max:
            self.x_max = 0
        if not self.y_max:
            self.y_max = 0
        if self.x_max and self.y_max and self.x_max != self.y_max:
            assert 0, 'right-most positions must be the same for X and Y axes'
        elif self.x_max == 0 and self.y_max == 0:
            self.x_max = 10
            while self.x_max < self.timeout:
                self.x_max *= 10
            self.y_max = self.x_max
        else:
            self.x_max = self.y_max = max(self.x_max, self.y_max)

        # setting timeout-line label
        if not self.t_label:
            self.t_label = '{0} sec. timeout'.format(int(self.timeout))

        with open(self.def_path, 'r') as fp:
            self.marker_style = json.load(fp)['scatter_style']

    def create(self, data):
        """
            Does the plotting.
        """

        if len(data[0][1]) != len(data[1][1]):
            raise ScatterException('Number of instances for each competitor must be the same')

        step = int((self.x_max - self.x_min) / 10)
        x = np.arange(self.x_min, self.x_max + self.x_min + step, step)

        # "good" area
        plt.plot(x, x, color='black', ls=':', lw=1.5, zorder=3)
        plt.plot(x, 0.1 * x, 'g:', lw=1.5, zorder=3)
        plt.plot(x, 10 * x, 'g:', lw=1.5, zorder=3)
        plt.fill_between(x, 0.1 * x, 10 * x, facecolor='green', alpha=0.15,
            zorder=3)

        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])

        # timeout lines
        plt.axvline(self.timeout, linewidth=1, color='red', ls=':',
            label=str(self.timeout), zorder=3)
        plt.axhline(self.timeout, linewidth=1, color='red', ls=':',
            label=str(self.timeout), zorder=3)

        if self.tlb_loc == 'after':
            plt.text(2 * self.x_min, self.timeout + self.x_max / 40,
                self.t_label, horizontalalignment='left',
                verticalalignment='bottom', fontsize=self.f_props['size'] * 0.8)
            plt.text(self.timeout + self.x_max / 40, 2 * self.x_min,
                self.t_label, horizontalalignment='left',
                verticalalignment='bottom', fontsize=self.f_props['size'] * 0.8,
                rotation=90)
        else:
            plt.text(2 * self.x_min, self.timeout - self.x_max / 3.5,
                self.t_label, horizontalalignment='left',
                verticalalignment='bottom', fontsize=self.f_props['size'] * 0.8)
            plt.text(self.timeout - self.x_max / 3.5, 2 * self.x_min,
                self.t_label, horizontalalignment='left',
                verticalalignment='bottom', fontsize=self.f_props['size'] * 0.8,
                rotation=90)

        # scatter
        plt.scatter(data[0][1], data[1][1], c=self.marker_style['color'],
            marker=self.marker_style['marker'], s=self.marker_style['size'],
            alpha=self.alpha, zorder=5)

        plt.suptitle(self.title)

        # axes' labels
        if self.x_label:
            plt.xlabel(self.x_label)
        else:
            plt.xlabel(data[0][0])

        if self.y_label:
            plt.ylabel(self.y_label)
        else:
            plt.ylabel(data[1][0])

        # turning the grid on
        if not self.no_grid:
            plt.grid(True, color='black', ls=':', lw=1, zorder=1)

        # choosing logarithmic scales
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')

        # setting ticks font properties
        # set_*ticklables() seems to be not needed in matplotlib 1.5.0
        if float(mpl_version[:3]) < 1.5:
            ax.set_xticklabels(ax.get_xticks(), self.f_props)
            ax.set_yticklabels(ax.get_yticks(), self.f_props)

        # formatter
        majorFormatter = plt.LogFormatterMathtext(base=10)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.yaxis.set_major_formatter(majorFormatter)

        # setting frame thickness
        for i in six.itervalues(ax.spines):
            i.set_linewidth(1)

        plt.savefig(self.save_to, bbox_inches='tight', transparent=self.transparent)
    
    def create_pairwise(self, stat_array, task, benchmark_name):
        num_stat_objs = len(stat_array.stat_objs)
        for i in range(0, num_stat_objs):
                    temp_stat_list = []
                    temp_stat_list.append(stat_array.stat_objs[i])
                    for j in range(i + 1, num_stat_objs):
                        temp_stat_list.append(stat_array.stat_objs[j])
                        temp_stat_array = Stats.StatArray(temp_stat_list)
                        save_file_name = os.path.join(self.options['save_to'], "{}_{}_{}_{}.{}".format(task, benchmark_name,
                                                                                     temp_stat_list[0].meta_data[
                                                                                         'solver']['name'] + 'v' +
                                                                                     temp_stat_list[0].meta_data[
                                                                                         'solver']['version'],
                                                                                     temp_stat_list[1].meta_data[
                                                                                         'solver']['name'] + 'v' +
                                                                                     temp_stat_list[1].meta_data[
                                                                                         'solver']['version'],self.backend))
                       
                       
                        data = temp_stat_array.prepare_plotting(self.options)
                        plotter = Scatter(self.options)
                        plotter.save_to = save_file_name
                        plotter.create(data)
                        del temp_stat_list[1]

