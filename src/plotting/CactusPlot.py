#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## cactus.py
##
##  Created on: Jun 05, 2015
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
# ==============================================================================
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
import six
from matplotlib import __version__ as mpl_version

from src.plotting.Plot import Plot


#
# ==============================================================================
class Cactus(Plot, object):
    """
        Cactus plot class.
    """

    def __init__(self, options):
        """
            Cactus constructor.
        """

        super(Cactus, self).__init__(options)

        with open(self.def_path, 'r') as fp:
            self.linestyles = json.load(fp)['cactus_linestyle']
    
    def create(self, data):
        #seaborn.set_style('whitegrid')
        lines  = seaborn.lineplot(data=data,x='rank', y='runtime', hue="Solver", style="Solver", markers=True, dashes=False)

        # turning the grid on
        if not self.no_grid:
            plt.grid(True, color=self.grid_color, ls=self.grid_style, lw=self.grid_width, zorder=1)

        plt.suptitle(self.title)

        # axes labels
        if self.x_label:
            plt.xlabel(self.x_label)
        else:
            plt.xlabel('instances')

        if self.y_label:
            plt.ylabel(self.y_label)
        else:
            plt.ylabel('CPU time (s)')
        
        if self.lgd_loc != 'off':
           plt.legend(loc=self.lgd_loc)

        plt.savefig(self.save_to, bbox_inches='tight', transparent=self.transparent)

  