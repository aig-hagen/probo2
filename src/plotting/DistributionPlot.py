import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
import six
from matplotlib import __version__ as mpl_version

from src.plotting.Plot import Plot

class Distribution(Plot, object):
    """
        Distribution plot class.
    """

    def __init__(self, options):
        """
            Distribution constructor.
        """

        super(Distribution, self).__init__(options)

        with open(self.def_path, 'r') as fp:
            self.linestyles = json.load(fp)['cactus_linestyle']
    def create(self,data):
        print("Creating distplot")
        distribution_plot = seaborn.kdeplot(data)

        plt.suptitle(self.title)
        if self.x_label:
            plt.xlabel(self.y_label)
        else:
            plt.xlabel('CPU time (s)')

        # turning the grid on
        if not self.no_grid:
            plt.grid(True, color=self.grid_color, ls=self.grid_style, lw=self.grid_width, zorder=1)

        plt.savefig(self.save_to, bbox_inches='tight', transparent=self.transparent)
        plt.clf()