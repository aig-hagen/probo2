import matplotlib.pyplot as plt
import numpy as np
import os

class Plot:
  """
      Basic plotting class.
  """

  def __init__(self, options):
    """
        Constructor.
    """
    settings = options['settings']
    self.options = settings
    self.plot_type = settings['plot_type']
    self.alpha = settings['alpha']
    self.backend = settings['backend']
    self.save_to = settings['save_to']
    self.def_path = options['def_path']
    self.transparent = settings['transparent']

    self.timeout = settings['timeout']
    self.t_label = settings['t_label']
    self.tlb_loc = settings['tlb_loc']
    self.title = settings['title']

    self.x_label = settings['x_label']
    self.x_log = settings['x_log']
    self.x_max = settings['x_max']
    self.x_min = settings['x_min']
    self.y_label = settings['y_label']
    self.y_log = settings['y_log']
    self.y_max = settings['y_max']
    self.y_min = settings['y_min']
    self.axis_scale = settings['axis_scale']

    self.lgd_loc = settings['lgd_loc']
    self.lgd_ncol = settings['lgd_ncol']
    self.lgd_alpha = settings['lgd_alpha']
    self.lgd_fancy = settings['lgd_fancy']
    self.lgd_shadow = settings['lgd_shadow']

    self.no_grid = settings['no_grid']
    self.grid_color = settings['grid_color']
    self.grid_style = settings['grid_style']
    self.grid_width = settings['grid_width']

    # where to save
    self.save_to = '{0}.{1}'.format(os.path.splitext(self.save_to)[0], self.backend)

    # font properties
    self.f_props = {'serif': ['Times'], 'sans-serif': ['Helvetica'],
                    'weight': 'normal', 'size': settings['font_sz']}

    if settings['font'].lower() in ('sans', 'sans-serif', 'helvetica'):  # Helvetica
      self.f_props['family'] = 'sans-serif'
    elif settings['font'].lower() in ('serif', 'times'):  # Times
      self.f_props['family'] = 'serif'
    elif settings['font'].lower() == 'cmr':  # Computer Modern Roman
      self.f_props['family'] = 'serif'
      self.f_props['serif'] = 'Computer Modern Roman'
    elif settings['font'].lower() == 'palatino':  # Palatino
      self.f_props['family'] = 'serif'
      self.f_props['serif'] = 'Palatino'

    # plt.rc('text', usetex=options['usetex'])
    # plt.rc('font', **self.f_props)

    # figure properties
    nof_subplots = 1
    fig_width_pt = 252.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5) + 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt + 0.2  # width in inches
    fig_height = fig_width / golden_mean * nof_subplots + 0.395 * (nof_subplots - 1)  # height in inches
    if settings['shape'] == 'squared':
      fig_width = fig_height
    elif len(settings['shape']) >= 4 and settings['shape'][:4] == 'long':
      coeff = settings['shape'][4:]
      fig_width *= 1.2 if not coeff else float(coeff)  # default coefficient is 1.2

    fig_size = [fig_width * 2.5, fig_height * 2.5]

    params = {'backend': 'pdf', 'text.usetex': settings['usetex'], 'figure.figsize': fig_size}

    plt.rcParams.update(params)

    # choosing backend
    if self.backend in ('pdf', 'ps', 'svg'):  # default is pdf
      plt.switch_backend(self.backend)
    elif self.backend == 'pgf':  # PGF/TikZ
      pgf_params = {'pgf.texsystem': 'pdflatex',
                    'pgf.preamble': [r'\usepackage[utf8x]{inputenc}', r'\usepackage[T1]{fontenc}']}
      params.update(pgf_params)
      plt.rcParams.update(params)
      plt.switch_backend(self.backend)
    elif self.backend == 'png':
      plt.switch_backend('agg')

    # funny mode
    if settings['xkcd']:
      plt.xkcd()




