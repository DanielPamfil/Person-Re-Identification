# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('.')
from fastreid.utils.visualizer import Visualizer

if __name__ == "__main__":
    #baseline_res = Visualizer.load_roc_info("logs/duke_vis/roc_info.pickle")
    baseline_res = Visualizer.load_roc_info("logs/moco/latest/roc_info.pickle")
    baseline_res_2 = Visualizer.load_roc_info("logs/moco/cuhk03/low_test/roc_info.pickle")
    #mgn_res = Visualizer.load_roc_info("logs/mgn_duke_vis/roc_info.pickle")
    #mgn_res = Visualizer.load_roc_info("logs/moco/roc_info.pickle")

    #fig = Visualizer.plot_roc_curve(baseline_res['fpr'], baseline_res['tpr'], name='baseline')
    fig = Visualizer.plot_roc_curve_new(baseline_res['fpr'], baseline_res['tpr'])

    #Visualizer.plot_roc_curve(mgn_res['fpr'], mgn_res['tpr'], name='mgn', fig=fig)
    #print('fpr', baseline_res['fpr'], 'tpr', baseline_res['tpr'])
    plt.savefig('roc_maket_plot.jpg')

    fig = Visualizer.plot_distribution(baseline_res['pos'], baseline_res['neg'], name='baseline')
    #Visualizer.plot_distribution(mgn_res['pos'], mgn_res['neg'], name='mgn', fig=fig)
    plt.savefig('dist.jpg')
