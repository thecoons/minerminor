"""Gr<aph drawing test Script."""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_basic_cycle_12:28:00.799510.txt")

mmd.create_fmeasure_curve(data)
