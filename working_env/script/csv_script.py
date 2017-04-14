""" GRaph drawing test Script"""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_basic_cycle_11:12:43.234093.txt")

mmd.create_fmeasure_curve(data)
