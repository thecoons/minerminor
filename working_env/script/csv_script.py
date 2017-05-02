""" GRaph drawing test Script"""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_basic_cycle_14:09:18.309328.txt")

mmd.create_fmeasure_curve(data)
