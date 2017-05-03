""" GRaph drawing test Script"""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_tw_17:39:54.207630.txt")

mmd.create_fmeasure_curve(data)
