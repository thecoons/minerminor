""" GRaph drawing test Script"""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_15_bis_15:05:29.027560.txt")

mmd.create_fmeasure_curve(data)
