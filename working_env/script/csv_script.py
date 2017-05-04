"""Gr<aph drawing test Script."""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_tw_2_transf_15:23:20.340990.txt")

mmd.create_fmeasure_curve(data)
