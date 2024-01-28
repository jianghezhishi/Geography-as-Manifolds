# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:14:37 2023

@author: 86136
"""

from empirical import read_cp,multiscale_while,plot_multiscale
file='test6_mat.csv'
cp_d=read_cp(file)
minratio=0.8
maxd0=2
res=multiscale_while(minratio,cp_d,maxd0,inf=5,change=0.1)
plot_multiscale(res)