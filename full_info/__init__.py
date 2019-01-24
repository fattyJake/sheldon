#-*- coding: utf-8 -*-
###############################################################################
# this is a python package designed to get full information for a patient
# from inovalons records. This includes personal data, claims (health codes
# as categorical data), ccds (zip or xml files), or scanned documents 
# (a list of strings from pages) relating to that patient.
#
# Authors:  William Kinsman, Chenyu Ha, Yage Wang
# Created:  2018.03.16
# Version:  1.0.0
###############################################################################

# check for dependancies on import
import imp
needed_packages = ''
for i in ['matplotlib','sklearn','numpy','pyodbc','scipy','pickle']:
    try:    imp.find_module(i)
    except: needed_packages = needed_packages+'\n\''+i+'\' python package'
if needed_packages: raise ImportError('\nRequired shakespeare dependencies not detected. Please install: ' + needed_packages)

from . import fetch_db
from . import dumb_parser