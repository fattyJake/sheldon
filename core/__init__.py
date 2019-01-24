# -*- coding: utf-8 -*-
###############################################################################
# Module:      core
# Description: repo of essential parsing tools
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

import json as std_json
import logging

from . import xml
from . import _core


# Initialize the logging module
logging.getLogger(__name__).addHandler(logging.NullHandler())


def json():
    raise NotImplementedError()


def parse_data(source):
#    source_stripped = strip_whitespace(source)

#    if source_stripped.startswith('<?xml'):
    return xml.parse(source)

strip_whitespace = _core.strip_whitespace
