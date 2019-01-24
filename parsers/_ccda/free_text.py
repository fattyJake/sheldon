# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for any freetext section (i.e., contains just a single <text> element)
"""

from ... import core
from sheldon.core import wrappers

def free_text(ccda, section_name):

    doc = ccda.section(section_name)
    text = core.strip_whitespace(doc.tag('text').val())
    if isinstance(text, list): text = ' '.join(text)
    text = str(text)

    return wrappers.ObjectWrapper(
        text=text
    )