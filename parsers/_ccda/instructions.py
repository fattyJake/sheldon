# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA instructions section
"""

from ...core import wrappers
from ... import core


def instructions(ccda):

    data = wrappers.ListWrapper()

    instructions = ccda.section('instructions')

    for entry in instructions.entries():

        el = entry.tag('code')
        name = el.attr('displayName')
        code = el.attr('code')
        code_system = el.attr('codeSystem')
        code_system_name = el.attr('codeSystemName')

        text = core.strip_whitespace(entry.tag('text').val())

        data.append(wrappers.ObjectWrapper(
            text=text,
            name=name,
            code=code,
            code_system=code_system,
            code_system_name=code_system_name
        ))

    return data
