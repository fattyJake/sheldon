# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA functional statuses section
"""

from ... import documents
from ...core import wrappers
from ... import core


def functional_statuses(ccda):

    statuses = ccda.section("functional_statuses")

    el = statuses.tag("code")

    name = el.attr("displayName")
    code = el.attr("code")
    code_system = el.attr("codeSystem")
    code_system_name = el.attr("codeSystemName")

    text = None
    text = core.strip_whitespace(statuses.tag("text").val())

    data = wrappers.ObjectWrapper(
        name=name,
        code=code,
        code_system=code_system,
        code_system_name=code_system_name,
        text=text,
    )

    return data
