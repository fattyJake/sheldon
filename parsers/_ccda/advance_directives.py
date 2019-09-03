# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA advance directives section
"""

from sheldon import core
from sheldon import documents
from ...core import wrappers
from ...documents import parse_date


def advance_directives(ccda):

    data = []

    advance_directives = ccda.section("advance_directives")

    for entry in advance_directives.entries():

        name = None
        code = None
        code_system = None
        code_system_name = None
        text = None

        el = entry.tag("code")
        name = el.attr("displayName")
        code = el.attr("code")
        code_system = el.attr("codeSystem")
        code_system_name = el.attr("codeSystemName")

        text = core.strip_whitespace(entry.tag("text").val())

        el = entry.tag("effectiveTime")
        start_date = parse_date(el.tag("low").attr("value"))
        end_date = parse_date(el.tag("high").attr("value"))

        data.append(
            wrappers.ObjectWrapper(
                date_range=wrappers.ObjectWrapper(
                    start=start_date, end=end_date
                ),
                name=name,
                code=code,
                code_system=code_system,
                code_system_name=code_system_name,
                text=text,
            )
        )

    return wrappers.ListWrapper(data)
