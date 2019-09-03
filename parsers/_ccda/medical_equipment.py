# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA medications section
"""

from ... import documents
from ...core import wrappers
from ... import core


def medical_equipment(ccda):

    parse_date = documents.parse_date
    data = wrappers.ListWrapper()

    medical_equipment = ccda.section("medical_equipment")

    for entry in medical_equipment.entries():
        el = entry.tag("effectiveTime")
        date = el.tag("high")
        if date.is_empty():
            date = el.tag("center")
        if date.is_empty():
            date = el.tag("low")
        date = parse_date(date.attr("value"))

        el = entry.tag("code")
        name = el.attr("displayName")
        code = el.attr("code")
        code_system = el.attr("codeSystem")
        code_system_name = el.attr("codeSystemName")

        quantity = entry.tag("quantity").attr("value")
        description = core.strip_whitespace(entry.tag("desc").val())

        data.append(
            wrappers.ObjectWrapper(
                name=name,
                code=code,
                code_system=code_system,
                code_system_name=code_system_name,
                date=date,
                quantity=quantity,
                description=description,
            )
        )

    return data
