# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA procedures section
"""

from ...core import wrappers
from ... import core
from ... import documents


def procedures(ccda):

    parse_date = documents.parse_date
    parse_address = documents.parse_address
    data = wrappers.ListWrapper()

    procedures = ccda.section("procedures")

    for entry in procedures.entries():

        el = entry.tag("effectiveTime")
        date = parse_date(el.attr("value"))

        el = entry.tag("code")
        name = el.attr("displayName")
        code = el.attr("code")
        code_system = el.attr("codeSystem")

        if not name:
            name = core.strip_whitespace(entry.tag("originalText").val())

        # 'specimen' tag not always present
        el = (
            entry.tag("specimen")
            .tag("specimenRole")
            .tag("specimenPlayingEntity")
            .tag("code")
        )
        specimen_name = el.attr("displayName")
        specimen_code = el.attr("code")
        specimen_code_system = el.attr("codeSystem")

        el = entry.tag("performer").tag("addr")
        organization = el.tag("name").val()
        phone = el.tag("telecom").attr("value")

        performer_dict = parse_address(el)
        performer_dict.organization = organization
        performer_dict.phone = phone

        # participant => device
        el = entry.template("2.16.840.1.113883.10.20.22.4.37").tag("code")
        device_name = el.attr("displayName")
        device_code = el.attr("code")
        device_code_system = el.attr("codeSystem")

        data.append(
            wrappers.ObjectWrapper(
                date=date,
                name=name,
                code=code,
                code_system=code_system,
                specimen=wrappers.ObjectWrapper(
                    name=specimen_name,
                    code=specimen_code,
                    code_system=specimen_code_system,
                ),
                performer=performer_dict,
                device=wrappers.ObjectWrapper(
                    name=device_name,
                    code=device_code,
                    code_system=device_code_system,
                ),
            )
        )

    return data
