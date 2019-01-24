# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA plan of care section
"""

from sheldon import core
from sheldon.core import codes
from sheldon import documents
from ...core import wrappers


def care_plan(ccda):

    parse_date = documents.parse_date
    data = []
    care_plan = ccda.section('care_plan')

    for entry in care_plan.entries():

        name = None
        code = None
        code_system = None
        code_system_name = None

        # Plan of care encounters, which have no other details
        el = entry.tag('code')

        name = el.attr('displayName')
        code = el.attr('code')
        code_system = el.attr('codeSystem')
        code_system_name = el.attr('codeSystemName')

        date = parse_date(entry.tag('effectiveTime').attr('value'))
        text = core.strip_whitespace(entry.tag('text').val())

        data.append(
            wrappers.ObjectWrapper(
                text=text,
                name=name,
                code=code,
                code_system=code_system,
                code_system_name=code_system_name,
                date=date
        ))

    return wrappers.ListWrapper(data)
