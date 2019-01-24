# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA social history section
"""

from ...core import wrappers
from ... import core
from ... import documents


def social_history(ccda):

    parse_date = documents.parse_date
    data = wrappers.ListWrapper()

    # We can parse all of the social_history sections
    # but in practice, this section seems to be used for
    # smoking status, so we're just going to break that out.
    # And we're just looking for the first non-empty one.
    social_history = ccda.section('social_history')
    entries = social_history.entries()
    for entry in entries:

        name = None
        code = None
        code_system = None
        code_system_name = None
        value = None
        start_date = None
        end_date = None
        
        social_history_ = entry.template('2.16.840.1.113883.10.20.22.4.78')
        if social_history_.is_empty(): social_history_ = entry.template('2.16.840.1.113883.10.22.4.78')
        if social_history_.is_empty(): social_history_ = entry.template('2.16.840.1.113883.10.20.22.4.38')

        if social_history_.is_empty():
            continue
        
        effective_times = entry.els_by_tag('effectiveTime')

        # the first effectiveTime is the med start date
        try:
            el = effective_times[0]
        except IndexError:
            el = None

        if el:
            start_date = parse_date(el.tag('low').attr('value'))
            if el.tag('high'): end_date = parse_date(el.tag('high').attr('value'))

        el = social_history_.tag('code')
        name = el.attr('displayName')
        code = el.attr('code')
        code_system = el.attr('codeSystem')
        code_system_name = el.attr('codeSystemName')
        
        el = social_history_.tag('value')
        value = core.strip_whitespace(el.val())

        data.append(wrappers.ObjectWrapper(
            date_range=wrappers.ObjectWrapper(
                start=start_date,
                end=end_date
            ),
            name=name,
            code=code,
            code_system=code_system,
            code_system_name=code_system_name,
            value=value
        ))

    return data
