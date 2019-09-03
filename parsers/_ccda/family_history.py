# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

"""
Parser for the CCDA family history section
"""

from ...core import wrappers
from ... import documents


def family_history(ccda):

    parse_date = documents.parse_date
    data = wrappers.ListWrapper()

    family_history = ccda.section("family_history")

    for entry in family_history.entries():

        el = entry.tag("subject")
        family_member = el.tag("code").attr("displayName")
        member_dob = parse_date(el.tag("birthTime").attr("value"))

        # conditions
        conditions = entry.els_by_tag("component")
        condition_data = wrappers.ListWrapper()

        for condition in conditions:
            onset_age = None
            concequence = None

            el = condition.tag("value")
            name = el.attr("displayName")
            code = el.attr("code")
            code_system = el.attr("codeSystem")
            for sub_entry in condition.els_by_tag("entryRelationship"):
                el = sub_entry.tag("code")
                if el.attr("displayName"):
                    if el.attr("displayName").lower() == "age":
                        onset_age = sub_entry.tag("value").attr("value")
                        continue
                else:
                    concequence = sub_entry.tag("value").attr("displayName")

            condition_data.append(
                wrappers.ObjectWrapper(
                    name=name,
                    code=code,
                    code_system=code_system,
                    onset_age=onset_age,
                    concequence=concequence,
                )
            )

        data.append(
            wrappers.ObjectWrapper(
                family_member=family_member,
                member_dob=member_dob,
                history=condition_data,
            )
        )

    return data
