# -*- coding: utf-8 -*-
###############################################################################
# Module:      core
# Description: repo of essential parsing tools
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################


def strip_whitespace(text):
    """ Remove leading and trailing whitespace from a string """
    if not isinstance(text, str):
        return text
    return (
        text.strip()
        .replace("<br>", " ")
        .replace("<br />", " ")
        .replace("<div>", " ")
        .replace("</div>", " ")
    )
