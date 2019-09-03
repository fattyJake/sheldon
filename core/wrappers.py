# -*- coding: utf-8 -*-
###############################################################################
# Module:      core
# Description: repo of essential parsing tools
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

import datetime
import json
import re


class FixedOffset(datetime.tzinfo):
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset, name):
        self.__offset = datetime.timedelta(minutes=offset)
        self.__name = name

    @classmethod
    def UTC(cls):
        return cls(0, "UTC")

    @classmethod
    def from_string(cls, tz):
        stripped = str(tz).strip()

        if not stripped or "Z" == stripped:
            return cls.UTC()

        hour = int(stripped[1:3])
        minutes = hour * 60 + int(stripped[3:5])
        if stripped[0] == "-":
            minutes *= -1

        return cls(minutes, stripped)

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return datetime.timedelta(0)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime.datetime):
            try:
                utc = (o - o.utcoffset()).replace(tzinfo=FixedOffset.UTC())
            except:
                utc = o.replace(tzinfo=None)
            return utc.isoformat().replace("+00:00", "Z")
        elif isinstance(o, datetime.date):
            return o.strftime("%m/%d/%Y")
        elif isinstance(o, ObjectWrapper):
            return o.__dict__

        return json.JSONEncoder.default(self, o)


class ObjectWrapper(object):
    """
    Object Wrapper to append attributes and dump all data into JSON format.
    """

    def __init__(self, **kwargs):
        for keyword, value in kwargs.items():
            setattr(self, keyword, value)

    def __setattr__(self, key, value):
        val = value
        if callable(value):
            method = value.__get__(self, self.__class__)
            val = method
        object.__setattr__(self, key, val)

    def json(self):
        return json.dumps(self, cls=JSONEncoder)


class ListWrapper(list):
    """
    List Wrapper all ObjectWapper into JSON format.
    """

    def json(self):
        return json.dumps(self, cls=JSONEncoder)


def parse_number(s):
    """
    Somewhat mimics JavaScript's parseFloat() functionality
    """
    if not s:
        return None
    s = re.sub("[^0-9]", "", s)
    if s:
        value = float(s)
        return int(value) if value == int(value) else value
    else:
        return None
