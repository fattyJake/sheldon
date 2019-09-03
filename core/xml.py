# -*- coding: utf-8 -*-
###############################################################################
# Module:      core
# Description: repo of essential parsing tools
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

from __future__ import absolute_import
import logging
from xml.etree import ElementTree as etree

from . import wrappers
from . import _core as core


logging.getLogger(__name__).addHandler(logging.NullHandler())


def parse(data):
    if not data or not isinstance(data, str):
        logging.info("BB Error: XML data is not a string")
        return None

    try:
        root = etree.fromstring(data)
    except:
        logging.info("BB Error: Could not parse XML")
        return None

    return _Element(root, root)


class _Element(object):
    """
    An object of higher level wrapper of xml ElementTree for deeper usage

    @param element: the text of CCD xml segment
    @param root: the full text of CCD xml in case to retrieve parent tags
    """

    def __init__(self, element, root):
        self._element = element
        self._root = root

    def __setattr__(self, key, value):
        val = value.__get__(self, self.__class__) if callable(value) else value
        object.__setattr__(self, key, val)

    def attr(self, attribute_name):
        """
        Get readable attribute value in a tag

        @param attribute_name: target attribute (e.g. 'value' in <effectiveTime value="20170509093009-0400"/>)
        """
        if self._element is None:
            return None

        name = attribute_name.replace(
            "xsi:", "{http://www.w3.org/2001/XMLSchema-instance}"
        )

        attr_val = self._element.get(name)
        if attr_val:
            return _unescape_special_chars(attr_val)
        return None

    def bool_attr(self, attribute_name):
        """
        Get boolean format attribute value in a tag

        @param attribute_name: target attribute (e.g. 'value' in <effectiveTime value="20170509093009-0400"/>)
        """
        raw_attr = self.attr(attribute_name)
        return raw_attr == "true" or raw_attr == "1"

    def content(self, content_id):
        """
        Search for a content tag by "ID", and return it as an element.
        These are used in the unstructured versions of each section but
        referenced from the structured version sometimes.
        E.g. <content ID="UniqueNameReferencedElsewhere"/> can be found using:
        el = dom.content('UniqueNameReferencedElsewhere');
        
        We can't use `getElementById` because `ID` (the standard attribute name
        in this context) is not the same attribute as `id` in XML, so there are
        no matches.

        @param content_id: value of target ID
        """
        el = _tag_attr_val(self._element, "content", "ID", content_id)
        if el is None:
            # check the <td> tag too, which isn't really correct but
            # will inevitably be used sometimes because it looks like very
            # normal HTML to put the data directly in a <td>
            el = _tag_attr_val(self._element, "td", "ID", content_id)

        if el is None:
            # Ugh, Epic uses really non-standard locations.
            el = _tag_attr_val(self._element, "caption", "ID", content_id)
            if el is None:
                el = _tag_attr_val(
                    self._element, "paragraph", "ID", content_id
                )
            if el is None:
                el = _tag_attr_val(self._element, "tr", "ID", content_id)
            if el is None:
                el = _tag_attr_val(self._element, "item", "ID", content_id)

        if el is None:
            return _Element.empty()
        else:
            return self._wrap_element(el)

    def els_by_tag(self, tag):
        """
        Iterrate all targeted child elements

        @param tag: target tag to find in all chile elements
        """
        els = self._element.findall(
            ".//{ns}{name}".format(name=tag, ns="{urn:hl7-org:v3}")
        )
        return self._wrap_element(els)

    @classmethod
    def empty(cls):
        """
        Create an empty element
        """
        return cls(etree.Element("empty"), root=None)

    def is_empty(self):
        """
        Decide whether the element is None
        """
        return self._element.tag.lower() == "empty"

    def tag(self, name):
        """
        Find the first targeted child elements

        @param name: name of the tag
        """
        el = self._element.find(
            ".//{ns}{name}".format(name=name, ns="{urn:hl7-org:v3}")
        )
        if el is None:
            return _Element.empty()
        else:
            return self._wrap_element(el)

    def loinc_section(self, code):
        """
        Search for a section LOINC code, and return its PARENT element (section). Usually used for locating sections

        Examples
        --------
        <ns0:code code="10157-6" codeSystem="2.16.840.1.113883.6.1" codeSystemName="LOINC" displayName="Family History" />
        The section can be found using:
        >>> el = doc.loinc_section('10157-6');
        """
        el = _tag_attr_val(self._element, "code", "code", code)
        # WARNING: DO NOT use "if not el:"
        # http://effbot.org/zone/element.htm#truth-testing
        if el is None:
            return _Element.empty()
        else:
            if not hasattr(el, "parent"):
                parent_map = {c: p for p in self._element.iter() for c in p}
                el_parent = parent_map[el]
            return self._wrap_element(el_parent)

    def template(self, template_id):
        """
        Search for a template ID, and return its PARENT element.

        Examples
        --------
        <templateId root="2.16.840.1.113883.10.20.22.2.17"/>
        Can be found using:
        >>> el = doc.template('2.16.840.1.113883.10.20.22.2.17')
        """
        el = _tag_attr_val(self._element, "templateId", "root", template_id)
        # WARNING: DO NOT use "if not el:"
        # http://effbot.org/zone/element.htm#truth-testing
        if el is None:
            return _Element.empty()
        else:
            if not hasattr(el, "parent"):
                parent_map = {c: p for p in self._element.iter() for c in p}
                el_parent = parent_map[el]
            return self._wrap_element(el_parent)

    def val(self):
        """
        Retrieve the element's value.
        This function also knows how to retrieve the value of <reference> tags,
        which can store their content in a <content> tag in a totally different
        part of the document.

        Examples
        --------
        If the element is:
           <city>Madison</city>
        Use:
        >>> value = el.tag('city').val()
        """
        if self._element is None:
            return None

        if self.is_empty():
            return None

        text_context = _text_content(self._element)

        # if there's no text value here and the only thing inside is a
        # <reference> tag, see if there's a linked <content> tag we can
        # get something out of
        if not core.strip_whitespace(text_context):

            content_id = None
            # "no text value" might mean there's just a reference tag
            if len(self._element) == 1 and self._element[0].tag.endswith(
                "reference"
            ):
                content_id = self._element[0].get("value")

            # or maybe a newlines on top/above the reference tag
            elif len(self._element) == 3 and self._element[1].tag.endswith(
                "reference"
            ):
                content_id = self._element[1].get("value")
            else:
                return _unescape_special_chars(text_context)

            if content_id and content_id[0] == "#":
                content_id = content_id[1:]
                doc_root = self._get_root()
                content_tag = doc_root.content(content_id)
                return content_tag.val()

        return _unescape_special_chars(text_context)

    @classmethod
    def wrap_root(cls, root):
        return cls(root, root)

    def _get_root(self):
        return self._wrap_element(self._root)

    def _wrap_element(self, element):
        if issubclass(type(element), list):
            return wrappers.ListWrapper(
                [_Element(e, self._root) for e in element]
            )
        else:
            return _Element(element, self._root)


############################# PRIVATE FUNCTIONS ###############################


def _tag_attr_val(element, tag, attribute, value):
    namespace = "{urn:hl7-org:v3}"
    for el in element.iter(namespace + tag):
        if el.get(attribute) == value:
            return el


def _text_content(element):
    # emulates DOM's Node.textContent property
    if element is None:
        return ""

    children_text = [_text_content(child) for child in element]
    # removes NoneType and empty entries
    children_text = [c for c in children_text if c]

    portions = []
    if element.text is not None:
        portions.append(element.text)

    if children_text:
        portions += children_text

    if element.tail is not None:
        portions.append(core.strip_whitespace(element.tail))

    if not portions:
        return None

    return " ".join([p for p in portions if p])


def _unescape_special_chars(s):
    if not s:
        return s
    return (
        s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )
