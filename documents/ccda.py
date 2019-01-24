# -*- coding: utf-8 -*-
###############################################################################
# Module:      documents
# Description: repo of segmenting xml source
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

from .. import documents


def process(ccda):
    """
    Preprocesses the CCDA docuemnt
    """
    ccda.section = section
    return ccda


def section(ccda, name):
    """
    Finds the section of a CCDA document
    """

    entries = documents.entries

    if 'document' == name:
        return ccda.loinc_section('34133-9')
    if 'allergies' == name:
        el = ccda.loinc_section('48765-2')
        el.entries = entries
        return el
    if 'care_plan' == name:
        el = ccda.loinc_section('18776-5')
        el.entries = entries
        return el
    if 'chief_complaint' == name:
        el = ccda.loinc_section('46239-0')
        # no entries in Chief Complaint
        return el
    if 'demographics' == name:
        return ccda.loinc_section('34133-9')
    if 'diagnosis' == name:
        return ccda.loinc_section('51848-0')
    if 'encounters' == name:
        el = ccda.loinc_section('46240-8')
        el.entries = entries
        return el
    if 'functional_statuses' == name:
        el = ccda.loinc_section('47420-5')
        el.entries = entries
        return el
    if 'family_history' == name:
        el = ccda.loinc_section('10157-6')
        el.entries = entries
        return el
    if 'history_of_present_illness' == name:
        el = ccda.loinc_section('11348-0')
        return el
    if 'immunizations' == name:
        el = ccda.loinc_section('11369-6')
        el.entries = entries
        return el
    if 'instructions' == name:
        el = ccda.loinc_section('69730-0')
        el.entries = entries
        return el
    if 'results' == name:
        el = ccda.loinc_section('30954-2')
        el.entries = entries
        return el
    if 'medical_equipment' == name:
        el = ccda.loinc_section('46264-8')
        el.entries = entries
        return el
    if 'medications' == name:
        el = ccda.loinc_section('10160-0')
        el.entries = entries
        return el
    if 'problems' == name:
        el = ccda.loinc_section('11450-4')
        el.entries = entries
        return el
    if 'procedures' == name:
        el = ccda.loinc_section('47519-4')
        el.entries = entries
        return el
    if 'physical_exam' == name:
        el = ccda.loinc_section('29545-1')
        return el
    if 'social_history' == name:
        el = ccda.loinc_section('29762-2')
        el.entries = entries
        return el
    if 'system_review' == name:
        el = ccda.loinc_section('10187-3')
        return el
    if 'vitals' == name:
        el = ccda.loinc_section('8716-3')
        el.entries = entries
        return el
    if 'advance_directives' == name:
        el = ccda.loinc_section('42348-3')
        el.entries = entries
        return el

    return None

#    if 'document' == name:
#        return ccda.template('2.16.840.1.113883.10.20.22.1.1')
#    if 'allergies' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.6.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.2')
#        el.entries = entries
#        return el
#    if 'care_plan' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.10')
#        el.entries = entries
#        return el
#    if 'chief_complaint' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.13')
#        if el.is_empty():
#            el = ccda.template('1.3.6.1.4.1.19376.1.5.3.1.1.13.2.1')
#        # no entries in Chief Complaint
#        return el
#    if 'demographics' == name:
#        return ccda.template('2.16.840.1.113883.10.20.22.1.1')
#    if 'encounters' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.22')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.3.88.11.83.127')
#        if el.is_empty(): el = ccda.template('1.3.6.1.4.1.19376.1.5.3.1.1.5.3')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.3')
#        el.entries = entries
#        return el
#    if 'functional_statuses' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.14')
#        el.entries = entries
#        return el
#    if 'family_history' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.15')
#        el.entries = entries
#        return el
#    if 'history_of_present_illness' == name:
#        el = ccda.template('1.3.6.1.4.1.19376.1.5.3.1.3.4')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.4.9')
#        return el
#    if 'immunizations' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.2.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.2')
#        el.entries = entries
#        return el
#    if 'instructions' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.45')
#        el.entries = entries
#        return el
#    if 'results' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.3.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.3')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.14')
#        el.entries = entries
#        return el
#    if 'medical_equipment' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.23')
#        el.entries = entries
#        return el
#    if 'medications' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.1.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.8')
#        el.entries = entries
#        return el
#    if 'problems' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.5.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.5')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.11')
#        el.entries = entries
#        return el
#    if 'procedures' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.7.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.7')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.12')
#        el.entries = entries
#        return el
#    if 'physical_exam' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.2.10')
#        return el
#    if 'social_history' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.17')
#        el.entries = entries
#        return el
#    if 'system_review' == name:
#        el = ccda. template('1.3.6.1.4.1.19376.1.5.3.1.3.18')
#        return el
#    if 'vitals' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.4.1')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.22.2.4')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.16')
#        el.entries = entries
#        return el
#    if 'advance_directives' == name:
#        el = ccda.template('2.16.840.1.113883.10.20.22.2.21')
#        if el.is_empty(): el = ccda.template('2.16.840.1.113883.10.20.1.1')
#        el.entries = entries
#        return el