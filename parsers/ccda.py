# -*- coding: utf-8 -*-
###############################################################################
# Module:      parsers
# Description: repo of parsing xml elements
# Authors:     Yage Wang
# Created:     2018.03.28
###############################################################################

from ._ccda.advance_directives import advance_directives
from ._ccda.allergies import allergies
from ._ccda.care_plan import care_plan
from ._ccda.demographics import demographics
from ._ccda.document import document
from ._ccda.encounters import encounters
from ._ccda.free_text import free_text
from ._ccda.functional_statuses import functional_statuses
from ._ccda.family_history import family_history
from ._ccda.immunizations import immunizations
from ._ccda.instructions import instructions
from ._ccda.medical_equipment import medical_equipment
from ._ccda.medications import medications
from ._ccda.problems import problems
from ._ccda.procedures import procedures
from ._ccda.results import results
from ._ccda.social_history import social_history
from ._ccda.vitals import vitals
from ..core import wrappers


def run(ccda):
    """
    Run parsing on certain section
    """
    data = wrappers.ObjectWrapper()

    data.document = document(ccda)
    data.advance_directives = advance_directives(ccda)
    data.allergies = allergies(ccda)
    data.care_plan = care_plan(ccda)
    data.chief_complaint = free_text(ccda, "chief_complaint")
    data.diagnosis = free_text(ccda, "diagnosis")
    data.demographics = demographics(ccda)
    data.encounters = encounters(ccda)
    data.functional_statuses = functional_statuses(ccda)
    data.family_history = family_history(ccda)
    data.history_of_present_illness = free_text(
        ccda, "history_of_present_illness"
    )
    data.immunizations = immunizations(ccda).administered
    data.immunization_declines = immunizations(ccda).declined
    data.instructions = instructions(ccda)
    data.results = results(ccda)
    data.medical_equipment = medical_equipment(ccda)
    data.medications = medications(ccda)
    data.problems = problems(ccda)
    data.procedures = procedures(ccda)
    data.physical_exam = free_text(ccda, "physical_exam")
    data.social_history = social_history(ccda)
    data.system_review = free_text(ccda, "system_review")
    data.vitals = vitals(ccda)

    return data
