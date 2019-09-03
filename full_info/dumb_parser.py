# -*- coding: utf-8 -*-
###############################################################################
# Module:      ehr
# Description: repo of EHR text extraction functions for robbins
# Authors:     Yage Wang
# Created:     03.28.2018
###############################################################################

import re
from bs4 import BeautifulSoup


def generate_text_from_CCD(
    parsed_data,
    sections=[],
    excludeCode=True,
    excludeNone=False,
    wholeTxt=False,
):
    """
    Extract all of the sections as text within an HL7 XML file
    
    @param parsed_data: dict of parsed data from Parser
    @param sections: predefined sections to seek, else all present
    @param excludeCode: if no, include the codes of many conditions after corresponding text. E.g. "drug allergy (SNOMED CT-416098002)"
    @param excludeNone: if yes, delete lines that have empty data
    @param wholeTxt: if yes, return single text report contains all info; if no, return dict of text of each section
    
    Return
    --------
    whole text report of the member
    OR
    section text dictionary of the member: {k:section, v:text of the section}

    Examples
    --------
    >>> from robbins.ehr import generate_text_from_CCD
    >>> generate_text_from_CCD(parsed_data, excludeCode=False, wholeTxt=True)
    '''demographics
        
       name: Adam Frankie Everyman
       dob: 11/25/1954
       gender: male
       ...
       allergies
        
       condition 1: drug allergy (SNOMED CT-416098002)
        	date range: 09/02/2009 - 01/03/2010
        	status: active
        	severity: moderate to severe
        	reaction: hives (247472004)
        ...
       system review

       text: None'''

    >>> generate_text_from_CCD(parsed_data, sections=['procedures'])
    {'procedures': '''condition 1: colonic polypectomy
         date: 02/15/2011
     ...
             phone: None
             device: None'''}
    """
    # initialize
    text_seperator = "\n\n" + "#" * 50 + "\n\n"
    if not isinstance(sections, list):
        sections = [sections]
    sections_dict = {}

    # section retrieval
    secs = [
        "demographics",
        "allergies",
        "medications",
        "problems",
        "procedures",
        "results",
        "encounters",
        "immunizations",
        "family_history",
        "history_of_present_illness",
        "medical_equipment",
        "care_plan",
        "physical_exam",
        "social_history",
        "vitals",
        "advance_directives",
        "functional_statuses",
        "system_review",
    ]
    if sections != []:
        secs = sections
    for sec in secs:
        if sec == "demographics":
            sections_dict[sec] = _demographic_section(parsed_data)
        else:
            sections_dict[sec] = _general_section(
                parsed_data, sec, excludeCode
            )

    if excludeNone:
        for section, text in sections_dict.items():
            clean_lines = []
            for line in text.splitlines():
                if line:
                    if (
                        ("None" not in line)
                        & ("none" not in line)
                        & (line[-1] != ":")
                    ):
                        clean_lines.append(line)
            sections_dict[section] = "\n".join(clean_lines)

    if wholeTxt:
        return text_seperator.join(
            [
                re.sub(r"_", " ", k) + "\n\n" + v
                for k, v in sections_dict.items()
            ]
        )
    else:
        return sections_dict


############################# PRIVATE FUNCTIONS ###############################


def _general_retriever(j, key):
    key = re.sub(r"_", " ", key)
    return ": ".join([key, str(j).lower()])


def _name_retriever(j):
    try:
        return (
            "name: "
            + " ".join([str(i) for i in j["given"]])
            + " "
            + str(j["family"])
        )
    except:
        return "name: None"


def _address_retriever(j):
    if j:
        return "address: " + " ".join(
            [
                " ".join(j["street"]),
                str(j["city"]),
                str(j["state"]),
                str(j["zip"]),
                str(j["country"]),
            ]
        )


def _phone_retriever(j):
    if isinstance(j, str):
        return "phone: " + j.replace("tel:", "")
    elif j is None:
        return "phone: None"
    else:
        return "phone: " + "; ".join(
            [
                str(i).replace("tel:", "")
                for i in list(j.values())
                if i is not None
            ]
        )


def _birthpalce_retriever(j):
    if j:
        return "birthpalce: " + " ".join(
            [str(j["state"]), str(j["zip"]), str(j["country"])]
        )


def _guardian_retriever(j):
    entries = ["guardian:"]
    entries.append("\t" + _name_retriever(str(j["name"])))
    entries.append("\trelationship: " + str(j["relationship"]).lower())
    entries.append("\t" + _address_retriever(j["address"]))
    entries.append("\t" + _phone_retriever(j["phone"]))
    return "\n".join(entries)


def _provider_retriever(j):
    entries = ["provider:"]
    entries.append("\torganization: " + str(j["organization"]))
    entries.append("\t" + _address_retriever(j["address"]))
    entries.append("\t" + _phone_retriever(j["phone"]))
    return "\n".join(entries)


def _date_range_retriever(j):
    return "date range: " + str(j["start"]) + " - " + str(j["end"])


def _condition_with_code_retriever(j, key, excludeCode):
    key = re.sub(r"_", " ", key)
    if excludeCode | (j["code"] is None):
        return key + ": " + str(j["name"]).lower()
    else:
        if "code_system_name" in j.keys():
            return (
                key
                + ": "
                + str(j["name"]).lower()
                + " ("
                + str(j["code_system_name"])
                + "-"
                + str(j["code"])
                + ")"
            )
        else:
            return (
                key
                + ": "
                + str(j["name"]).lower()
                + " ("
                + str(j["code"])
                + ")"
            )


def _product_retriever(j, key, excludeCode):
    key = re.sub(r"_", " ", key)
    entries = []
    entries.append(
        "\t" + _condition_with_code_retriever(j, "product", excludeCode)
    )
    del j["name"], j["code"], j["code_system"]
    if "code_system_name" in j.keys():
        del j["code_system_name"]
    for k in j.keys():
        if k == "value":
            entries.append("\t\tquantity: " + str(j[k]) + " " + str(j["unit"]))
        elif k == "unit":
            continue
        elif isinstance(j[k], dict):
            entries.append(
                "\t\t" + _condition_with_code_retriever(j[k], k, excludeCode)
            )
        else:
            entries.append("\t\t" + _general_retriever(j[k], k))
    return "\n".join(entries)


def _quantity_retriever(j, key):
    key = re.sub(r"_", " ", key)
    if isinstance(j, dict):
        if j["unit"] is None:
            return key + ": " + str(j["value"])
        else:
            return key + ": " + str(j["value"]) + " " + j["unit"]
    else:
        return str("quantity: " + str(j))


def _schedule_retriever(j, key):
    key = re.sub(r"_", " ", key)
    return (
        key
        + ": "
        + str(j["type"]).lower()
        + " as "
        + str(j["period_value"])
        + str(j["period_unit"])
    )


def _performer_retriever(j, key, level):
    key = re.sub(r"_", " ", key)
    entries = [key + ":"]
    entries.append(
        "\t" * (level + 1) + "organization: " + str(j["organization"])
    )
    entries.append("\t" * (level + 1) + _address_retriever(j))
    if "phone" in j.keys():
        entries.append("\t" * (level + 1) + _phone_retriever(j["phone"]))
    return "\n".join(entries)


def _sub_dict_retriever(j, key, level):
    key = re.sub(r"_", " ", key)
    entries = ["\t" * level + key + ":"]
    for k in j.keys():
        entries.append("\t" * (level + 1) + k + ": " + str(j[k]).lower())
    return "\n".join(entries)


def _tests_retriever(j, key, excludeCode):
    key = re.sub(r"_", " ", key)
    entries = []
    for i, chunk in enumerate(j):
        entries.append(
            _condition_with_code_retriever(
                chunk, "\t" + key + " " + str(i + 1), excludeCode
            )
        )
        del chunk["name"], chunk["code"], chunk["code_system"]
        if "code_system_name" in chunk.keys():
            del chunk["code_system_name"]

        for k in chunk.keys():
            if k == "value":
                entries.append(
                    "\t\tquantity: " + str(chunk[k]) + " " + str(chunk["unit"])
                )
            elif k == "unit":
                continue
            elif k == "reference_range":
                entries.append(_sub_dict_retriever(chunk[k], k, level=2))
            elif isinstance(chunk[k], dict):
                entries.append(
                    "\t\t"
                    + _condition_with_code_retriever(chunk[k], k, excludeCode)
                )
            else:
                entries.append("\t\t" + _general_retriever(chunk[k], k))
    return "\n".join(entries)


def _demographic_section(data):
    entries = []
    for key in data["demographics"].keys():
        j = data["demographics"][key]
        if key == "name":
            entries.append(_name_retriever(j))
        elif key == "address":
            entries.append(_address_retriever(j))
        elif key == "phone":
            entries.append(_phone_retriever(j))
        elif key == "birthplace":
            entries.append(_birthpalce_retriever(j))
        elif key == "guardian":
            entries.append(_guardian_retriever(j))
        elif key == "provider":
            entries.append(_provider_retriever(j))
        else:
            entries.append(_general_retriever(j, key))
    return "\n".join(entries)


def _general_section(data, name, excludeCode):
    entries = []
    if isinstance(data[name], dict):
        for key in data[name].keys():
            if isinstance(data[name][key], dict):
                entries.append(
                    _condition_with_code_retriever(
                        data[name][key], key, excludeCode
                    )
                )
            else:
                entries.append(_general_retriever(data[name][key], key))
        return "\n".join(entries)

    for i, chunk in enumerate(data[name]):
        if "name" in chunk.keys():
            entries.append(
                _condition_with_code_retriever(
                    chunk, "condition " + str(i + 1), excludeCode
                )
            )
            del chunk["name"], chunk["code"], chunk["code_system"]
            if "code_system_name" in chunk.keys():
                del chunk["code_system_name"]
        else:
            entries.append("condition " + str(i + 1))

        for key in chunk.keys():
            if key == "date_range":
                entries.append("\t" + _date_range_retriever(chunk[key]))
            elif (key == "performer") & (name == "procedures"):
                entries.append(
                    "\t" + _performer_retriever(chunk[key], key, level=1)
                )
            elif key == "location":
                entries.append(
                    "\t" + _performer_retriever(chunk[key], key, level=1)
                )
            elif (key == "tests") | (key == "results") | (key == "history"):
                entries.append(_tests_retriever(chunk[key], key, excludeCode))
            elif key == "product":
                entries.append(
                    _product_retriever(chunk[key], key, excludeCode)
                )
            elif "quantity" in key:
                entries.append("\t" + _quantity_retriever(chunk[key], key))
            elif key == "schedule":
                entries.append("\t" + _schedule_retriever(chunk[key], key))
            elif key == "prescriber":
                entries.append(_sub_dict_retriever(chunk[key], key, level=1))
            elif key == "findings":
                entries.append(
                    "\n".join(
                        [
                            "\t"
                            + _condition_with_code_retriever(
                                sub, "finding " + str(l + 1), excludeCode
                            )
                            for l, sub in enumerate(chunk[key])
                        ]
                    )
                )
            elif isinstance(chunk[key], dict):
                entries.append(
                    "\t"
                    + _condition_with_code_retriever(
                        chunk[key], key, excludeCode
                    )
                )
            else:
                entries.append("\t" + _general_retriever(chunk[key], key))
    return "\n".join(entries)


#
# def extract_sections(filepath,sections=None):
#     """
#     extract all of the sections as text within an EHR HL7 XML file
#     @param filepath: path to the xml
#     @param sections: predefined sections to seek, else all present
#     """
#     # null fields will be empty
#     stoplist = ['None recorded.', 'Unknown.', 'None recorded. (No additional sig information)', 'Code Code System Name Reaction Severity Onset NKDA',
#                'Reminders Provider Appointments None recorded.   Lab None recorded.   Referral None recorded.   Procedures None recorded.   Surgeries None recorded.   Imaging None recorded.',]

#     # build the dict
#     sections_dict = {}
#     soup = BeautifulSoup(open(filepath).read(), "lxml")
#     for node in soup.find_all("section"):
#         text = ""
#         title = node.find('title').text.lower()
#         body = node.find('text')
#         table = body.find('table')
#         if table is not None:
#             headers = []
#             for i in table.find_all('th') : headers.append(i.text)
#             if len(headers)!=0:
#                 rows = []
#                 for i in table.find('tbody').find_all('tr'):
#                     cells = []
#                     for j in i.find_all('td'): cells.append(j.text.strip())
#                     rows.append(cells)
#                 for row in rows:
#                     for i, cell in enumerate(row): text += headers[i] + ': ' + cell + '\n'
#             else: text += table.text.strip()
#         else:
#             text += body.text.strip()
#         if sections and title not in sections: continue
#         text = re.sub(r"[^\x00-\x7f]+|\n+",' ',text.strip())
#         text = "" if text in stoplist else text
#         sections_dict[title] = text
#     return sections_dict
