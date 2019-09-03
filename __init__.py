# -*- coding: utf-8 -*-
###############################################################################
#             oooo                  oooo        .o8                           #
#             `888                  `888       "888                           #
#     .oooo.o  888 .oo.    .ooooo.   888   .oooo888   .ooooo.  ooo. .oo.      #
#    d88(  "8  888P"Y88b  d88' `88b  888  d88' `888  d88' `88b `888P"Y88b     #
#    `"Y88b.   888   888  888ooo888  888  888   888  888   888  888   888     #
#    o.  )88b  888   888  888    .o  888  888   888  888   888  888   888     #
#    8""888P' o888o o888o `Y8bod8P' o888o `Y8bod88P" `Y8bod8P' o888o o888o    #
#                                                                             #
###############################################################################
# sheldon is a python package designed to parse HL7 FHIR standard CCD xml files
# in to categorized, ordered format output for easy data retreival while not
# requiring people fully understand HL7 structures.
#
# Authors:  Yage Wang (credited to https://github.com/ctsit/bluebutton.py)
# Created:  2018.03.28
# Version:  1.0.0
###############################################################################

from . import core
from . import full_info
from .documents import ccda as docCCDA
from .parsers import ccda as parseCCDA
from .vectorizer import Vectorizer

import io
import re
import os
import glob
import json
import pickle
import collections
import random
from datetime import datetime
from tqdm import tqdm
import zipfile

import pandas as pd


class Parser(object):
    """
    Object to parse CCDA xml source. The whole parser is based on two custome object: "ObjectWrapper"
    and "ListWrapper". Such structure enable developer to freely appedn attributes to object ontology
    and easily convert any-level object into JSON format.

    Parameters
    --------
    source: string
        The full text of CCD xml file
    
    Attributes
    --------
    source: an object of higher level wrapper of xml etree for deeper usage.
        Please refer to help(sheldon.core.xml._Element)

    data: an ObjectWrapper of all parsed data
        Direct sub-attributes can be all major sections in xml, including demographics,
        allergies, medications, diagnosis, problems, procedures, results, encounters,
        immunizations, family_history, history_of_present_illness, medical_equipment, care_plan,
        physical_exam, social_history, vitals, advance_directives, functional_statuses, system_review.
        All sections can be further retreived by finding sub-attributes; if a section is a ListWrapper,
        it can be subscriptable. Besides all sections, data attribute also has one sub-method json() to
        convert all data into JSON format; json() method can be used on any level of ObjectWapper.

    Examples
    --------
    >>> from sheldon import Parser
    >>> with open('CCD.sample.xml') as f: ccd = Parser(f.read())
    >>> ccd.data.json()
    '{"document": {"date": "2005-03-29T12:15:04Z", "title": 
    ...
    "value": 145, "unit": "mm[Hg]"}]}]}'

    >>> ccd.data.allergies
    [<sheldon.core.wrappers.ObjectWrapper at 0x9b57b70>,
     <sheldon.core.wrappers.ObjectWrapper at 0x9b57780>,
     <sheldon.core.wrappers.ObjectWrapper at 0x9b572b0>]
    
    >>> ccd.data.allergies[0].name
    'drug allergy'

    >>> ccd.data.allergies[0].json()
    '{"date_range": {"start": "09/02/2009",
    ...
    "code_system_name": "RxNorm"}}'
    """

    def __init__(self, source):
        parsed_document, parsed_data = None, None

        # remove xml ASCII codes to prevent ElemenetTree parsing error
        source = re.sub(r"\\?&?#?x[0-9a-fA-F]{2}", " ", source)
        parsed_data = core.parse_data(source)

        parsed_data = docCCDA.process(parsed_data)
        parsed_document = parseCCDA.run(parsed_data)

        self.data = parsed_document
        self.source = parsed_data


def member_json(member="KC87853B", client="HEALTHFIRST", sections=[]):
    """
    Concat all EHR pieces of one member (in case of data storage method in MRRBWFS1)
    
    Parameters
    --------
    member: string
        member client ID
    client: string
        the table name of client DB
    sections: list of strings
        predefined sections to seek, else all present
    
    Return
    --------
    member CCD json

    Examples
    --------
    >>> from sheldon import member_json
    >>> member_json(member='KC87853B', client='HEALTHFIRST')
    {
    'meta':
        {
        'name_first': 'RENEE',
        'name_middle': '',
        'name_last': 'CAMPBELL',
     ...
             'code_system_name': 'LOINC'
             },
             'comment': None
        }],
    'timestamp': datetime.datetime(2018, 1, 9, 6, 7, 39)
    }
    """

    # initialize
    output = {}
    meta = full_info.fetch_db.info(member, client)
    output["meta"] = meta

    if not meta:
        return ""
    filepaths = meta["ccds"]
    if filepaths is None:
        return ""

    for filepath in sorted(filepaths)[::-1]:
        try:
            zf = zipfile.ZipFile(filepath, "r")
        except FileNotFoundError:
            continue

        file_list = [
            f.filename
            for f in zf.filelist
            if ("Content/" in f.filename)
            & ("FileMetadata" not in f.filename)
            & (f.filename.endswith("xml"))
        ]

        # parse into text
        for file in file_list:
            xmlContent = io.TextIOWrapper(
                zf.open(file, "r"), encoding="utf-8"
            ).read()
            if not re.sub(r"\s", "", xmlContent):
                continue
            ccd = Parser(xmlContent)
            parsed_data = json.loads(ccd.data.json())
            ccd = {k: v for k, v in parsed_data.items() if v}
            _update_rec(output, ccd)

        #        for k, v in output.items():
        #            if isinstance(v, list): output[k] = [dict(y) for y in set(tuple(x.items()) for x in v)]
        time = re.search(r"[0-9]{14}", filepath)
        if time:
            output["timestamp"] = datetime.strptime(
                time.group(), "%Y%m%d%H%M%S"
            )
        else:
            output["timestamp"] = None
        break

    if sections:
        output = {k: v for k, v in output.items() if k in sections}
    return output


def file_processor(filepaths):
    """
    Parse all EHR under file paths and dump as JSON
    
    Parameters
    --------
    filepaths: string or list of string
        one or more file paths

    Examples
    --------
    >>> from sheldon import file_processor
    >>> file_processor(r"X:\")
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for filepath in filepaths:
        output = {}

        if filepath.endswith(".xml") or filepath.endswith(".XML"):
            xmlContent = open(filepath).read()
            ccd = Parser(xmlContent)
            output = json.loads(ccd.data.json())
        if filepath.endswith(".zip") or filepath.endswith(".ZIP"):
            try:
                zf = zipfile.ZipFile(filepath, "r")
            except FileNotFoundError:
                continue

            file_list = [
                f.filename
                for f in zf.filelist
                if ("Content/" in f.filename)
                & ("FileMetadata" not in f.filename)
                & (f.filename.endswith("xml"))
            ]

            # parse into text
            for file in file_list:
                xmlContent = io.TextIOWrapper(
                    zf.open(file, "r"), encoding="utf-8"
                ).read()
                ccd = Parser(xmlContent)
                parsed_data = json.loads(ccd.data.json())
                ccd = {k: v for k, v in parsed_data.items() if v}
                _update_rec(output, ccd)
        _, ext = os.path.split(filepath)
        if ext not in ["zip", "xml", "ZIP", "XML"]:
            file_list = []

            def globit(srchDir):
                srchDir = os.path.join(srchDir, "*")
                for file in glob.glob(srchDir):
                    if (
                        ("Content\\" in file)
                        and ("FileMetadata" not in file)
                        and (file.endswith("xml"))
                    ):
                        file_list.append(file)
                    globit(file)

            globit(filepath)

            # parse into text
            for file in file_list:
                # short_unc = win32api.GetShortPathName(file)
                xmlContent = open(file, encoding="utf-8").read()
                ccd = Parser(xmlContent)
                parsed_data = json.loads(ccd.data.json())
                ccd = {k: v for k, v in parsed_data.items() if v}
                _update_rec(output, ccd)

        if output:
            parent_path = os.path.dirname(os.path.realpath(filepath))
            filename = (
                os.path.split(filepath)[1]
                .replace(".zip", "")
                .replace(".xml", "")
                .replace(".ZIP", "")
                .replace(".XML", "")
            )
            with open(os.path.join(parent_path, filename + ".json"), "w") as f:
                f.write(json.dumps(output, default=_datetime_serializer))


def condition_labeler(
    member="KC87853B", client="HEALTHFIRST", latest_encounter_date=None
):
    assert latest_encounter_date is not None, print(
        "AttributeError: must provide latest_encounter_date as standardized integer type."
    )
    # ICD10 codes related to HCC85-88
    client_mappings = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"client_mappings",
            ),
            "rb",
        )
    )
    condition_mappings = list(
        pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    r"codes_ICD_mappings",
                ),
                "rb",
            )
        )
    )

    client, db = client_mappings[client]
    claims = full_info.fetch_db.claims(member, client, db, with_time=True)
    if not claims:
        return
    after_30_claims = {
        k: v
        for k, v in claims.items()
        if k > latest_encounter_date and k - latest_encounter_date <= 1440
    }
    for codes in after_30_claims.values():
        if list(set(codes) & set(condition_mappings)):
            return 1
    return 0


def readmission(member="KC87853B", client="HEALTHFIRST"):

    readmission = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"readmission",
            ),
            "rb",
        )
    )
    client_mappings = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"client_mappings",
            ),
            "rb",
        )
    )
    client, db = client_mappings[client]

    claims = full_info.fetch_db.readmission_claims(member, client, db)
    if not claims:
        return 0
    df = pd.DataFrame(claims)
    df.columns = ["ENC_ID", "ADM_DATE", "DSC_DATE", "CODE"]
    df = (
        df.groupby(["ENC_ID", "ADM_DATE", "DSC_DATE"])["CODE"]
        .agg(list)
        .reset_index()
    )
    df["main_1"] = df["CODE"].map(
        lambda x: True
        if list(set(x) & set(readmission["inpatient_stays"]))
        else False
    )
    df["main_2"] = df["CODE"].map(
        lambda x: False
        if list(set(x) & set(readmission["nonacute_inpatient_stays"]))
        else True
    )
    df = df.loc[
        df["main_1"] & df["main_2"], ["ENC_ID", "ADM_DATE", "DSC_DATE", "CODE"]
    ]
    if df.shape[0] == 0:
        return (0, None)

    df["exc_1"] = df["CODE"].map(
        lambda x: False
        if list(
            set(x)
            & set(
                readmission["pregnancy"]
                + readmission["perinatal"]
                + readmission["chemotherapy"]
                + readmission["rehabilitation"]
                + readmission["transplant"]
            )
        )
        else True
    )
    df["exc_2"] = df["CODE"].map(
        lambda x: False
        if list(set(x) & set(readmission["planned_procedure"]))
        else True
    )
    df["exc_3"] = df["CODE"].map(
        lambda x: True
        if list(set(x) & set(readmission["acute_condition"]))
        else False
    )
    df = df.loc[
        df["exc_1"] & (df["exc_2"] | df["exc_3"]),
        ["ENC_ID", "ADM_DATE", "DSC_DATE", "CODE"],
    ]
    if df.shape[0] == 0 or df.shape[0] == 1:
        return (0, None)

    for idx, val in df.iterrows():
        if pd.isnull(val["DSC_DATE"]):
            df.at[idx, "DSC_DATE"] = val["ADM_DATE"]
        if val["DSC_DATE"] == datetime(8888, 12, 31, 0, 0):
            df.at[idx, "DSC_DATE"] = val["ADM_DATE"]
    df["delta"] = (
        df["ADM_DATE"].shift(-1) - df["DSC_DATE"].astype("datetime64")
    ).map(lambda x: x.days)
    df = df.loc[(df["delta"] > 0) & (df["delta"] <= 30), :]
    if df.shape[0] == 0:
        return (0, None)
    else:
        return (1, _datetime_std(df.at[df.index[-1], "ADM_DATE"]))


def member_text(
    member="KC87853B",
    client="HEALTHFIRST",
    sections=[],
    excludeCode=False,
    excludeNone=True,
):
    """
    Concat all EHR pieces of one member and convert to natural language section reports (in case of data storage method in MRRBWFS1)
    
    @param member ID (client)
    @param client: the name of the client table
    @param sections: predefined sections to seek, else all present
    @param excludeCode: if no, include the codes of many conditions after corresponding text. E.g. "drug allergy (SNOMED CT-416098002)"
    @param excludeNone: if yes, delete lines that have empty data
    
    Return
    --------
    member CCD text

    Examples
    --------
    >>> from sheldon import member_text
    >>> member_text(member='KC87853B', client='HEALTHFIRST')
    {'allergies': '''condition 1: unknown (unknown-ASSERTION)
        status: active
    ...
        results 9: oxygen saturation (LOINC-59408-5)
            quantity: 100 %'''}
    """

    # initialize
    text_seperator = "\n\n" + "#" * 50 + "\n\n"
    filepaths = full_info.fetch_db.info(member, client.upper())

    if filepaths is None:
        return ""
    filepaths = filepaths["ccds"]
    if filepaths is None:
        return ""

    output = {}
    for filepath in filepaths:
        try:
            zf = zipfile.ZipFile(filepath, "r")
        except FileNotFoundError:
            continue
        file_list = [
            f.filename
            for f in zf.filelist
            if ("Content/" in f.filename)
            & ("FileMetadata" not in f.filename)
            & (f.filename.endswith("xml"))
        ]

        # parse into text
        for file in file_list:
            xmlContent = io.TextIOWrapper(
                zf.open(file, "r"), encoding="utf-8"
            ).read()
            ccd = Parser(xmlContent)
            parsed_data = json.loads(ccd.data.json())
            ccd = full_info.dumb_parser.generate_text_from_CCD(
                parsed_data, excludeCode=False, excludeNone=True
            )
            ccd = {k: v for k, v in ccd.items() if v}
            output.update(ccd)
    if sections != []:
        output = {k: v for k, v in output.items() if k in sections}
    return text_seperator.join(
        [re.sub(r"_", " ", k) + "\n\n" + v for k, v in output.items()]
    )


############################# PRIVATE FUNCTIONS ###############################


def _update_rec(d, u):
    for k, v in u.items():
        if not v:
            continue
        elif isinstance(v, collections.Mapping):
            d[k] = _update_rec(d.get(k, {}), v)
        elif isinstance(v, list):
            new_list = d.get(k, []) + v
            new_list = [json.dumps(el) for el in new_list]
            d[k] = [json.loads(el) for el in list(set(new_list))]
        else:
            d[k] = v
    return d


def _datetime_serializer(o):
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.__str__()


def _datetime_std(d):
    base = datetime.strptime("01/01/1900", "%m/%d/%Y")
    std_dt = d - base
    std_dt = int(std_dt.total_seconds() / 3600)
    return std_dt
