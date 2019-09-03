# -*- coding: utf-8 -*-
###############################################################################
# Module:      vectorizers
# Description: repo of vectorize perticular sections
# Authors:     Yage Wang
# Created:     2018.05.18
###############################################################################

import os
import re
import pickle
from datetime import datetime
import numpy as np
from text_tools import preprocessing


class Vectorizer(object):
    """
    Aim to vectorize EHR CCD data into event contaiers for further Deep Learning use.
    """

    def __init__(self):
        """
        Initialize a vectorizer to repeat use; load section variable spaces
        """
        self.code_system = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "code_system",
                ),
                "rb",
            )
        )

        self.all_variables = list(
            pickle.load(
                open(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        r"pickle_files",
                        "all_variables_new",
                    ),
                    "rb",
                )
            )
        )
        self.punctuation = """!"#$%&'*,.:;/<=>?@[\]^_`{|}~•"""
        self.stopwords = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "stopwords",
                ),
                "rb",
            )
        )
        self.bigram = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "bigram",
                ),
                "rb",
            )
        )
        self.vocab = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "vocab",
                ),
                "rb",
            )
        )

        self.lab_norm = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "lab_norm",
                ),
                "rb",
            )
        )
        self.vital_norm = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "vital_norm",
                ),
                "rb",
            )
        )
        self.time_norm = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    "time_norm",
                ),
                "rb",
            )
        )

        self.variable_size = len(self.all_variables) + len(self.vocab.dfs)

    def fit_transform(self, ehr, max_sequence_length, encounter_limit=None):
        """
        Transform EHR JSON result from sheldon.Parser into event containers

        Parameters
        --------
        ehr: JSON (dict) type object
            The parsed data from sheldon.Parser
        
        max_sequence_length: int
            Fixed padding latest number of time buckets
        
        max_token_length: int
            Fixed padding number within one time bucket of one section

        Return
        --------
        T: numpy array, shape (num_timestamp,)
            All standardized time bucket numbers

        X: numpy array, shape (num_timestamp,)
            The index of each event based on each section variable space

        Q: numpy array, shape (num_timestamp,)
            The quantity or time range of corresponding variable in X
        
        latest_ect_time: int
            The standardized integer type of the date of the latest encounter time

        Examples
        --------
        >>> from sheldon import Vectorizer
        >>> vec = Vectorizer()
        >>> vec.fit_transform(ehr, 200)[0]
        array([84954, 85460, 85560, 85582, 85584, 85740, 85741, 85834, 85835,
               85880, 85884, 85926, 85950, 85951, 85962, 85968, 86132])
        
        >>> vec.fit_transform(ehr, 200)[1]
        array([[[  138,  1146,  1457, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               ...,
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [    0,     3,     5, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [   24,   151,   169, ...,  8579,  8579,  8579]]])
    
        >>> vec.fit_transform(ehr, 200)[2]
        array([[[ 1.62456340e-04,  1.62456340e-04,  1.62456340e-04, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])
        """
        events = []
        events += self._medication_vectorizer(ehr)
        events += self._allergen_vectorizer(ehr)
        events += self._immuzation_vectorizer(ehr)
        events += self._lab_vectorizer(ehr)
        events += self._encounter_vectorizer(ehr)
        events += self._problem_vectorizer(ehr)
        events += self._procedure_vectorizer(ehr)
        events += self._history_vectorizer(ehr)
        events += self._vital_vectorizer(ehr)

        # retrieve the latest encounter time
        try:
            latest_ect_time = max([e[0] for e in events if e[1] == "ECT"])
        except ValueError:
            try:
                latest_ect_time = self._DT_standardizer(
                    ehr["document"]["date"]
                )
            except KeyError:
                return
        events += self._text_vectorizer(ehr, latest_ect_time)

        if not events:
            return
        events = sorted(events)

        T = [i[0] for i in events]
        if encounter_limit:
            T = [t for t in T if t <= encounter_limit]

        T_delta = []
        for i, t in enumerate(T):
            if i == 0:
                T_delta.append(0)
            else:
                T_delta.append(t - T[i - 1])

        timeline_X = [i[2] for i in events[: len(T)]]
        timeline_Q = [i[3] for i in events[: len(T)]]

        T = np.array(T_delta, dtype="int32")
        X = np.array(timeline_X, dtype="int32")
        Q = np.array(timeline_Q, dtype="float32")

        if T.shape[0] >= max_sequence_length:
            T = T[-max_sequence_length:]
            X = X[-max_sequence_length:]
            Q = Q[-max_sequence_length:]
        else:
            short_seq_length = max_sequence_length - T.shape[0]
            T = np.pad(
                T, (short_seq_length, 0), "constant", constant_values=(0, 0)
            )
            Q = np.pad(
                Q, (short_seq_length, 0), "constant", constant_values=(0, 0)
            )
            padding_values = np.array([self.variable_size] * short_seq_length)
            X = np.concatenate((padding_values, X), 0)

        return T, X, Q

    ########################### PRIVATE FUNCTIONS #############################

    def _DT_standardizer(self, dt_str):
        if not dt_str:
            return None
        # use 1900-1-1 00:00:00 as base datetime
        base = datetime.strptime("01/01/1900", "%m/%d/%Y")
        # use time delta of base time to event time as rep
        std_dt = _try_parsing_date(dt_str) - base
        # convert time delta from seconds to 12-hour bucket-size integer
        std_dt = int(std_dt.total_seconds() / 3600)
        if std_dt <= 0:
            return None
        else:
            return std_dt

    def _add_ngrams(self, l):
        for token in self.bigram[l]:
            if "_" in token:
                l.append(token)
        return l

    # medication vectorizer
    def _medication_vectorizer(self, ehr):
        output = []
        if "medications" not in ehr.keys():
            return output
        for product in ehr["medications"]:
            s = self._DT_standardizer(product["date_range"]["start"])
            e = self._DT_standardizer(product["date_range"]["end"])
            if not s and not e:
                continue
            # TODO: deal with condition that no part of date range
            # if no start or no end date, make it one week long
            if not s and e:
                s = e - 14
            if s and not e:
                e = s + 14
            delta = e - s if e >= s else 0

            code = product["product"]["code"]
            system = product["product"]["code_system"]
            if (not code) or (not system):
                code = product["product"]["translation"]["code"]
                system = product["product"]["translation"]["code_system"]
                # short-circuit code_system too general
            if not system or len(system.split(".")) < 7:
                continue
            if code and system:
                code = ":".join(
                    [
                        self.code_system[
                            system.replace(
                                "2.16.840.1.000000", "2.16.840.1.113883"
                            )
                        ],
                        str(code),
                    ]
                )
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append(
                    (
                        s,
                        "MED",
                        index,
                        np.asscalar(self.time_norm.transform(delta)),
                    )
                )
        return list(set(output))

    # allergen vectorizer
    def _allergen_vectorizer(self, ehr):
        output = []
        if "allergies" not in ehr.keys():
            return output
        for allergen in ehr["allergies"]:
            s = self._DT_standardizer(allergen["date_range"]["start"])
            e = self._DT_standardizer(allergen["date_range"]["end"])
            # TODO: deal with condition that no part of date range
            # if no start or no end date, make it one year long or till the document date
            if not e:
                e = self._DT_standardizer(ehr["document"]["date"])
            if not e:
                continue
            if not s:
                s = e - 365 * 24 * 2
            if not s:
                continue
            delta = e - s if e >= s else 0

            code = allergen["allergen"]["code"]
            system = allergen["allergen"]["code_system"]
            if code and system:
                code = ":".join(
                    [
                        self.code_system[
                            system.replace(
                                "2.16.840.1.000000", "2.16.840.1.113883"
                            )
                        ],
                        str(code),
                    ]
                )
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append(
                    (
                        s,
                        "ALG",
                        index,
                        np.asscalar(self.time_norm.transform(delta)),
                    )
                )
        return list(set(output))

    # immuzation vectorizer
    def _immuzation_vectorizer(self, ehr):
        output = []
        if "immunizations" not in ehr.keys():
            return output
        for product in ehr["immunizations"]:
            date = self._DT_standardizer(product["date"])
            if not date:
                continue

            code = product["product"]["code"]
            system = product["product"]["code_system"]
            if (not code) or (not system):
                code = product["product"]["translation"]["code"]
                system = product["product"]["translation"]["code_system"]
                # short-circuit code_system too general
            if not system or len(system.split(".")) < 7:
                continue
            if code and system:
                code = ":".join(
                    [
                        self.code_system[
                            system.replace(
                                "2.16.840.1.000000", "2.16.840.1.113883"
                            )
                        ],
                        str(code),
                    ]
                )
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append((date, "IMU", index, 0))
        return list(set(output))

    # lab vectorizer
    def _lab_vectorizer(self, ehr):
        output = []
        if "results" not in ehr.keys():
            return output
        for results in ehr["results"]:
            #            test_code = results['code']
            #            if not test_code: test_code = 'UNK'
            #            test_system = results['code_system']
            #            if not test_system: test_code = ':'.join(['UNK', str(test_code)])
            #            else:
            #                try: test_code = ':'.join([self.code_system[test_system], str(test_code)])
            #                except KeyError: continue
            for test in results["tests"]:
                date = self._DT_standardizer(test["date"])
                if not date:
                    continue

                code = test["code"]
                system = test["code_system"]
                # if code or code_system not available, ckeck out translations
                if (not code) or (not system):
                    code = test["translation"]["code"]
                    system = test["translation"]["code_system"]
                    # short-circuit code_system too general
                if not system or len(system.split(".")) < 7:
                    continue
                if code and system:
                    value = test["value"]
                    if isinstance(value, str):
                        # convert positive or negative results into integers
                        if "pos" in value.lower():
                            value = 1
                        elif "neg" in value.lower():
                            value = 0
                        # sometimes value and unit together as text format, extract if possible
                        else:
                            value = re.search(
                                r"\-?(\d+|\d+,?)*\.\d+(e\d+)?|\d+", value
                            )
                            if value:
                                value = float(re.sub(r",", "", value.group()))
                    # eliminate codes numbers, lot numbers, or other non-value numbers
                    if (
                        str(value) in str(test["name"])
                        or "lot number" in str(test["name"]).lower()
                    ):
                        value = 1

                    # trim some code_system that too complicated
                    if (system.startswith("2")) and (
                        len(system.split(".")) > 7
                    ):
                        system = ".".join(system.split(".")[:7])
                    try:
                        code = ":".join(
                            [
                                self.code_system[
                                    system.replace(
                                        "2.16.840.1.000000",
                                        "2.16.840.1.113883",
                                    )
                                ],
                                str(code),
                            ]
                        )
                    except KeyError:
                        continue
                    #                    code = test_code+'--'+code
                    try:
                        index = self.all_variables.index(code)
                    except ValueError:
                        continue

                    # z-score normalization
                    if isinstance(value, (int, float)):
                        value = np.asscalar(
                            self.lab_norm[code].transform(value)
                        )
                    else:
                        value = 0
                    output.append((date, "LAB", index, value))
        return list(set(output))

    # encounter vectorizer
    def _encounter_vectorizer(self, ehr):
        output = []
        if "encounters" not in ehr.keys():
            return output
        for event in ehr["encounters"]:
            date = self._DT_standardizer(event["date"])
            if not date:
                continue

            for encounter in event["findings"]:
                code = encounter["code"]
                system = encounter["code_system"]
                if code and system:
                    # trim some code_system that too complicated
                    if (system.startswith("2")) and (
                        len(system.split(".")) > 7
                    ):
                        system = ".".join(system.split(".")[:7])
                    try:
                        code = ":".join(
                            [
                                self.code_system[
                                    system.replace(
                                        "2.16.840.1.000000",
                                        "2.16.840.1.113883",
                                    )
                                ],
                                str(code),
                            ]
                        )
                    except KeyError:
                        continue
                    try:
                        index = self.all_variables.index(code)
                    except ValueError:
                        continue
                    output.append((date, "ECT", index, 0))
        return list(set(output))

    # problem vectorizer
    def _problem_vectorizer(self, ehr):
        output = []
        if "problems" not in ehr.keys():
            return output
        for problem in ehr["problems"]:
            s = self._DT_standardizer(problem["date_range"]["start"])
            e = self._DT_standardizer(problem["date_range"]["end"])
            if not s:
                continue
            # TODO: deal with condition that no part of date range
            # if no end date, make it till the document date
            if not e:
                e = self._DT_standardizer(ehr["document"]["date"])
            if not e:
                continue
            delta = e - s if e >= s else 0

            code = problem["code"]
            system = problem["code_system"]
            if (not code) or (not system):
                code = problem["translation"]["code"]
                system = problem["translation"]["code_system"]
                # short-circuit code_system too general
            if not system or len(system.split(".")) < 7:
                continue
            if code and system:
                # trim some code_system that too complicated
                if (system.startswith("2")) and (len(system.split(".")) > 7):
                    system = ".".join(system.split(".")[:7])
                try:
                    code = ":".join(
                        [
                            self.code_system[
                                system.replace(
                                    "2.16.840.1.000000", "2.16.840.1.113883"
                                )
                            ],
                            str(code),
                        ]
                    )
                except KeyError:
                    code = ":".join([system, str(code)])
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append(
                    (
                        s,
                        "PRL",
                        index,
                        np.asscalar(self.time_norm.transform(delta)),
                    )
                )
        return list(set(output))

    # procedure vectorizer
    def _procedure_vectorizer(self, ehr):
        output = []
        if "procedures" not in ehr.keys():
            return output
        for procedure in ehr["procedures"]:
            date = self._DT_standardizer(procedure["date"])
            if not date:
                continue

            code = procedure["code"]
            system = procedure["code_system"]
            if code and system:
                # trim some code_system that too complicated
                if (system.startswith("2")) and (len(system.split(".")) > 7):
                    system = ".".join(system.split(".")[:7])
                try:
                    code = ":".join(
                        [
                            self.code_system[
                                system.replace(
                                    "2.16.840.1.000000", "2.16.840.1.113883"
                                )
                            ],
                            str(code),
                        ]
                    )
                except KeyError:
                    code = ":".join([system, str(code)])
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append((date, "PCD", index, 0))
        return list(set(output))

    # history vectorizer
    def _history_vectorizer(self, ehr):
        output = []
        if "social_history" not in ehr.keys():
            return output
        for record in ehr["social_history"]:
            s = self._DT_standardizer(record["date_range"]["start"])
            e = self._DT_standardizer(record["date_range"]["end"])
            if not s:
                continue
            # TODO: deal with condition that no part of date range
            # if no end date, make it till the document date
            if not e:
                e = self._DT_standardizer(ehr["document"]["date"])
            delta = e - s if e >= s else 0

            code = record["code"]
            system = record["code_system"]
            if code and system:
                code = ":".join(
                    [
                        self.code_system[
                            system.replace(
                                "2.16.840.1.000000", "2.16.840.1.113883"
                            )
                        ],
                        str(code),
                    ]
                )
                try:
                    index = self.all_variables.index(code)
                except ValueError:
                    continue
                output.append(
                    (
                        s,
                        "HTR",
                        index,
                        np.asscalar(self.time_norm.transform(delta)),
                    )
                )
        return list(set(output))

    # vital vectorizer
    def _vital_vectorizer(self, ehr):
        output = []
        if "vitals" not in ehr.keys():
            return output
        for results in ehr["vitals"]:
            date = self._DT_standardizer(results["date"])
            if not date:
                continue

            for test in results["results"]:
                code = test["code"]
                system = test["code_system"]
                if code and system:
                    value = test["value"]
                    if isinstance(value, str):
                        # convert positive or negative results into integers
                        if "pos" in value.lower():
                            value = 1
                        elif "neg" in value.lower():
                            value = 0
                        # sometimes value and unit together as text format, extract if possible
                        else:
                            value = re.search(
                                r"\-?(\d+|\d+,?)*\.\d+(e\d+)?|\d+", value
                            )
                            if value:
                                value = float(re.sub(r",", "", value.group()))
                    # eliminate codes numbers, lot numbers, or other non-value numbers
                    if (
                        str(value) in str(test["name"])
                        or "lot number" in str(test["name"]).lower()
                    ):
                        value = 1

                    if (system.startswith("2")) and (
                        len(system.split(".")) > 7
                    ):
                        system = ".".join(system.split(".")[:7])
                    code = ":".join(
                        [
                            self.code_system[
                                system.replace(
                                    "2.16.840.1.000000", "2.16.840.1.113883"
                                )
                            ],
                            str(code),
                        ]
                    )
                    try:
                        index = self.all_variables.index(code)
                    except ValueError:
                        continue

                    # z-score normalization
                    if isinstance(value, (int, float)):
                        value = np.asscalar(
                            self.vital_norm[code].transform(value)
                        )
                    else:
                        value = 0
                    output.append((date, "VIT", index, value))
        return list(set(output))

    # text vectorizer
    def _text_vectorizer(self, ehr, latest_ect_time):
        output = []
        texts = []
        if "chief_complaint" not in ehr.keys():
            pass
        else:
            texts.append(
                re.sub(
                    r"(\n|[^\x00-\x7F]+)",
                    " ",
                    str(ehr["chief_complaint"]["text"]),
                )
            )
        if "diagnosis" not in ehr.keys():
            pass
        else:
            texts.append(
                re.sub(
                    r"(\n|[^\x00-\x7F]+)", " ", str(ehr["diagnosis"]["text"])
                )
            )
        if "history_of_present_illness" not in ehr.keys():
            pass
        else:
            texts.append(
                re.sub(
                    r"(\n|[^\x00-\x7F]+)",
                    " ",
                    str(ehr["history_of_present_illness"]["text"]),
                )
            )
        if "instructions" not in ehr.keys():
            pass
        else:
            for item in ehr["instructions"]:
                texts.append(
                    re.sub(r"(\n|[^\x00-\x7F]+)", " ", str(item["text"]))
                )
        if "physical_exam" not in ehr.keys():
            pass
        else:
            texts.append(
                re.sub(
                    r"(\n|[^\x00-\x7F]+)",
                    " ",
                    str(ehr["physical_exam"]["text"]),
                )
            )
        if "system_review" not in ehr.keys():
            pass
        else:
            texts.append(
                re.sub(
                    r"(\n|[^\x00-\x7F]+)",
                    " ",
                    str(ehr["system_review"]["text"]),
                )
            )

        for text in texts:
            text = preprocessing.preprocess(text, negex=True)
            text = re.sub(r"[A-Z]+", "", text)
            text = re.sub(r"(&#39;|<\/?p>|&nbsp;)", "", text)
            date = re.search(
                r"[0-9]{1,4}(\/|-)[0-9]{1,2}(\/|-)[0-9]{1,4}(\s*T?[0-9]{1,2}(:[0-9]{1,2})+Z?)?",
                text,
            )
            try:
                date = self._DT_standardizer(date.group())
            except:
                date = latest_ect_time

            text = "".join(c for c in text if c not in self.punctuation)
            text = re.sub(self.stopwords, "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            text = [i for i in re.split("\s", text) if i]
            text = self._add_ngrams(text)
            text = [i for i in self.vocab.doc2idx(text) if i >= 0]
            for i in text:
                output.append((date, "TXT", i + len(self.all_variables), 0))

        return list(set(output))


class ClaimVectorizer(object):
    """
    Aim to extract and vectorize EHR CCD data into common claims codes
    """

    def __init__(self):
        """
        Initialize a vectorizer to repeat use; load section variable spaces
        """
        self.code_system = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "pickle_files",
                    "code_system",
                ),
                "rb",
            )
        )
        self.rxnorm_to_gpi = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "pickle_files",
                    "rxnorm_to_gpi",
                ),
                "rb",
            )
        )
        self.snomed_to_icd = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "pickle_files",
                    "snomed_to_icd",
                ),
                "rb",
            )
        )

    def fit_transform(self, ehr, date_start=None):
        if not date_start:
            date_start = datetime.strptime("01/01/1900", "%m/%d/%Y")
        else:
            date_start = datetime.strptime(date_start, "%Y-%m-%d")
        output = []

        output += _general_code_extract(
            "medications", ehr, self.code_system, date_start
        )
        output += _general_code_extract(
            "immunizations", ehr, self.code_system, date_start
        )
        output += _hierachical_code_extract(
            "results", "tests", ehr, self.code_system, date_start
        )
        output += _general_code_extract(
            "problems", ehr, self.code_system, date_start
        )
        output += _hierachical_code_extract(
            "encounters", "findings", ehr, self.code_system, date_start
        )
        output += _general_code_extract(
            "procedures", ehr, self.code_system, date_start
        )
        output += _hierachical_code_extract(
            "vital", "results", ehr, self.code_system, date_start
        )

        output = [
            self.rxnorm_to_gpi[c] if c in self.rxnorm_to_gpi else c
            for c in output
        ]
        output = [
            self.snomed_to_icd[c] if c in self.snomed_to_icd else c
            for c in output
        ]
        return list(set(output))


############################# PRIVATE FUNCTIONS ###############################


def _try_parsing_date(dt_str):
    if not dt_str:
        return None
    # TODO: add potential datetime format if possible
    for fmt in (
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m-%d-%Y",
        "%m-%d-%Y %H:%M",
    ):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    raise ValueError("No valid datetime format found for " + dt_str)


def _general_code_extract(tag, ehr, code_system, date_start):
    if tag not in ehr.keys():
        return []
    output = []
    for product in ehr[tag]:
        try:
            date = _try_parsing_date(product["date_range"]["start"])
        except KeyError:
            date = _try_parsing_date(product["date"])
        if not date or date < date_start:
            continue

        if tag == "medications" or tag == "immunizations":
            product = product["product"]
        code = product["code"]
        system = product["code_system"]
        if (not code) or (not system):
            if "translation" in product:
                code = product["translation"]["code"]
                system = product["translation"]["code_system"]
            else:
                continue
            # short-circuit code_system too general
        if not system or len(system.split(".")) < 7:
            continue
        if code and system:
            # trim some code_system that too complicated
            if (system.startswith("2")) and (len(system.split(".")) > 7):
                system = ".".join(system.split(".")[:7]).replace(
                    "2.16.840.1.000000", "2.16.840.1.113883"
                )
            try:
                code = "-".join([code_system[system], str(code)])
                output.append(code)
            except:
                continue
    return output


def _hierachical_code_extract(tag, sub_tag, ehr, code_system, date_start):
    if tag not in ehr.keys():
        return []
    output = []
    for results in ehr[tag]:
        try:
            date = _try_parsing_date(results["date"])
            if not date or date < date_start:
                continue
        except KeyError:
            pass

        for test in results[sub_tag]:
            try:
                date = _try_parsing_date(results["date"])
                if not date or date < date_start:
                    continue
            except KeyError:
                pass

            code = test["code"]
            system = test["code_system"]
            # if code or code_system not available, ckeck out translations
            if (not code) or (not system):
                if "translation" in test:
                    code = test["translation"]["code"]
                    system = test["translation"]["code_system"]
                else:
                    continue
                # short-circuit code_system too general
            if not system or len(system.split(".")) < 7:
                continue
            if code and system:
                # trim some code_system that too complicated
                if (system.startswith("2")) and (len(system.split(".")) > 7):
                    system = ".".join(system.split(".")[:7]).replace(
                        "2.16.840.1.000000", "2.16.840.1.113883"
                    )
                try:
                    code = "-".join([code_system[system], str(code)])
                    output.append(code)
                except KeyError:
                    continue
    return output
