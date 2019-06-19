# -*- coding: utf-8 -*-
###############################################################################
# Module:      fetch_db
# Description: repo of database functions for robbins
# Authors:     William Kinsman, Yage Wang
# Created:     03.26.2017
###############################################################################

import re
import pyodbc
from datetime import datetime

def info(member='KC87853B',client='HEALTHFIRST',db='MRRBWDB',date_start=None,date_end=None):
    """
    @param member: member ID (client)
    @param client: the name of the client table
    @param db: the database of the table
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD
    """

    # fetch personal info
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+str(db)+';').cursor()
    
    # if dates provided
    if date_start and date_end:
        data = cursor.execute("""
        Declare @Mem_ID NVARCHAR(200) = '"""+str(member)+"""'
        Declare @ClientName VARCHAR (200) = '"""+str(client)+"""'
        Declare @StartDate Date = '"""+date_start+"""'
        Declare @EndDate Date = '"""+date_end+"""'
        SELECT M.[MemberFirstName] AS name_first,M.[MemberMiddleName] AS name_middle,M.[MemberLastName] AS name_last,
           M.[MemberDOB] AS dob,M.[MemberGender] as gender,CRD.[ZipCode] as zip,CRD.[PackageId] as ccds,CRD.[Address1]+' '+CRD.[Address2] as address,
    		CRD.[City] as city,CRD.[State] as state
        FROM [EHR].[EHR].[ChaseRequestDetail] CRD WITH(NOLOCK)
           JOIN [EHR].[EHR].[Client] C WITH(NOLOCK) ON CRD.ClientId = C.Id
           JOIN [SAFRMPMPD000005].[MRRDataCollection].[DCL].[Member] M WITH(NOLOCK) ON CRD.SafhireClientMemberId = M.ClientMemberId
           JOIN [SAFRMPMPD000005].[MRRProjectManagement].[LKP].[Client] C1 WITH(NOLOCK) ON C.SafhireClientId = C1.ID
        WHERE m.ClientMemberId = @Mem_ID
    		AND CRD.StartDate = @StartDate AND CRD.EndDate = @EndDate
    		AND C1.ShortName = @ClientName""").fetchall()
        
    #if no dates provided
    else:
        data = cursor.execute("""
        Declare @Mem_ID NVARCHAR(200) = '"""+str(member)+"""'
        Declare @ClientName VARCHAR (200) = '"""+str(client)+"""'
        SELECT M.[MemberFirstName] AS name_first,M.[MemberMiddleName] AS name_middle,M.[MemberLastName] AS name_last,
           M.[MemberDOB] AS dob,M.[MemberGender] as gender,CRD.[ZipCode] as zip,CRD.[PackageId] as ccds,CRD.[Address1]+' '+CRD.[Address2] as address,
    		CRD.[City] as city,CRD.[State] as state
        FROM [EHR].[EHR].[ChaseRequestDetail] CRD WITH(NOLOCK)
           JOIN [EHR].[EHR].[Client] C WITH(NOLOCK) ON CRD.ClientId = C.Id
           JOIN [SAFRMPMPD000005].[MRRDataCollection].[DCL].[Member] M WITH(NOLOCK) ON CRD.SafhireClientMemberId = M.ClientMemberId
           JOIN [SAFRMPMPD000005].[MRRProjectManagement].[LKP].[Client] C1 WITH(NOLOCK) ON C.SafhireClientId = C1.ID
        WHERE m.ClientMemberId = @Mem_ID
    		AND C1.ShortName = @ClientName""").fetchall()
    
    # organize the info into a dictionary
    output = None
    try:
        output = {cursor.description[i][0]:data[0][i] for i in range(len(cursor.description))}
        output['ccds'] = _ccdID_to_path([i[6] for i in data])
    except : pass
    return output


def claims(member='KC87853B',client='CD_HEALTHFIRST',db='CARABWDB03',with_time=False,date_start=None,date_end=None):
    """
    @param member: member ID (client)
    @param client: the name of the client table
    @param db: the database of the table
    @param with_time: if True, return codes with encounter timestamps
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD
    """
    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+db+';').cursor()
    
    # convert member ID
    cursor.execute("""SELECT mem_id, mem_ClientMemberID
                   FROM """+client+""".dbo.tbMember M
                   WHERE mem_ClientMemberID = '"""+str(member)+"""'
                   """)
    try: member = list(i for i in cursor)[0][0]
    except Exception as ex:
        # print(ex, "Member not found!")
        return
    
    date_start, date_end = str(date_start).replace('-', ''), str(date_end).replace('-', '')
    # if dates are provided
    sql = """
    DECLARE @Mem_ID INT = """ +str(member)+ """
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'CPT' AS CodeType, eCPT.cpt_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'DRG' AS CodeType, eDRG.DRG_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbEncounterDRG eDRG ON e.enc_id = eDRG.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'HCPCS' AS CodeType, eHCPCS.HCPCS_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbEncounterHCPCS eHCPCS ON e.enc_id = eHCPCS.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""' 
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9Proc' WHEN 10 THEN 'ICD10Proc' ELSE 'ICD9Proc' END AS CodeType, eProc.icd_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, p.pha_id AS EncounterID, pha_ServiceDate AS ServiceDate, 'NDC9' AS CodeType, NDC.ndcl_NDC9Code AS Code
    FROM """+client+""".dbo.tbPharmacy p WITH(NOLOCK)
                    INNER JOIN """+client+""".dbo.tbPharmacyNDC NDC ON p.pha_id = NDC.pha_id
    WHERE mem_id = @Mem_id AND p.pha_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    ORDER BY enc_ServiceDate, e.enc_id
    """
    sql = re.sub(r" AND (e|p)\.[a-z\_]+ServiceDate BETWEEN 'None' AND 'None'", '', sql)
    cursor.execute(sql)
    
    if with_time:
        output = {}
        base = datetime.strptime('01/01/1900', '%m/%d/%Y')
        for i in cursor:
            date = i[2]
            std_dt = date - base
            std_dt = int(std_dt.total_seconds() / 3600)
            if std_dt <= 0: std_dt = 0
            if std_dt in output: output[std_dt].append(i[3] + '-' + i[4])
            else: output[std_dt] = [i[3] + '-' + i[4]]
        for k,v in output.items(): output[k] = list(set(v))
        return output
    else: return list(set([i[3] + '-' + i[4] for i in cursor]))

def readmission_claims(member='KC87853B',client='CD_HEALTHFIRST',db='CARABWDB03',date_start=None,date_end=None):
    """
    @param member: member ID (client)
    @param client: the name of the client table
    @param db: the database of the table
    @param with_time: if True, return codes with encounter timestamps
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD
    """
    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+db+';').cursor()
    
    # convert member ID
    cursor.execute("""SELECT mem_id, mem_ClientMemberID
                   FROM """+client+""".dbo.tbMember M
                   WHERE mem_ClientMemberID = '"""+str(member)+"""'
                   """)
    try: member = list(i for i in cursor)[0][0]
    except Exception as ex:
        # print(ex, "Member not found!")
        return
    
    date_start, date_end = str(date_start).replace('-', ''), str(date_end).replace('-', '')
    # if dates are provided
    sql = """
    DECLARE @Mem_ID INT = """ +str(member)+ """
    SELECT @Mem_ID AS MemberID, enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, 'UBTOB' AS CodeType, CASE LEN(bltp_Code) WHEN 3 THEN '0'+bltp_Code ELSE bltp_Code END AS Code
    FROM """+client+""".dbo.tbEncounter WITH(NOLOCK)
    WHERE mem_id = @Mem_id AND bltp_Code IS NOT NULL AND enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, 'UBREV' AS CodeType, CASE LEN(eREV.rev_Code) WHEN 3 THEN '0'+eREV.rev_Code ELSE eREV.rev_Code END AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
    INNER JOIN """+client+""".dbo.tbEncounterRevenue eREV ON e.enc_id = eREV.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
    INNER JOIN """+client+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, CASE icdVersionInd WHEN 9 THEN 'ICD9Proc' WHEN 10 THEN 'ICD10Proc' ELSE 'ICD9Proc' END AS CodeType, eProc.icd_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
    INNER JOIN """+client+""".dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, 'CPT' AS CodeType, eCPT.cpt_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
    INNER JOIN """+client+""".dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT @Mem_ID AS MemberID, e.enc_ID AS EncounterID, enc_serviceDate AS AdmissionDate, enc_DischargeDate as DischargeDate, 'HCPCS' AS CodeType, eHCPCS.HCPCS_Code AS Code
    FROM """+client+""".dbo.tbEncounter e WITH(NOLOCK)
    INNER JOIN """+client+""".dbo.tbEncounterHCPCS eHCPCS ON e.enc_id = eHCPCS.enc_id
    WHERE mem_id = @Mem_id AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    ORDER BY AdmissionDate, EncounterID
    """
    sql = re.sub(r" AND (e|p)?\.?[a-z\_]+ServiceDate BETWEEN 'None' AND 'None'", '', sql)
    cursor.execute(sql)
    
    return list(set([(i[1], i[2], i[3], i[4] + '-' + i[5]) for i in cursor]))

##############################  PRIVATE FUNCTIONS #############################

def _ccdID_to_path(packageID):
    """
    @param packageID or list of packageIDs (a zipped ccd package)
    """
    # initialization
    if packageID is None: return None
    if isinstance(packageID, (int, str)): packageID = [packageID]
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER=EQBWDB1;').cursor()

    # fetch results
    ccds = None
    for table in ['Package', 'Package_Archive']:
        ccds = cursor.execute("""
            SELECT Id, [ArchiveFilePath] 
            FROM [EnterpriseApplications.Queue].[EnterpriseWorkflow].["""+table+"""] WITH(NOLOCK)
            WHERE Id IN ("""+','.join(['\''+str(i)+'\'' for i in packageID if i is not None])+""")""")
        paths = [i[1] for i in ccds]
        if paths: break
    return paths
