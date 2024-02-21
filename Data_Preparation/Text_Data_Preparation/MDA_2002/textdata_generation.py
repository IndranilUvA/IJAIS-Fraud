import numpy as np
import pandas as pd
import csv
import os
import re


filepath="C:\\Users\\indra\\Desktop\\NLP Ana paper3\\Data Preparation\\2002"
filepath2="C:\\Users\\indra\\Desktop\\NLP Ana paper3\\Data Preparation\\2002"
filepath3="C:\\Users\\indra\\Desktop\\NLP Ana paper3\\Data Preparation\\2002"

SD=os.path.join(filepath3,"SampleData.csv")
download=os.path.join(filepath3,"DOWNLOADLOG.txt")

with open(SD,'w') as f: 
    f.write("File,CompanyName,CIK,SIC,ReportDate,Section,Text\n")

sayings=["the following discussion","this discussion and analysis","should be read in conjunction", "should be read together with", "the following managements discussion and analysis"]
acq=["proprietary","intellectual","patent","trademark","intangible","technology"]    
personnel = ["key personnel", "personnel","talented employee", "key talent"]

check = pd.DataFrame({'File': [],'Name': [], 'CIK': [] , 'SIC':[], 'report_date': [], 'Sections':[], 'text': []})

with open(download, 'r') as txtfile:
    reader = csv.reader(txtfile, delimiter='\t')
    next(reader, None)
    for line in reader:
        FileNUM=line[0].strip()
        Sections=int(line[1].strip())
        if Sections!=0:
            Filer=os.path.join(filepath,str(FileNUM)+".txt")
            CLEAN=os.path.join(filepath3,str(FileNUM)+".txt")
            SIC=""
            Info=[str(FileNUM)]
            hand=open(Filer)
            for line in hand:    
                line=line.strip()
                if re.findall('^COMPANY NAME:',line):
                    COMNAM=line.replace("COMPANY NAME: ","")
                if re.findall('^CIK:',line):
                    CIK=line.replace("CIK: ","")
                if re.findall('^SIC:',line):
                    SIC=line.replace("SIC: ","")
                if re.findall('^REPORT PERIOD END DATE:',line):
                    REPDATE=line.replace("REPORT PERIOD END DATE: ","")
            Info.append(COMNAM)
            Info.append(CIK)
            if SIC=="":
                SIC='9999'
                Info.append(SIC)
            else:
                Info.append(SIC)
                
            Info.append(REPDATE)
            Info.append(str(Sections))
         
            str1=open(Filer).read()
            locations=[]
            for m in re.finditer("<SECTION>",str1):
                a=m.end()
                locations.append(a)
            for m in re.finditer("</SECTION>",str1):
                a=m.start()
                locations.append(a)
            if locations!=[]:
                locations.sort()
                            
            if Sections==1:
                substring1=str1[locations[0]:locations[1]]
                substring1=substring1.lower()
                substring1=re.sub('\d','',substring1)
                substring1=substring1.replace(',','')
                substring1=substring1.replace(':',' ')
                substring1=substring1.replace('$','')
                substring1=substring1.replace('(','')
                substring1=substring1.replace(')','')
                substring1=substring1.replace('%','')
                substring1=substring1.replace('"','')
                substring1=substring1.replace('-',' ')
                substring1=substring1.replace('[','')
                substring1=substring1.replace(';',' ')
                substring1=substring1.replace(']','')
                substring1=substring1.replace('_','')
                substring1=substring1.replace('|','')
                substring1=substring1.replace('/','')
                substring1=substring1.replace('`','')
                substring1=substring1.replace("'",'')
                substring1=substring1.replace('&','')
                TWORD = "NULL"
                TWORD = substring1
                Post=[]
                Post.extend(Info)
                Post.append(str(TWORD))
                
                with open(CLEAN,'a') as f:
                    f.write("<SECTION>\n")
                    f.write(' '.join(substring1)+"\n")
                    f.write("</SECTION>\n")
                    f.close()
                with open(SD,'a') as f:
                    f.write(','.join(Post)+'\n')
                    f.close    

                Info.append(substring1)
                check = check.append({'File': Info[0],
                                              'Name': Info[1],
                                              'CIK': Info[2] ,
                                              'SIC':Info[3],
                                              'report_date': Info[4],
                                              'Sections':Info[5],
                                              'text': Info[6]}, ignore_index=True)
                Post=[]
            else:
                for k in range(0,len(locations),2):
                    filed=0
                    substring1=str1[locations[0+k]:locations[1+k]]
                    substring1=substring1.lower()
                    substring1=substring1.split(". ")
                    if len(substring1)>5:
                        for j in range(0,6):
                            if any(s in substring1[j] for s in sayings):
                                filed=1
                                break
                    if filed==1:
                        substring1=str1[locations[0+k]:locations[1+k]]
                        substring1=substring1.lower()
                        substring1=re.sub('\d','',substring1)
                        substring1=substring1.replace(',','')
                        substring1=substring1.replace(':',' ')
                        substring1=substring1.replace('$','')
                        substring1=substring1.replace('(','')
                        substring1=substring1.replace(')','')
                        substring1=substring1.replace('%','')
                        substring1=substring1.replace('"','')
                        substring1=substring1.replace('-',' ')
                        substring1=substring1.replace('[','')
                        substring1=substring1.replace(';',' ')
                        substring1=substring1.replace(']','')
                        substring1=substring1.replace('_','')
                        substring1=substring1.replace('|','')
                        substring1=substring1.replace('/','')
                        substring1=substring1.replace('`','')
                        substring1=substring1.replace("'",'')
                        substring1=substring1.replace('&','')
                        text = substring1
                        TWORD = "NULL"
                        TWORD = substring1
                        Post=[]
                        Post.extend(Info)
                        Post.append(str(TWORD))
                        
                        with open(CLEAN,'a') as f:
                            f.write("<SECTION>\n")
                            f.write(' '.join(substring1)+"\n")
                            f.write("</SECTION>\n")
                            f.close()
                        with open(SD,'a') as f:
                            f.write(','.join(Post)+'\n')
                            f.close
                        Info.append(substring1)

                        Post=[]
                        
                        check = check.append({'File': Info[0],
                                              'Name': Info[1],
                                              'CIK': Info[2] ,
                                              'SIC':Info[3],
                                              'report_date': Info[4],
                                              'Sections':Info[5],
                                              'text': Info[6]}, ignore_index=True)

header_list = ["File", "url_cut"]
old_file = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/2002/downloadlist.txt", sep=',', names=header_list)
old_file.File = old_file.File.astype(str)
new = pd.merge(check, old_file, on="File")

master_df = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/master_df.csv")
master_df_10k = master_df[master_df["filing_type"] == "10-K"]

master_df_10k["month"] = master_df_10k.apply(lambda row: row["filing_date"][5:-3] ,axis=1)
master_df_10k["year"] = master_df_10k.apply(lambda row: row["filing_date"][0:4] ,axis=1)
master_df_10k["date"] = master_df_10k.apply(lambda row: row["filing_date"][8:] ,axis=1)

master_df_10k_2002 = master_df_10k[master_df_10k["year"] == "2002"]
master_df_10k_2002["url_cut"] = master_df_10k_2002.apply(lambda row: row["url"][29:] ,axis=1)

final_2002_text = pd.merge(new, master_df_10k_2002, on="url_cut")
final_2002_text.CIK = final_2002_text.CIK.astype(int)
final_2002_text.cik = final_2002_text.cik.astype(str)
#### From AAER exploration, the following list is obtained from the AAER data ####
fraud_2002_cik = []# This is list of CIKs that are identified as fraudulent
final_2002_text["fraud"] = np.where(final_2002_text["cik"].isin(fraud_2002_cik), 1 , 0)

cols = ['cik','filing_date', 'url','month', 'year', 'date','text','fraud']
final_2002_text = final_2002_text[cols]
final_2002_text['text'] = final_2002_text['text'].apply(lambda x: x.replace("\n",''))
final_2002_text.to_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/2002/final_2002_text.csv", index = False)