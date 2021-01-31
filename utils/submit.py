import numpy as np
import glob
import os
import pandas as pd
import datefinder
import cv2

paths = glob.glob("rs1/*")
node_labels = ['other', 'company', 'address', 'date', 'total']
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def hasCharacters(inputString):
    return any(char.isalpha() for char in inputString)


data_frame = []
count = 0
for path in paths :
    results = {"company":[],"address":[],"date":[],"total":[]}
    outputs   = {"company":"","address":"","date":"","total":""}
    name = os.path.basename(path)
    name_id = name.replace("txt","jpg")
#     image = cv2.imread(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/data_test/{name_id}")
    image_show = image.copy()
    with open(path,"r") as file :
        data = file.readlines()
        for line in data:
            tmp = line.strip().split("\t")
            x,y = tmp[:2]
            text = tmp[10]
            class_id = tmp[11]
            save = [int(x),int(y),text,class_id]
            results[class_id].append(save)      
        for key,v in results.items():       
            if key == "date" :
                for j in results[key]:
                    if "Ngày" in j[2] :
                        j[2] =  "Ngày" + j[2].split("Ngày")[1]
                    if "Thời" in j[2] :
                        j[2] =  "Thời" + j[2].split("Thời")[1]
                if len(results[key])==0:
                    with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/OCR_PARSER/{name}","r") as f :
                        file = f.readlines()
                        for i in file :
                            x1,y1 = i.strip().split("\t")[:2]
                            tm = i.strip().split("\t")[8]
                            tm2 = tm.split(" ")
                            for k in tm2 :
                                if len(k)<10 and (k.count("-") <2 or k.count("/")<2):
                                    continue
                                else :
                                    try  :
                                        matches = datefinder.find_dates(k)
                                        for match in matches:
#                                             print(tm)
                                            
                                            if "Ngày" in tm:
                                                
                                                results[key].append([int(x1),int(y1),"Ngày" + tm.split("Ngày")[1],"date"])
                                            
#                                                 break
                                            elif "Thời" in tm :
                                                results[key].append([int(x1),int(y1),"Thời" + tm.split("Thời")[1],"date"])
#                                                 break
                                            else :
                                                results[key].append([int(x1),int(y1),tm,"date"])
#                                                 break
                                    except :
                                        continue
    #sorted(my_list , key=lambda k: [k[1], k[0]])
    company = "|||".join([str(i[2]) for i in sorted(results["company"] , key=lambda k: k[0])]) 
    adress = "|||".join([str(i[2]) for i in sorted(results["address"] , key=lambda k: k[0])])
#     print(results["date"])
    date = "|||".join([str(i[2]) for i in sorted(results["date"] , key=lambda k: k[0])])
#     print("--->",date)
    total = "|||".join([str(i[2]) for i in sorted(results["total"] , key=lambda k: k[0])])
    labels_save =  company+"|||"+ adress + "|||" + date + "|||" + total
    print("="*20,name_id,"="*20)
#     print(results["date"])
    print(labels_save)
    print("-"*50)
    data_frame.append([name_id,0.5,labels_save])


columns = ["img_id","anno_image_quality","anno_texts"]
df = pd.DataFrame(data_frame,columns=columns)
print(df.head())
df1 = pd.read_csv("/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/mcocr_test_samples_df.csv")
df1.head()

df_final = df1.reset_index()[['index', 'img_id']].merge(df, on='img_id', how='left').fillna('|||||||||').sort_values(by='index').drop(columns='index')
df_final.to_csv("submit_final/results.csv",index=None)
