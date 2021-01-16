import numpy as np
import glob
import os
import pandas as pd
import datefinder

paths = glob.glob("results/*")
node_labels = ['other', 'company', 'address', 'date', 'total']
data_frame = []
count = 0
for path in paths :
    results = {"company":[],"address":[],"date":[],"total":[]}
    outputs   = {"company":"","address":"","date":"","total":""}
    name_id = os.path.basename(path).replace("txt","jpg")
    with open(path,"r") as file :
        data = file.readlines()
        for line in data:
            tmp = line.strip().split("\t")
            x,y = tmp[:2]
            text = tmp[10]
            class_id = tmp[11]
            save = [int(x),int(y),text,class_id]
            results[class_id].append(save)
        for k,v in results.items():
            if k =="company" or k == "address": 
                tmp2 = sorted(v , key=lambda k: [k[1]])
            if k=="total"  or k =="date":
                tmp2 = sorted(v , key=lambda k: [k[0]])
            labels = " ".join([str(i[2]) for i in tmp2])
            
            if k== "date" and "Ngày" in labels :
                labels =  "Ngày" + labels.split("Ngày")[1]
            if k== "date" and "Thời" in labels :
                labels =  "Thời" + labels.split("Thời")[1]
            outputs[k] =  labels
        if len(outputs["date"]) == 0 :
            name =  os.path.basename(path)
            with open(f"OCR_rotated/{name}","r") as f :
                file = f.readlines()
                for i in file :
                    tmp = i.strip().split("\t")[8]
                    tmp2 = tmp.split(" ")
                    for k in tmp2 :
                        if len(k)<10 and (k.count("-") <2 or k.count("/")<2):
                            continue
                        else :
                            try  :
                                matches = datefinder.find_dates(k)
                                for match in matches:
                                    if "Ngày" in tmp:
                                        outputs["date"] = "Ngày" + tmp.split("Ngày")[1]

                                        break
                                    elif "Thời" in tmp :
                                        outputs["date"] = "Thời" + tmp.split("Thời")[1]
                                        break
                                    else :
                                        outputs["date"] = tmp
                                        break
                            except :
                                continue
        if len(outputs["date"]) == 0 :
            count += 1
                    
    labels_save = outputs["company"]+"|||"+ outputs["address"]+"|||" + outputs["date"]+"|||" + outputs["total"]
    print(labels_save)
    data_frame.append([name_id,0.5,labels_save])

data_frame.append(["mcocr_val_145114unyae.jpg",0.5,"|||||||||"])
columns = ["img_id","anno_image_quality","anno_texts"]
df = pd.DataFrame(data_frame,columns=columns)
print(df.head())
print(len(df))
df.to_csv("results.csv",index=None,columns=columns)
