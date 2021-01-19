import glob
import os
import numpy  as np
import cv2
import pandas as pd
import json
import ast
import random
import tqdm
import re

from faker import Faker
import random
fake = Faker()


def gen_company(fake):
    company_prefix = ["NHÀ SÁCH","Mart","Siêu thị","Cửa hàng","SHOP","công ty","nhà ăn","quán ăn"]
    comapy_post = ["Mart","COFFEE","Co.op","Minimart","Store","book"]
    company_dic = ["BIG C DI AN","VinCommerce","MINIMART ANAN","CƠM GÀ BẢO NGỌC","NHÀ SÁCH GD-TC CẨM PHẢ",
              "Phúc Anh Minimart","PHỐ MỎ","CỬA HÀNG NĂM OÁNH","Guitar Cafe","Siêu thị Co.op Mart Foodcosa",
              "THE COFFEE HQUSE","THỨC COFFEE","co.op smile","VINMART",
              "SIEU THI BACH HOA TONG HOP","Co.opMart HAU GIANG","Saigon Co.op","Ngọc Fruit","Phúc Anh Minimimart",
              "Siêu thị Vinmart  Quang Trung","Vitafuist","King Fruit","Bác Tôm","Hoa quả Minh Hoan","Mobifood",
              "T&T Fruit Shop","Aeon citimart","Cửa hàng Satrafoods","Thực phẩm NS xanh","Cửa hàng Tiện Lợi",
              "Ngôi Sao Xanh","MILANO COFFEE","Thực phẩm sạch","Circle K",
              "7-Eleven","Shop & Go","Satrafoods","Cheers","GS25","Speed L","Oceanbank","GPBank","Agribank"," ACB",
                  "VIB","VietBank","Sacombank","Techcombank"]
#     compamy = fake.company()
#     if random.random() > 0.7 :
#         return random.choice(company_dic)
#     elif  (random.random() < 0.7)  and (random.random() > 0.3):
#         return random.choice(company_prefix) + " " + compamy
#     else:
    return random.choice(company_dic)

pattern_total_text = ["Tổng tiền:",'TỔNG TIỀN PHẢI T.TOÁN','Tong so tien thanh toan:',
                     'TẠI QUẦY:','Cộng tiền hàng','Tiên Thanh Toán','Tổng thanh toán','Tổng cộng :',
                     'Thanh Toán:','Thành Tiền','Cộng tiền hàng','Tiên Thanh Toán',
                      'Phải thu khách hàng:',"Total","KHACH PHAI TRA",]
def gen_total(fake):
    x = random.randint(10,1000)
    first = "{:,}".format(x)
    if random.random() > 0.5 :
        return str(first).replace(",",".") + ".000" +  "d"
    else :
         return str(first).replace(",",".") + ".000"

streets = []
with open("address.txt","r") as f1 :
    st = f1.readlines()
    for i in st :
        streets.append(i.strip())
quan = []
with open("quan.txt","r") as f1 :
    st = f1.readlines()
    for i in st :
        quan.append(i.strip())
phuong = []
with open("phuong.txt","r") as f1 :
    st = f1.readlines()
    for i in st :
        phuong.append(i.strip())
tp = ["HCM","HA NOI","QUANG NINH","BINH DUONG","HUE","DA NANG","NGHE AN","DONG NAI","QUANG NAM","HA TINH"]
def gen_address(stype = 1):
    if stype == 1:
        return random.choice(["DC: ",""]) +  random.choice(streets)  \
                + ", "+ random.choice(quan) + ", " +  random.choice(tp)
    if stype==2:
        p1 = random.choice(["DC: ",""]) +  random.choice(streets)
        p2= random.choice(phuong) + ", "+random.choice(quan) + ", " +  random.choice(tp)
        return [p1,p2]
    if stype==3 :
        p1 = random.choice(["DC: ",""]) +  random.choice(streets)
        p2= random.choice(streets) + ", " + random.choice(phuong)  
        p3 = random.choice(quan) + ", " +  random.choice(tp)
        return [p1,p2,p3]



def hasNumbers(inputString):
    inputString = re.sub(r'[^\w\s]','',inputString)
    return any(char.isdigit() for char in inputString)

def hasCharacters(inputString):
    inputString = re.sub(r'[^\w\s]','',inputString)
    return all(char.isalpha() for char in inputString)
paths = glob.glob("/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/data_train_graph/*")
for i in tqdm.tqdm(range(3000)):
    choice = random.choice(paths)
    results = []
    with open(choice,"r") as f :
        data = f.readlines()
        save_dic = {"company":[],"address":[],"date":[],"total":[]}
        for line in data :
            each_line = line.strip().split("\t")
            class_id = each_line[-1]
            if class_id == "other":
                results.append(each_line)
            else :
                save_dic[class_id].append(each_line)
        for k,v in save_dic.items():
            if k == "company":
                if len(v) ==  1 :
                    save_dic[k][0][8] = gen_company(fake)
#             if k == "address":
                
#                 if len(v)==1 :
#                     save_dic[k][0][8] = gen_address(stype=1)
#                 if len(v)==2:
#                     gen =  gen_address(stype=2)
#                     save_dic[k][0][8] =  gen[0]
#                     save_dic[k][1][8] = gen[1]
#                 if len(v)==3:
#                     gen =  gen_address(stype=3)
#                     save_dic[k][0][8] =  gen[0]
#                     save_dic[k][1][8] = gen[1]
#                     save_dic[k][1][8] = gen[2]

            if k=="date":
                if len(v)==1 :
                    save_dic[k][0][8] = gen_date(fake)
            if k=="total":
                if len(v)==1:
                    save_dic[k][0][8] = random.choice(pattern_total_text) + " " +  gen_total(fake)
                if len(v)==2:
                    if hasCharacters(save_dic[k][0][8]):
                        save_dic[k][0][8] = random.choice(pattern_total_text)
                    if hasNumbers(save_dic[k][0][8]):
                        save_dic[k][0][8] = gen_total(fake)  
        for k,v in save_dic.items():
            for item in v:
                results.append(item)
        with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/data_generate/train_{i}.txt","w") as f:
            for line in results :
                f.write("\t".join([str(i) for i in line]) + "\n")