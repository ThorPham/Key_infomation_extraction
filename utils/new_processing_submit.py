import numpy as np
import glob
import os
import pandas as pd
import datefinder
import cv2

def correct(text,field="total"):
    address_pattern= add_patterns = {
"CAM CARA UC": "",
"Thuận 104": "104",
"Bi ,h Duong": "Binh Duong",
"THỊ CẦN BÌNH": "CAM BINH",
"CAN PHA": "CAM PHA",
"CẦN PHÁ": "CAM PHA",
"Can Pha Quan": "CAM PHA QN",
"CAM PHA QUANG MINH": "CAM PHA QUANG NINH",
"CAN HIA QUNG NINH": "CAM PHA QUANG NINH",
"CAN PHA QUANG NINH": "CAM PHA QUANG NINH",
"CAN PHA QUNG NINH": "CAM PHA QUANG NINH",
"PHA QUANG MINH": "CAM PHA QUANG NINH",
"UA MA QM, NH": "CAM PHA QUANG NINH",
"NUM DRUG NG": "CAM TRUNG",
"THE CO.OP": "CO.OP",
"Cam Dinh Can Pha Quan": "Cam Binh Cam Pha Quang Ninh",
"Can Ha Ung Hinh": "Cam Pha Quang Ninh",
"Co Con Ma Quoc Ninh": "Cam Pha Quang Ninh",
"Chiến Tháng": "Chiến Thắng",
"Thu Chợ": "Chợ",
"Thuy Chợ": "Chợ",
"Thế Chợ": "Chợ",
"Trong Chợ": "Chợ",
"Và Chợ": "Chợ",
"Chọ Sài": "Chợ Sủi",
"Chọ Sĩ": "Chợ Sủi",
"Chợ Sùi": "Chợ Sủi",
"Chợ Súi": "Chợ Sủi",
"Chợ Sũi": "Chợ Sủi",
"Chợ Sửi": "Chợ Sủi",
"Thuy Chợ Sũi": "Chợ Sủi",
"Thuy Chợ Sủi": "Chợ Sủi",
"Co. op": "Co.op",
"Co.0p": "Co.op",
"The Ty": "Cong ty",
"The Ty TNHH TMOT-DV": "Cty TNHH TMQT-DV",
"THỊ CẦN BÌNHCẦN PHÁ": "CẨM BÌNH CẨM PHẢ",
"CẢ THỊ": "CẨM THỊ",
"CÂN TRUNG": "CẨM TRUNG",
"CẦM TRUNG": "CẨM TRUNG",
"CẦN TRUNG": "CẨM TRUNG",
"A Cảm Bình": "Cẩm Bình",
"Câm Bình": "Cẩm Bình",
"Cảm Bình": "Cẩm Bình",
"Cấm Bình": "Cẩm Bình",
"Cấm Đình": "Cẩm Bình",
"Cầm Binh": "Cẩm Bình",
"Cầm Bình": "Cẩm Bình",
"Cẩm Binh": "Cẩm Bình",
"CMMONG": "Cẩm Phả",
"Càm Phà": "Cẩm Phả",
"Càm Phá": "Cẩm Phả",
"Càm Phả": "Cẩm Phả",
"Cám Phà": "Cẩm Phả",
"Cát Phá": "Cẩm Phả",
"Câm Pha": "Cẩm Phả",
"Câm Phà": "Cẩm Phả",
"Câm Phá": "Cẩm Phả",
"Câm Phả": "Cẩm Phả",
"Căm Thuận": "Cẩm Phả",
"Cảm Pha": "Cẩm Phả",
"Cảm Phà": "Cẩm Phả",
"Cảm Phá": "Cẩm Phả",
"Cảm Phả": "Cẩm Phả",
"Cấm Phà": "Cẩm Phả",
"Cấm Phả": "Cẩm Phả",
"Cấm Thế": "Cẩm Phả",
"Cầm Pha": "Cẩm Phả",
"Cầm Phà": "Cẩm Phả",
"Cầm Phá": "Cẩm Phả",
"Cầm Phú": "Cẩm Phả",
"Cầm Phả": "Cẩm Phả",
"Cầm Phủ": "Cẩm Phả",
"Cẩm Pha": "Cẩm Phả",
"Cẩm Phà": "Cẩm Phả",
"Cẩm Phá": "Cẩm Phả",
"Cẩm Phả": "Cẩm Phả",
"Cẩm Phả Nội": "Cẩm Phả",
"Cẩm Phải": "Cẩm Phả",
"Cẩm Phả, Quảng Hình": "Cẩm Phả, Quảng Ninh",
"Cẩm Phả, Quảng Nam": "Cẩm Phả, Quảng Ninh",
"Cẩm Phả, T. Oàng Mrà": "Cẩm Phả, T. Quảng Ninh",
"Cẩm Phả, T. Quang Ninh": "Cẩm Phả, T. Quảng Ninh",
"Càm Son": "Cẩm Sơn",
"Càm Sơn": "Cẩm Sơn",
"Câm Sơn": "Cẩm Sơn",
"Câm Sơn Thu": "Cẩm Sơn",
"Cảm Sơn": "Cẩm Sơn",
"Cầm Sơn": "Cẩm Sơn",
"câm sơn": "Cẩm Sơn",
"Cầm Thành": "Cẩm Thành",
"Còm Thuy": "Cẩm Thủy",
"Cảm Thúy": "Cẩm Thủy",
"Cấm Thủy Nam": "Cẩm Thủy",
"Cầm Thuy": "Cẩm Thủy",
"Cầm Thủy": "Cẩm Thủy",
"Của TVị": "Cẩm Thủy",
"Gầm Thuy": "Cẩm Thủy",
"Càm Trung": "Cẩm Trung",
"Cảm Trung": "Cẩm Trung",
"Cầm Trung": "Cẩm Trung",
"Cầm Trưng": "Cẩm Trung",
"Cần Trung": "Cẩm Trung",
"Cần Trưng": "Cẩm Trung",
"TCàm Trung": "Cẩm Trung",
"Câm Tây": "Cẩm Tây",
"Căm Tày": "Cẩm Tây",
"Căm Tây": "Cẩm Tây",
"Cảm Tây": "Cẩm Tây",
"Cấm Tâ": "Cẩm Tây",
"Cấm Tây": "Cẩm Tây",
"Cầm Tây": "Cẩm Tây",
"Cẩm Tây": "Cẩm Tây",
"Cẩm Tâyy": "Cẩm Tây",
"Câm Đông": "Cẩm Đông",
"Cấm Đồng": "Cẩm Đông",
"Thị DA khu": "DA Khu",
"The Du an": "Du an",
"Đầu Dầu tuy trong p. dân có": "Du an Dau tu xay dung Khu nha o",
"THUY DỰ ÁN KHU DÂN CƯ LẦN BIÊN ĐỘ": "DỰ ÁN KHU DÂN CƯ LẤN BIỂN",
"GHa Làm Chí": "Gia Lâm",
"Gia Lam": "Gia Lâm",
"Gia Làm": "Gia Lâm",
"Gia Làm Đó": "Gia Lâm",
"Gia Làm Đại": "Gia Lâm",
"Gia Làm Đội": "Gia Lâm",
"Gia Lâm": "Gia Lâm",
"Gia Lâm Chính": "Gia Lâm",
"Gia Lâm Thị": "Gia Lâm",
"Gin Lâm": "Gia Lâm",
"Gò Vấp Nam": "Gò Vấp",
"H-CT2 thuoc thanh": "H-CT2 thuoc",
"H-CT2 thuoc thu": "H-CT2 thuoc",
"H-CT2 thuoc tra": "H-CT2 thuoc",
"H-cT2 thuoc": "H-CT2 thuoc",
"H-cT2 thuoc thu": "H-CT2 thuoc",
"H-cT2 thuoc tranh": "H-CT2 thuoc",
"H-cr2 thuoc": "H-CT2 thuoc",
"Ha Noi Noi": "Ha Noi",
"HÀ ĐỒNG": "HÀ ĐÔNG",
"Hà Nội Nội": "Hà Nội",
"Hồ Chí": "Hồ Chí Minh",
"Hồ Chí Minh Minh": "Hồ Chí Minh",
"KHU MINH TIÊN A": "KHU MINH TIẾN A",
"Km 18": "Khu 18",
"Khu Diêm Thúy": "Khu Diêm Thủy",
"KM NINH TIÊM A.P.CA": "Khu Minh Tiến A",
"NHU MINH TIÊN A": "Khu Minh Tiến A",
"Ky (uc": "Ky tuc",
"Lãng Hạ": "Láng Hạ",
"Làng Trung": "Láng Trung",
"Làm Chí": "Lâm",
"Lâm Thị": "Lâm",
"Lâm Đó": "Lâm",
"IẤN BIÊN": "LẤN BIỂN",
"LẦN BIÊN": "LẤN BIỂN",
"LẦN BIÊN CỌC": "LẤN BIỂN CỌC",
"NINH 11ĐN A": "MINH TIẾN A",
"NINH TIÊM A": "MINH TIẾN A",
"Hinh Tiến": "Minh Tiến",
"Hình Tiến": "Minh Tiến",
"Hinh Tiến A": "Minh Tiến A",
"Hình Tiến A": "Minh Tiến A",
"Minh Tiên A": "Minh Tiến A",
"Minh Tiên Á": "Minh Tiến A",
"Minh Tiên Ả": "Minh Tiến A",
"Minh Tiến A 1": "Minh Tiến A",
"Minh Tiến A Nam": "Minh Tiến A",
"Minh Tiến Á": "Minh Tiến A",
"Minh Tiến Ả": "Minh Tiến A",
"Minh Tiền A": "Minh Tiến A",
"Minh Tiền Á": "Minh Tiến A",
"Minh Tiền Ả": "Minh Tiến A",
"Minh Tiền Ả Nam": "Minh Tiến A",
"Mình Tiến A": "Minh Tiến A",
"Theo VM+Tiến A Nam": "Minh Tiến A",
"Mề Trì": "Mễ Trì",
"My Đình": "Mỹ Đình",
"Mỹ Đinh": "Mỹ Đình",
"Mỹ Đinh 1": "Mỹ Đình",
"Mỹ Đỉnh": "Mỹ Đình",
"Mỹ Đỉnh 1": "Mỹ Đình",
"Nguyễn Kiêm": "Nguyễn Kiệm",
"P.CẢ": "P. CẨM THỊ",
"P. Cảm Thúy": "P. Cẩm Thủy",
"P.Cấm Thủy Nam": "P.Cẩm Thủy",
"P.Cẩm Thủy -": "P.Cẩm Thủy",
"p. cẩm thủy": "P.Cẩm Thủy",
"p.cấm thủy": "P.Cẩm Thủy",
"p.cẩm thủy": "P.Cẩm Thủy",
"Phu Lay Q. Ha Dong": "Phu La Q. Ha Dong",
"Phu Lay Qu Ha Dong": "Phu La Q. Ha Dong",
"Phu 4": "Phu,",
"Phu, Ho": "Phu,",
"Phu, Tan": "Phu,",
"Phu Thị": "Phú Thị",
"Phù Thị": "Phú Thị",
"Phú Thuy": "Phú Thụy",
"Phủ Thuy": "Phú Thụy",
"Phả - Quang": "Phả - Quảng",
"Phá - Quảng Ninh": "Phả - Quảng Ninh",
"Phả, QIH": "Phả, QN",
"Phả, QIM": "Phả, QN",
"Phả, QNH": "Phả, QN",
"Phả, T. GN": "Phả, T. Quảng Ninh",
"Phà, QNH": "Phả. QN",
"Phổ Sui": "Phố Sủi",
"Phổ Súi": "Phố Sủi",
"0. Ha Dong": "Q. Ha Dong",
"0. T, Dong": "Q. Ha Dong",
"9, Ha Dong": "Q. Ha Dong",
"Qu Ha Dong": "Q. Ha Dong",
"Đ. Hà Dong": "Q. Ha Dong",
"0NH": "QNH",
"ONH": "QNH",
"QUNG NINH": "QUANG NINH",
"LUOC GIA NGUYỄN": "QUOC GIA",
"Quáng Ninh": "Quảng Ninh",
"Quảng Hình": "Quảng Ninh",
"Quảng Ninh 1": "Quảng Ninh",
"Quảng Ninh Trị": "Quảng Ninh",
"Số Số": "Số",
"Thuận Sơ": "Số",
"Thành Số": "Số",
"Thế Số": "Số",
"Sò 5": "Số 5",
"T. GN": "T. Quang Ninh",
"T. Oàng Mrà": "T. Quảng Ninh",
"T. Quang Ninh": "T. Quảng Ninh",
"T. Quàng Ninh": "T. Quảng Ninh",
"THƯƠNG . MẠI": "THƯƠNG MẠI",
"TỪ 8 KM)": "TO 8 KHU",
"TU 6 M Br CM TING": "TO 8 Khu 3B P CAM TRUNG",
"TP.CM Hu, UH": "TP. QUANG NINH",
"TRUNG, 1%M": "TRUNG TAM",
"TRÊN PHÚ": "TRẦN PHÚ",
"TRẦN NŨ": "TRẦN PHÚ",
"THANH NA": "Thanh Niên",
"Thanh Hiên": "Thanh Niên",
"Thanh Hiện": "Thanh Niên",
"Thành phố Căm": "Thành Phố Cẩm",
"Thành phố Cấm": "Thành Phố Cẩm",
"Thón": "Thôn",
"Xã Thôn": "Thôn",
"Thôn Phú Thuy": "Thôn Phú Thụy",
"Tiên A": "Tiến A",
"Tiến A Nam": "Tiến A",
"Tiến Ả": "Tiến A",
"Tiền A": "Tiến A",
"Tiền Á": "Tiến A",
"Tiền Ả": "Tiến A",
"10 7 NINH TIÊM A": "To 7 MINH TIEN A",
"TÙ 7 NINH 11ĐN A": "To 7 MINH TIEN A",
"SỰ TIẾP A": "Trần Phú",
"Tran Phú": "Trần Phú",
"Trân Phú": "Trần Phú",
"Trăn Phú": "Trần Phú",
"Trấn Phú": "Trần Phú",
"Trần Phú Trị": "Trần Phú",
"Trần Phố": "Trần Phú",
"Tvo Km P Cua": "Trần Phú",
", 1%M,": "TÂM",
"TâyThành": "Tây Thành",
"TÙ 7": "TỔ 7",
"Xã Tô": "Tổ",
"Tô 3": "Tổ 3",
"T6 7": "Tổ 7",
"T67": "Tổ 7",
"T67.": "Tổ 7",
"Tê 7": "Tổ 7",
"Tò 7": "Tổ 7",
"Tô 7": "Tổ 7",
"Tô7": "Tổ 7",
"TỐ 7": "Tổ 7",
"Tố 7": "Tổ 7",
"Tố7": "Tổ 7",
"Tộ 7": "Tổ 7",
"T61, Kho Minh Tien A": "Tổ 7, Khu Minh Tiến A",
"THỊ TÚ 8": "Tổ 8",
"THEO TO Ở KHU 3B": "Tổ 8 Khu 3B",
"THỊ TÚ 8P.CÂN TRUNG,1": "Tổ 8 P.CẨM TRUNG",
"VIA": "VM+",
"VIA%": "VM+",
"VIA4": "VM+",
"VIA?": "VM+",
"VIM": "VM+",
"VIM%": "VM+",
"VIM4": "VM+",
"VIM?": "VM+",
"VIN%": "VM+",
"VM%": "VM+",
"VM4": "VM+",
"VM?": "VM+",
"VMẻ": "VM+",
"VN%": "VM+",
"Và VM+": "VM+",
"Trong VINH QNH": "VM+ QNH",
"VIM4 ANH": "VM+ QNH",
"VINH ONH": "VM+ QNH",
"VINH QNH": "VM+ QNH",
"VM OM": "VM+ QNH",
"VM% ONH": "VM+ QNH",
"Và Vài% 17TM": "VM+ QNH",
"Van Phuy Nam": "Van Phu",
"Van Phuyen": "Van Phu",
"Van Phu ;": "Van Phu,",
"Van Phu y": "Van Phu,",
"Van Phus": "Van Phu,",
"Van Phuy": "Van Phu,",
"van phu vo": "Van Phu,",
"Xà Phú Thị": "Xã Phú Thị",
"cấm thủy": "cẩm thủy",
"căm tây": "cẩm tây",
"coc 6": "cọc 6",
"do thi noi": "do thi moi",
"dọc tuyên": "dọc tuyến",
"thanh ket hop": "ket hop",
"the ket hop": "ket hop",
"thu ket hop": "ket hop",
"khu tân lập 4": "khu Tân Lập 4",
"lần biên": "lấn biển",
"lần biến": "lấn biển",
"lần biển": "lấn biển",
"lấn biên cọc": "lấn biển cọc",
"lần biên coc": "lấn biển cọc",
"lần biên cọc": "lấn biển cọc",
"lần biến cọc": "lấn biển cọc",
"lần biển cọc": "lấn biển cọc",
"phường căm tây": "phường Cẩm",
"Tai tai": "tai",
"tr3, Long (H": "tren o dat H",
"tr3, Long (H-CT2": "tren o dat H-CT2",
"Đó Số": "Đc Số",
"Đo:": "Đc:",
"Đinh Thôn": "Đình Thôn",
"ĐƯƠNG3/9/39": "ĐƯỜNG 3/9/39",
"Dương Trần Phú": "Đường Trần Phú",
"BẠI LÝ": "ĐẠI LÝ",
"ĐAI LÝ": "ĐẠI LÝ",
"Dài Lý": "Đại lý",
"ĐỊA ONỈ": "ĐỊA CHỈ",
"Địa Qiít": "ĐỊA CHỈ",
"ĐỊA CHÍ BÀI LÝ": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ BẠI LÝ 3": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ MÌ LƯ": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ ĐAI LÝ": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ ĐAI LÝ 1": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ ĐÀI LY": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA CHỈ ĐẠI LY": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA OHÍ MI LY:": "ĐỊA CHỈ ĐẠI LÝ",
"ĐỊA ONỈ ĐẠI LÝ": "ĐỊA CHỈ ĐẠI LÝ",
"Địa Chỉ Mi Lý 1": "ĐỊA CHỈ ĐẠI LÝ",
"Địa chỉ bài VL": "ĐỊA CHỈ ĐẠI LÝ",
"Trong Dịa": "Địa",
"Trong Địa": "Địa",
"Và Địa": "Địa",
"Dịa Chỉ": "Địa chỉ",
"Đồng Đa": "Đống Đa",
"ĐƯU ĐÀM": "ƯU ĐÀM",
}
    correct_patterns = {
        'TIÊN': 'TIỀN',
        'TTỔNG': 'TỔNG',
        'Tiên': 'Tiền',
        'TIẾN': 'TIỀN',
        '11ỀN': 'TIỀN',
        'TSTOÁN': 'T. TOÁN',
        'tiên': 'tiền',
        'toc': 'toán',
        'QUẤY': 'QUẦY',
        'CŨNG': 'CỘNG',
        'Tiến': 'Tiền',
        'OÁN': 'TOÁN',
        'TĂNG': 'TỔNG',
        'TOÁNG': 'TOÁN',
        'TỜNG': 'TỔNG',
        'TÔNG': 'TỔNG',
        'toàn': 'toán',
        'QUÁY': 'QUẦY',
        'T,': 'T.',
        'TOÀN': 'TOÁN',
        'Tpán': 'Toán',
        'CÔNG': 'CỘNG',
        'Ống': 'Tổng',
        'Cộng': 'Cộng',
        'tiến': 'tiền',
        'toan(VND)': 'toan (VND)',
        'stong': 'tong',
        'CŨNG:': 'CỘNG:',
        "TŨNG" :"TỔNG",
        # '-': '',
        '1': '',
        'VÀ': '',
        'tiên:': 'tiền:',
        'QUẤY':'QUẦY',
        '"TỔNG': 'TỔNG',
        'TOÁN"': 'TOÁN'
    }

    date_patterns = {
        'bản': 'bán',
        'Ngayr': 'Ngay:',
        'họn': 'hẹn',
        'bản:': 'bán:',
        'Bản:': 'Bán:',
        'iểm': 'Điểm',

    }

    new_line = text.strip()
    if field =="total":
        for word in text.strip().split(' '):
            if word in correct_patterns: # date_patterns for date 
                new_line = new_line.replace(word, correct_patterns.get(word)) # date_patterns for date 
    if field =="date":
        for word in text.strip().split(' '):
            if word in date_patterns: # date_patterns for date 
                new_line = new_line.replace(word, date_patterns.get(word)) # date_patterns for date 
    if field == "address":
        for word in text.strip().split(' '):
            if word in address_pattern: # date_patterns for date 
                new_line = new_line.replace(word, address_pattern.get(word)) 
        
    return new_line

paths = glob.glob("rs1/*")
node_labels = ['other', 'company', 'address', 'date', 'total']
def hasNumbers(inputString):
    inputString = inputString.replace("Đ","")
    return any(char.isdigit() for char in inputString)

def hasCharacters(inputString):
    return any(char.isalpha() for char in inputString)
color = (0, 0, 255)

data_frame = []
count = 0
for path in paths :
    results = {"company":[],"address":[],"date":[],"total":[]}
    outputs   = {"company":"","address":"","date":"","total":""}
    name = os.path.basename(path)
    name_id = name.replace("txt","jpg")
#     image = cv2.imread(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/data_test/{name_id}")
#     image_show = image.copy()
    with open(path,"r") as file :
        data = file.readlines()
        for line in data:
            tmp = line.strip().split("\t")
            x,y = tmp[:2]
            ymax = tmp[5]
            text = tmp[10]
            class_id = tmp[11]
            save = [int(x),int(y),text,class_id,ymax]
            results[class_id].append(save)      
        for key,v in results.items():       
            if key == "date" :
                for j in results[key]:
                    if "Ngày" in j[2] :
                        j[2] =  "Ngày" + j[2].split("Ngày")[1]
                    if "Thời" in j[2] :
                        j[2] =  "Thời" + j[2].split("Thời")[1]
                if len(results[key])==0:
                    with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/OCR_spelling_correction/{name}","r") as f :
                        file = f.readlines()
                        for i in file :
                            x1,y1 = i.strip().split("\t")[:2]
                            tm = i.strip().split("\t")[8]
                            tm2 = tm.split(" ")
                            for k in tm2 :
                                if len(k)<10 and (k.count("-") <2 or k.count("/")<2  or k.count(".")<2):
                                    continue
                                else :
                                    try  :
                                        matches = datefinder.find_dates(k)
                                        for match in matches:
#                                             print(tm)           
                                            if "Ngày" in tm:
                                                
                                                results[key].append([int(x1),int(y1),"Ngày" + tm.split("Ngày")[1],"date",ymax])
            
#                                                 break
                                            elif "Thời" in tm :
                                                results[key].append([int(x1),int(y1),"Thời" + tm.split("Thời")[1],"date",ymax])
#                                                 break
                                            else :
                                                results[key].append([int(x1),int(y1),tm,"date",ymax])
#                                                 break
                                    except :
                                        continue
            
            
            
        if len(results["total"]) == 1 :  
            if (hasNumbers(results["total"][0][2])==False or hasCharacters(results["total"][0][2])==False) :
                box1 = results["total"][0]
                ymin_1,ymax_1 = int(box1[1]),int(box1[-1])
                yc_c1 = (ymax_1 + ymin_1)/2
                with open(f"/media/thorpham/PROJECT/OCR-challenge/preprocessing/data_for_submit/private_test/OCR_spelling_correction/{name}","r") as f :
                    file = f.readlines()
                    same_line = []
                    for i in file :
                        bboxs = i.strip().split("\t")
                        label_line = bboxs[8]
                        ymin_2,ymax_2 = int(bboxs[1]),int(bboxs[5])
                        yc_c2 = (ymax_2 + ymin_2)/2
                        if abs(yc_c1-yc_c2) < 10 and (label_line.lower() not in results["total"][0][2].lower()):
#                             results["total"].append([int(bboxs[0]),ymin_2,label_line,ymax_2])
                            same_line.append([int(bboxs[0]),ymin_2,label_line,ymax_2])
                    if len(same_line)==1:
                            results["total"].append(same_line[0])                           
    #sorted(my_list , key=lambda k: [k[1], k[0]])
    company = "|||".join([str(i[2]) for i in sorted(results["company"] , key=lambda k: k[1])]) 
    adress = "|||".join([correct(str(i[2]),field="address") for i in sorted(results["address"] , key=lambda k: k[1])])
#     print(results["date"])
    date = "|||".join([correct(str(i[2]),field="date") for i in sorted(results["date"] , key=lambda k: k[0])])
#     print("--->",date)
    total = "|||".join([correct(str(i[2]),field="total") for i in sorted(results["total"] , key=lambda k: k[0])])
    labels_save =  company+"|||"+ adress + "|||" + date + "|||" + total
    print("="*20,name_id,"="*20)
    print(labels_save)
    print("-"*50)
    data_frame.append([name_id,0.5,labels_save])
# data_frame.append(["mcocr_val_145114unyae.jpg",0.5,"|||||||||"])
columns = ["img_id","anno_image_quality","anno_texts"]
df = pd.DataFrame(data_frame,columns=columns)
print(df.head())
print(len(df))
df.to_csv("results.csv",index=None,columns=columns)

df1 = pd.read_csv("mcocr_test_samples_df.csv")
df2 = pd.read_csv("results.csv")
df_final.to_csv("submit_final/results.csv",index=None)
