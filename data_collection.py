from general_anlysis import extract_text_allpage,img2word
from paddleocr import PaddleOCR
import os

#获取要写入的文件
folder_path = r"C:\Users\Administrator\Desktop\体检报告\二期\体检" #pdf存放的路径
file_list = []
for file in os.listdir(folder_path):
    if not os.path.isdir(os.path.join(folder_path, file)):
        file_path = os.path.join(folder_path, file)
        file_list.append(file_path)



# 打开文件用于写入，如果文件不存在则创建
with open('./data_v1.txt', 'a+',encoding="utf-8") as file:
    for pdf_path in file_list:
        imgs_path = extract_text_allpage(pdf_path,"img")
        content,location,flag = img2word(imgs_path)
        file.write("#########"+pdf_path+'\n') # 添加换行符，先默认都是正常信息，因为本身体检正常的就高
        for line in content:
            if not line.isdigit() and len(line)>4: #首先去掉全是数字，和太短的名词等垃圾信息
                file.write(line+"&0" + '\n') # 添加换行符，先默认都是正常信息，因为本身体检正常的就高
                print("写入成功！")
