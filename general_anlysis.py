import PDF_save
from paddleocr import PaddleOCR
import re
import torch        #是下载torch而不是pytorch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.optim.lr_scheduler import StepLR
import multiprocessing
from datetime import datetime

#pdf转图片
def extract_text_allpage(pdf_path,img_folder):
    imgs_path = PDF_save.pdf_to_images(pdf_path,img_folder)
    print("pdf转换成功!!,存储路径为{}".format(imgs_path))
    return imgs_path

#预测标签
def predict_label(sentence, model, tokenizer):
    # 对输入句子进行编码
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()

    # 在模型中进行前向传播
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits

    # 获取预测的标签
    _, predicted_label = torch.max(logits, dim=1)

    return predicted_label.item()

#提取日期并转换成对应格式
def extract_date(text):
    patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',      #2012年12月12日
        r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', #2012-12-12
        r'(\d{4})(\d{1,2})(\d{1,2})'            #202110412
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            # 根据不同的日期格式尝试转换为datetime对象
            try:
                year, month, day = map(int, groups)
                date_obj = datetime(year, month, day)
                return date_obj
            except ValueError:
                return None  # 若无法转换，则忽略并尝试下一个格式

# ocr识别
def ocr(img_path):
    content = []
    location = []
    sorce = 0
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            if isinstance(line, list):
                location.append(line)
            else:
                content.append(line[0])
                if line[1]<0.85:
                    sorce+=1
    return content,location,sorce

#图片转文字
def img2word(img_paths):
    if type(img_paths) == str:               #格式为jpg的
        content,location,sorce = ocr(img_paths)
    else:                                   #格式为pdf的
        for img_path in img_paths:
            content,location,sorce = ocr(img_path)
    if (len(content)) == 0:
        print("未识别出内容,建议转人工查看！！！")
        content = ["1"]                     #这个占位置的，不用管

    return content,location,sorce/(len(content))

#检测项目是否做全
def is_enough(sentence,queue):
    flag = "是"
    info = [["视力","左眼","右眼","五官","视觉","色觉","辨色力"],
            ["耳","鼓膜","听力","五官"],
            ["口","扁桃体","咽喉","五官","牙齿"],        #口鼻不是特别关注的,但是利用正则不是特别好，以爱康国宾为例，它就算没做也会出现这个
            ["鼻","五官"],
            ["胸透","胸部","胸片","回声均匀","肺门"],          #双肺纹理清晰，这个可能会存在于内科
            "心电图",                       #不能写心率，那个可能是内科
        ["谷丙转氨酶","谷丙","ALT","血清丙氨酸氨基转移酶"],
        ["谷草转氨酶","谷草","AST","天冬氨酸氨基转移酶"],
        ["谷氨酰基转移酶","GGT","谷氨酰","rGT"],
        ["内科","肝","脾","肾"],                        #外科，内科只能通过找共性，看哪些可以作为匹配项
        ["外科","淋巴结","脊柱","甲状腺","皮肤"]]


    txt = "".join(sentence)
    eyes_pattern = "|".join(info[0])
    ear_pattern = "|".join(info[1])
    mouth_pattern = "|".join(info[2])
    nose_pattern = "|".join(info[3])
    Xray_pattern = "|".join(info[4])
    ECG_pattern = "".join(info[5])
    ALT_pattern = "|".join(info[6])
    AST_pattern = "|".join(info[7])
    GGT_pattern = "|".join(info[8])
    surgery_pattern = "|".join(info[9])
    internal_pattern = "|".join(info[10])
    patterns = [eyes_pattern,ear_pattern,mouth_pattern,nose_pattern,Xray_pattern,ECG_pattern,ALT_pattern,AST_pattern,GGT_pattern,surgery_pattern,internal_pattern]
    for p in patterns:
        matches = re.findall(p, txt)
        if not matches:
            flag = "否"
            question = p.split("|")[0]
            print(f"{question}项目没有做")
        # else:print(matches)
    if not flag: print("体检项目齐全")
    queue.put((4,flag))
    # return flag

#检测是否有异常
def is_worry(sentences,queue):
    flag = "否"
    # 加载预训练的BERT模型和tokenizer
    label_cnt = 2               #修改标签
    model_name = r'chinese-roberta-wwm-ext-lagre'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config.num_labels = label_cnt   # 将 num_labels 添加到 config 中
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    # # 加载训练好的模型参数，后面那个参数不加好像会报错
    model.load_state_dict(torch.load(r'doctor_v4.pt'), strict=False)

    # 设置模型为评估模式
    model.eval()

    # 测试例子
    for input_sentence in sentences:
        predicted_label = predict_label(input_sentence, model, tokenizer)
        if predicted_label ==1:
            flag = "是"
            print(f'异常情况: {input_sentence}')
    if not flag:print("体检报告无异常")
    queue.put((5,flag))
    # return flag

#提取基础信息【先利用分类器把潜在对象提出来，再通过hanlp命名实体识别或者正则解决问题】
def research(sentences,queue):
    data = r'((19\d{2}|200[0-7])[-./年\s]((1[0-2])|(0?[1-9]))[-./月\s](([12]\d)|(3[01])|(0?[1-9]))日?)|(19\d{2}|200[0-7])[-./年\s]((1[0-2])|(0?[1-9]))[-./月\s]?'
    locs = ['医院', '机构', '院',"部","中心","区","集团","门诊"]
    # 加载预训练的BERT模型和tokenizer
    label_cnt = 4               #修改标签
    model_name = r'chinese-roberta-wwm-ext-lagre'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config.num_labels = label_cnt   # 将 num_labels 添加到 config 中
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    # # 加载训练好的模型参数，后面那个参数不加好像会报错
    model.load_state_dict(torch.load(r'info.pt'), strict=False)

    # 设置模型为评估模式
    model.eval()

    # 测试例子
    TJ_name = []
    TJ_loc = []
    TJ_data = ""
    for input_sentence in sentences:
        predicted_label = predict_label(input_sentence, model, tokenizer)
        if predicted_label ==1 and len(input_sentence)>=3:      #至少要有“姓名：”
            TJ_name.append(input_sentence)
        elif predicted_label ==2 and any(input_sentence.endswith(e) for e in locs):     
            TJ_loc.append(input_sentence)
        elif predicted_label ==3 and not TJ_data:       #有过答案就不要再继续下去了
            D = extract_date(input_sentence)
            if not D :continue
            current_year = datetime.now().year
            if D.year<=current_year:        #防止ocr识别错误2023识别2029
                diff = datetime.now() - D
                TJ_data="超出有效期" if diff.days>93 else "体检日期在有效期内"      #默认一个月算31天
                    
    print(f"检索到的姓名有{TJ_name}\n检索到的机构有{TJ_loc}\n检索到的日期有{TJ_data}")
    queue.put((1,TJ_name[0])) if TJ_name else queue.put((1,"无法识别"))
    queue.put((2,TJ_loc[0])) if TJ_loc else queue.put((2,"无法识别"))
    queue.put((3,TJ_data)) if TJ_data else queue.put((3,"无法识别"))          
    #后期这几个都可以考虑加个判断只拿第一个，不要设置成列表，这样可以节省时间

#主程序
def main(imgs_path):
    content,location,flag = img2word(imgs_path)
    if flag > 0.1:
        print(f"##############################该简历质量不高,建议转人工处理！识别率仅为{1-flag}###################################")
    print(content)
        
    # 创建多个子进程
    processes = []
    queue = multiprocessing.Queue()         #多进程之间通信只能利用他提供的共享数据结构进行通信
    funs = [research,is_enough,is_worry]
    
    # 开始执行各个进程
    for f in funs:
        p0 = multiprocessing.Process(target=f, args=(content,queue)) #后面这个逗号不能省的
        processes.append(p0)
        p0.start()

    # 等待各个进程执行完毕
    for p in processes:
        p.join()

    # 按指定格式进行输出
    basic_info = []               
    while not queue.empty():
        task_id,res = queue.get()
        basic_info.append((task_id,res))
    basic_info.sort(key=lambda x: x[0])         #根据任务编号排序
    print(f"最终结果：\n姓名:{basic_info[0][1]}\t体检机构:{basic_info[1][1]}\t是否在3个月内:{basic_info[2][1]}\n项目是否做全:{basic_info[3][1]}\t项目是否异常:{basic_info[4][1]}")


if __name__=="__main__":
    pdf_path = r"C:\Users\Administrator\Desktop\体检报告\二期\10026937-TJ-02.jpg"
    # pdf_path = r"C:\Users\Administrator\Desktop\体检报告\二期\体检\10027065-TJ-01xy.pdf"
    if pdf_path.endswith(".pdf"):           
        imgs_path = extract_text_allpage(pdf_path,"img")            #得到是简历解析成图像的路径，所以不用担心会多读
        main(imgs_path)
    elif pdf_path.endswith(".jpg") or pdf_path.endswith(".png"):
        imgs_path = pdf_path
        main(imgs_path)
    else:
        print("不支持的文件类型！")
