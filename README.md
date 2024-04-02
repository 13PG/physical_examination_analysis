# 当前目录下文件（夹）作用：
名称  | 作用
 ---- | ----- 
data_collection.py        |          把对应文件夹下的pdf转成图片，再利用飞桨ocr把文字信息转成文字，并且标注成特定格式存放于data.txt中<br>
PDF_save.py               |          把pdf转成图片<br>
draw_pic.py               |          训练过程保存成图片<br>
train_doctor_model.py     |          训练异常情况提取模型<br>
train_info_model.py       |          训练基础信息提取模型<br>
general_anlysis.py        |          利用训练好的模型对体检报告进行检测。主程序<br>
img                       |          data_collection.py图片暂存文件夹,**你最好提前创建空的img文件夹**<br>
chinese-roberta-wwm-ext-lagre   |    bert模型【https://hf-mirror.com/hfl/chinese-roberta-wwm-ext-large/tree/main】<br>
ui.py                     |          检测界面源码
server.py                 |          启动ui界面


<br>
![示意图](https://img-blog.csdnimg.cn/direct/0ed36d13bac64cf9819d7fabc248e90b.png "体检检测界面")
<br>


# 暂未提供的文件
## 数据集文件【基本结构：每行前面是一个文本后面紧跟着&数字，数字由你自己设定】
数据集名称  | 类别比例\样本数量  | 训练后的模型
 ---- | ----- | ------  
data_v1  | 第一版数据，样本比例：  2:1  （1500） | doctor.pt
data_v2  | 第二版数据，样本比例：  5:1  (3063) | doctor_v1.pt,doctor_v2.pt
data_v3  | 暂无 |     info.pt
## 已经训练好的模型文件
### 识别异常情况的模型：0正常,1异常<br>
* doctor.pt                   在data_v1的基础上采用交叉熵损失函数<br>
* doctor_v1.pt                在data_v2的基础上采用focal损失函数<br>
* doctor_v2.pt                在data_v2的基础上采用交叉熵损失函数[基本上够用了]<br>
### 识别基础信息的模型：0垃圾,1姓名,2体检机构,3体检日期<br>
* info.pt<br>
