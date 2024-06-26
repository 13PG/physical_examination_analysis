# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import general_anlysis
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QLineEdit, QMainWindow, QFileDialog,QMessageBox
import multiprocessing


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(687, 493)       #固定窗口大小   

        # 补充
        self.doctor_path = r"doctor_v2.pt"
        self.info_path = r"info.pt"
        self.file_path = ""
        #


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(210, 150, 411, 291))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(0, QtWidgets.QFormLayout.LabelRole, spacerItem)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.show_name = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.show_name.setObjectName("show_name")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.show_name)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.show_location = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.show_location.setObjectName("show_location")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.show_location)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.show_date = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.show_date.setObjectName("show_date")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.show_date)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.in_enough = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.in_enough.setObjectName("in_enough")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.in_enough)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.is_worry = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.is_worry.setObjectName("is_worry")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.is_worry)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(7, QtWidgets.QFormLayout.LabelRole, spacerItem1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(7, QtWidgets.QFormLayout.FieldRole, spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(0, QtWidgets.QFormLayout.FieldRole, spacerItem3)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.show_name_2 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.show_name_2.setObjectName("show_name_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.show_name_2)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 621, 111))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.show_info_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.show_info_path.setObjectName("show_info_path")
        self.gridLayout.addWidget(self.show_info_path, 1, 1, 1, 1)
        self.show_file_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.show_file_path.setObjectName("show_file_path")
        self.gridLayout.addWidget(self.show_file_path, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.show_doctor_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.show_doctor_path.setObjectName("show_doctor_path")
        self.gridLayout.addWidget(self.show_doctor_path, 0, 1, 1, 1)
        self.choose_model_doctor = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.choose_model_doctor.setObjectName("choose_model_doctor")
        self.gridLayout.addWidget(self.choose_model_doctor, 0, 2, 1, 1)
        self.choose_model_info = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.choose_model_info.setObjectName("choose_model_info")
        self.gridLayout.addWidget(self.choose_model_info, 1, 2, 1, 1)
        self.choose_file = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.choose_file.setObjectName("choose_file")
        self.gridLayout.addWidget(self.choose_file, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.go = QtWidgets.QPushButton(self.centralwidget)
        self.go.setGeometry(QtCore.QRect(20, 250, 131, 61))
        self.go.setObjectName("go")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 687, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #补充
        self.show_info_path.setText(self.info_path)
        self.show_doctor_path.setText(self.doctor_path)
        self.choose_model_doctor.clicked.connect(self.get_doctor_path)
        self.choose_model_info.clicked.connect(self.get_info_path)
        self.choose_file.clicked.connect(self.get_file_path)
        self.go.clicked.connect(self.processing_data)
        #
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "姓名："))
        self.label_5.setText(_translate("MainWindow", "体检机构："))
        self.label_6.setText(_translate("MainWindow", "体检日期："))
        self.label_7.setText(_translate("MainWindow", "项目是否做全："))
        self.label_8.setText(_translate("MainWindow", "项目是否有异常："))
        self.label_9.setText(_translate("MainWindow", "识别精确率"))
        self.label_2.setText(_translate("MainWindow", "选择基本信息提取模型："))
        self.choose_model_doctor.setText(_translate("MainWindow", "选择模型"))
        self.choose_model_info.setText(_translate("MainWindow", "选择模型"))
        self.choose_file.setText(_translate("MainWindow", "选择文件"))
        self.label_3.setText(_translate("MainWindow", "选择要检测的体检报告："))
        self.label.setText(_translate("MainWindow", "选择异常识别模型："))
        self.go.setText(_translate("MainWindow", "开始测试"))

    # 获取文件路径
    def get_doctor_path(self):
        path1, _ = QFileDialog.getOpenFileName(self, "请选择文件", "", "All Files (*);;Text Files (*.txt)")
        if not path1:return  
        self.show_doctor_path.setText(path1)
        self.doctor_path = path1

    # 获取文件路径
    def get_info_path(self):
        path1, _ = QFileDialog.getOpenFileName(self, "请选择文件", "", "All Files (*);;Text Files (*.txt)")
        if not path1:return  
        self.show_info_path.setText(path1)
        self.info_path = path1
        
    # 获取文件路径
    def get_file_path(self):
        path1, _ = QFileDialog.getOpenFileName(self, "请选择文件", "", "All Files (*);;Text Files (*.txt)")
        if not path1:return  
        self.show_file_path.setText(path1)
        self.file_path = path1

    # 开启休眠提示框
    def show_message_box(self):
        #开启对话框
        self.msg_box_hint = QMessageBox()
        self.msg_box_hint.setIcon(QMessageBox.Information)
        self.msg_box_hint.setWindowTitle('Yahoo~')
        #标题自己设置
        self.msg_box_hint.setText('正在处理中，请稍后...\n出现未响应属于正常情况,请放心')
        self.msg_box_hint.show()
        QApplication.processEvents()

    # 关闭休眠提示框
    def close_message_box(self):
        #关闭对话框
        self.msg_box_hint.close()
    
    # 处理体检信息
    def processing_data(self):
        ##体检报告路径问题
        if not self.file_path.endswith(".pdf"):
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '体检报告格式不对,只支持pdf')
            msg_box.exec_()
            self.file_path=""
            self.show_file_path.setText(self.file_path)
            return 0
            
        ##模型路径问题
        if not self.info_path.endswith(".pt") or not self.doctor_path.endswith(".pt"):
            msg_box = QMessageBox(QMessageBox.Critical, '模型文件不匹配！', '若您找不到对应模型，建议关掉程序重新打开')
            msg_box.exec_()
            return 0
        
        # 开始干活！
        try:
            self.show_message_box()
            #
            ## 开始进行检测！！
            imgs_path = general_anlysis.extract_text_allpage(self.file_path,"img")            #得到是简历解析成图像的路径，所以不用担心会多读
            content,location,ocr_rate = general_anlysis.img2word(imgs_path)

            # 创建多个子进程    
            processes = []
            queue = multiprocessing.Queue()         #多进程之间通信只能利用他提供的共享数据结构进行通信
            funs = [general_anlysis.research,general_anlysis.is_enough,general_anlysis.is_worry]
            
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

            ## 输出结果保存
            self.show_name.setText(basic_info[0][1])
            self.show_name_2.setText(str(1-ocr_rate))
            self.show_location.setText(basic_info[1][1])
            self.show_date.setText(str(basic_info[2][1]))
            self.is_worry.setText(basic_info[4][1])
            self.in_enough.setText(basic_info[3][1])
            #
            self.close_message_box()
        except:
            msg_box = QMessageBox(QMessageBox.Critical, '异常错误', '未知的错误信息被发现')
            msg_box.exec_()
        finally:
            self.msg_box_finish.information(self, '成功！', '已完成！')
            
