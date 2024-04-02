import sys
 
from ui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QLineEdit, QMainWindow, QFileDialog
 
 
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        # MainWindow.setFixedSize(MainWindow.width(), MainWindow.height()); 
        # self.auto_resize()          #自动改变大小


 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.setWindowTitle("SSC体检报告信息提取");                        #设置窗口标题
    myWin.show()
    sys.exit(app.exec_())
