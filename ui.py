from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtOpenGL import QGLWidget
import sys
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MainWindow(QMainWindow):
    """docstring for Mainwindow"""

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.basic()
        self.obj1 = ""
        splitter_main = self.split_()
        self.setCentralWidget(splitter_main)

        # 窗口基础属性

    def basic(self):
        # 设置标题，大小，图标
        self.setWindowTitle("GT")
        self.resize(1100, 650)
        self.setWindowIcon(QIcon("./image/Gt.png"))
        # 居中显示
        screen = QDesktopWidget().geometry()
        self_size = self.geometry()
        self.move((screen.width() - self_size.width()) / 2, (screen.height() - self_size.height()) / 2)

    # 分割窗口
    def split_(self):

        # 左侧布局
        splitter = QSplitter(Qt.Vertical)

        # 左上obj文件
        frame = QFrame()
        vl = QVBoxLayout()
        vtkWidget = QVTKRenderWindowInteractor()
        vl.addWidget(vtkWidget)
        ren = vtk.vtkRenderer()
        vtkWidget.GetRenderWindow().AddRenderer(ren)
        self.iren = vtkWidget.GetRenderWindow().GetInteractor()
        self.CreateObj("result.obj", ren)
        frame.setLayout(vl)
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        splitter.addWidget(frame)

        # 左上选择文件按钮
        chooseFile1 = QPushButton("choose File")
        chooseFile1.clicked.connect(self.slot_btn_chooseFile)
        splitter.addWidget(chooseFile1)

        # 左下obj文件
        frame2 = QFrame()
        vl2 = QVBoxLayout()
        vtkWidget2 = QVTKRenderWindowInteractor()
        vl2.addWidget(vtkWidget2)
        ren2 = vtk.vtkRenderer()

        vtkWidget2.GetRenderWindow().AddRenderer(ren2)
        self.iren2 = vtkWidget2.GetRenderWindow().GetInteractor()
        self.CreateObj("output.obj", ren2)
        frame2.setLayout(vl2)
        renWin.AddRenderer(ren2)
        splitter.addWidget(frame2)

        # 左下选择文件按钮
        chooseFile2 = QPushButton("choose File")
        # chooseFile1.clicked.connect(self.getFile)
        # hbox.addWidget(chooseFile1)
        splitter.addWidget(chooseFile2)



        # 右侧布局
        splitter2 = QSplitter(Qt.Vertical)

        # 右上obj文件
        frame3 = QFrame()
        vl3 = QVBoxLayout()
        vtkWidget3 = QVTKRenderWindowInteractor()
        vl3.addWidget(vtkWidget3)
        ren3 = vtk.vtkRenderer()
        vtkWidget3.GetRenderWindow().AddRenderer(ren3)
        self.iren3 = vtkWidget3.GetRenderWindow().GetInteractor()
        self.CreateObj("output.obj", ren3)
        frame3.setLayout(vl3)
        renWin.AddRenderer(ren3)
        splitter2.addWidget(frame3)

        # 右上选择文件按钮
        chooseFile3 = QPushButton("choose File")
        chooseFile3.clicked.connect(self.getFile)
        splitter2.addWidget(chooseFile3)

        # 右下obj文件
        frame4 = QFrame()
        vl4 = QVBoxLayout()
        vtkWidget4 = QVTKRenderWindowInteractor()
        vl4.addWidget(vtkWidget4)
        ren4 = vtk.vtkRenderer()

        vtkWidget4.GetRenderWindow().AddRenderer(ren4)
        self.iren4 = vtkWidget4.GetRenderWindow().GetInteractor()
        self.CreateObj("output.obj", ren4)
        frame4.setLayout(vl4)
        renWin.AddRenderer(ren4)
        splitter2.addWidget(frame4)

        # 右下选择文件按钮
        chooseFile4 = QPushButton("Transfer")
        # chooseFile1.clicked.connect(self.getFile)
        # hbox.addWidget(chooseFile1)
        splitter2.addWidget(chooseFile4)

        splitter_main = QSplitter(Qt.Horizontal)
        # renWin.Render()
        splitter_main.addWidget(splitter)
        splitter_main.addWidget(splitter2)
        return splitter_main

    def CreateObj(self, filename, ren):
        # Create source
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        skinMapper = vtk.vtkPolyDataMapper()
        skinMapper.SetInputConnection(reader.GetOutputPort())
        skinMapper.ScalarVisibilityOff()
        skin = vtk.vtkActor()
        skin.SetMapper(skinMapper)
        ren.AddActor(skin)
        ren.SetBackground(255, 255, 255)

    def getFile(self):
        fname,_ =QFileDialog.getOpenFileName(self,'OpenFile',"c:/","Image files (*.obj *.jpg)")
        self.le.setPixmap(QPixmap(fname))

    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,'OpenFile',"c:/","Image files (*.obj)")

        if fileName_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件为:")
        print(fileName_choose)
        # self.obj1 = fileName_choose
        #
        # vtkWidget.GetRenderWindow().AddRenderer(ren)
        # self.iren = vtkWidget.GetRenderWindow().GetInteractor()
        #
        # self.CreateObj(fileName_choose, win.ren)
        # win.renWin.AddRenderer(win.ren)
        # # renWin.Render()
        # # win.show()
        # win.iren.Initialize()
        # self.win.update()
        # win.iren.Initialize()
        # print("文件筛选器类型: ", filetype)
        return fileName_choose

    def mousePressEvent(self,event): # 点击时记录点，并显示
        if event.buttons () == QtCore.Qt.LeftButton:
            print ("111")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    # ren = vtk.vtkRenderer()
    # renWin = vtk.vtkRenderWindow()
    win.show()
    win.iren.Initialize()
    win.iren2.Initialize()
    win.iren3.Initialize()
    win.iren4.Initialize()
    sys.exit(app.exec_())
