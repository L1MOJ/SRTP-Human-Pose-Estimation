# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'beta.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1160, 826)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setStyleSheet("background-color: qlineargradient( x0:0, y0:1, x1:0, y1:1, stop:0 rgb(20, 32, 44), stop:0.4 rgb(32, 73, 98),stop:1 rgb(37,85,117));")
        self.widget.setObjectName("widget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.topBar = QtWidgets.QWidget(self.widget)
        self.topBar.setMinimumSize(QtCore.QSize(0, 51))
        self.topBar.setMaximumSize(QtCore.QSize(16777215, 51))
        self.topBar.setStyleSheet("")
        self.topBar.setObjectName("topBar")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.topBar)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.topBar)
        self.label.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 24px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(677, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_5.addWidget(self.topBar)
        self.mainBoard = QtWidgets.QWidget(self.widget)
        self.mainBoard.setMinimumSize(QtCore.QSize(0, 651))
        self.mainBoard.setObjectName("mainBoard")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.mainBoard)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.widget_2 = QtWidgets.QWidget(self.mainBoard)
        self.widget_2.setMaximumSize(QtCore.QSize(311, 661))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_5 = QtWidgets.QWidget(self.widget_2)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.define_check = QtWidgets.QCheckBox(self.widget_5)
        self.define_check.setStyleSheet("\n"
"QCheckBox\n"
"{font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);;}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url(:/img/icon/button-off.png);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    \n"
"    image: url(:/img/icon/button-on.png);\n"
"}\n"
"")
        self.define_check.setChecked(True)
        self.define_check.setAutoExclusive(True)
        self.define_check.setObjectName("define_check")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.define_check)
        self.horizontalLayout_4.addWidget(self.define_check)
        self.predict_check = QtWidgets.QCheckBox(self.widget_5)
        self.predict_check.setStyleSheet("\n"
"QCheckBox\n"
"{font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);;}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url(:/img/icon/button-off.png);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    \n"
"    image: url(:/img/icon/button-on.png);\n"
"}\n"
"")
        self.predict_check.setAutoExclusive(True)
        self.predict_check.setObjectName("predict_check")
        self.buttonGroup_2.addButton(self.predict_check)
        self.horizontalLayout_4.addWidget(self.predict_check)
        self.verticalLayout.addWidget(self.widget_5)
        self.widget_6 = QtWidgets.QWidget(self.widget_2)
        self.widget_6.setStyleSheet("")
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.currentModel = QtWidgets.QLabel(self.widget_6)
        self.currentModel.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.currentModel.setObjectName("currentModel")
        self.horizontalLayout_8.addWidget(self.currentModel)
        self.modelBox = QtWidgets.QComboBox(self.widget_6)
        self.modelBox.setStyleSheet("QComboBox QAbstractItemView {\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 16px;\n"
"background:rgba(200, 200, 200,150);\n"
"selection-background-color: rgba(200, 200, 200,50);\n"
"color: rgb(218, 218, 218);\n"
"outline:none;\n"
"border:none;}\n"
"QComboBox{\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 16px;\n"
"color: rgb(218, 218, 218);\n"
"border-width:0px;\n"
"border-color:white;\n"
"border-style:solid;\n"
"background-color: rgba(200, 200, 200,0);}\n"
"\n"
"QComboBox::drop-down {\n"
"margin-top:8;\n"
"height:20;\n"
"background:rgba(255,255,255,0);\n"
"border-image: url(:/img/icon/下拉_白色.png);\n"
"}\n"
"")
        self.modelBox.setObjectName("modelBox")
        self.horizontalLayout_8.addWidget(self.modelBox)
        self.verticalLayout.addWidget(self.widget_6)
        self.widget_7 = QtWidgets.QWidget(self.widget_2)
        self.widget_7.setObjectName("widget_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.widget_7)
        self.label_7.setMinimumSize(QtCore.QSize(311, 61))
        self.label_7.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.widget_8 = QtWidgets.QWidget(self.widget_7)
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.fileButton = QtWidgets.QPushButton(self.widget_8)
        self.fileButton.setMinimumSize(QtCore.QSize(81, 51))
        self.fileButton.setMaximumSize(QtCore.QSize(81, 51))
        self.fileButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
"    image: url(:/img/icon/打开.png);\n"
"font-size: 14px;\n"
"font-weight: bold;\n"
"color:white;\n"
"text-align: center center;\n"
"padding-left: 5px;\n"
"padding-right: 5px;\n"
"padding-top: 4px;\n"
"padding-bottom: 4px;\n"
"border-style: solid;\n"
"border-width: 0px;\n"
"border-color: rgba(255, 255, 255, 255);\n"
"border-radius: 3px;\n"
"background-color: rgba(200, 200, 200,0);}\n"
"\n"
"QPushButton:focus{outline: none;}\n"
"\n"
"QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
"                     font-size: 14px;\n"
"                     font-weight: bold;\n"
"                     color:rgb(200,200,200);\n"
"                     text-align: center center;\n"
"                     padding-left: 5px;\n"
"                     padding-right: 5px;\n"
"                     padding-top: 4px;\n"
"                     padding-bottom: 4px;\n"
"                     border-style: solid;\n"
"                     border-width: 0px;\n"
"                     border-color: rgba(255, 255, 255, 255);\n"
"                     border-radius: 3px;\n"
"                     background-color:  #bf513b;}\n"
"\n"
"QPushButton::disabled{font-family: \"Microsoft YaHei\";\n"
"                     font-size: 14px;\n"
"                     font-weight: bold;\n"
"                     color:rgb(200,200,200);\n"
"                     text-align: center center;\n"
"                     padding-left: 5px;\n"
"                     padding-right: 5px;\n"
"                     padding-top: 4px;\n"
"                     padding-bottom: 4px;\n"
"                     border-style: solid;\n"
"                     border-width: 0px;\n"
"                     border-color: rgba(255, 255, 255, 255);\n"
"                     border-radius: 3px;\n"
"                     background-color:  #bf513b;}\n"
"QPushButton::hover {\n"
"border-style: solid;\n"
"border-width: 0px;\n"
"border-radius: 0px;\n"
"background-color: rgba(48,148,243,80);}")
        self.fileButton.setText("")
        self.fileButton.setObjectName("fileButton")
        self.horizontalLayout_5.addWidget(self.fileButton)
        self.cameraButton = QtWidgets.QPushButton(self.widget_8)
        self.cameraButton.setMinimumSize(QtCore.QSize(75, 51))
        self.cameraButton.setMaximumSize(QtCore.QSize(75, 51))
        self.cameraButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
"    image: url(:/img/icon/摄像头开.png);\n"
"font-size: 14px;\n"
"font-weight: bold;\n"
"color:white;\n"
"text-align: center center;\n"
"padding-left: 5px;\n"
"padding-right: 5px;\n"
"padding-top: 4px;\n"
"padding-bottom: 4px;\n"
"border-style: solid;\n"
"border-width: 0px;\n"
"border-color: rgba(255, 255, 255, 255);\n"
"border-radius: 3px;\n"
"background-color: rgba(48,148,243,0);}\n"
"\n"
"QPushButton:focus{outline: none;}\n"
"\n"
"QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
"                     font-size: 14px;\n"
"                     font-weight: bold;\n"
"                     color:rgb(200,200,200);\n"
"                     text-align: center center;\n"
"                     padding-left: 5px;\n"
"                     padding-right: 5px;\n"
"                     padding-top: 4px;\n"
"                     padding-bottom: 4px;\n"
"                     border-style: solid;\n"
"                     border-width: 0px;\n"
"                     border-color: rgba(255, 255, 255, 255);\n"
"                     border-radius: 3px;\n"
"                     background-color:  #bf513b;}\n"
"\n"
"QPushButton::disabled{font-family: \"Microsoft YaHei\";\n"
"                     font-size: 14px;\n"
"                     font-weight: bold;\n"
"                     color:rgb(200,200,200);\n"
"                     text-align: center center;\n"
"                     padding-left: 5px;\n"
"                     padding-right: 5px;\n"
"                     padding-top: 4px;\n"
"                     padding-bottom: 4px;\n"
"                     border-style: solid;\n"
"                     border-width: 0px;\n"
"                     border-color: rgba(255, 255, 255, 255);\n"
"                     border-radius: 3px;\n"
"                     background-color:  #bf513b;}\n"
"QPushButton::hover {\n"
"border-style: solid;\n"
"border-width: 0px;\n"
"border-radius: 0px;\n"
"background-color: rgba(48,148,243,80);}\n"
"image: url(:/img/icon/摄像头开.png);")
        self.cameraButton.setText("")
        self.cameraButton.setObjectName("cameraButton")
        self.horizontalLayout_5.addWidget(self.cameraButton)
        spacerItem1 = QtWidgets.QSpacerItem(86, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_3.addWidget(self.widget_8)
        self.verticalLayout.addWidget(self.widget_7)
        self.widget_4 = QtWidgets.QWidget(self.widget_2)
        self.widget_4.setMinimumSize(QtCore.QSize(301, 0))
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.saveButton = QtWidgets.QPushButton(self.widget_4)
        self.saveButton.setStyleSheet("QPushButton{\n"
"font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);;}\n"
"QPushButton::focus{outline: none;}\n"
"QPushButton::hover {\n"
"border-style: bold;\n"
"border-width: 0px;\n"
"border-radius: 0px;\n"
"background-color: rgba(223, 223, 223, 150);}")
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_3.addWidget(self.saveButton)
        self.verticalLayout.addWidget(self.widget_4)
        self.widget_3 = QtWidgets.QWidget(self.widget_2)
        self.widget_3.setMinimumSize(QtCore.QSize(301, 0))
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget_3)
        self.label_2.setMinimumSize(QtCore.QSize(0, 51))
        self.label_2.setStyleSheet("QLabel\n"
"{\n"
"    font-size:24px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.report_label = QtWidgets.QLabel(self.widget_3)
        self.report_label.setMinimumSize(QtCore.QSize(0, 251))
        self.report_label.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 20px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.report_label.setText("")
        self.report_label.setObjectName("report_label")
        self.verticalLayout_2.addWidget(self.report_label)
        self.verticalLayout.addWidget(self.widget_3)
        self.horizontalLayout_6.addWidget(self.widget_2)
        self.groupBox = QtWidgets.QGroupBox(self.mainBoard)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 651))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.splitter = QtWidgets.QSplitter(self.groupBox)
        self.splitter.setEnabled(True)
        self.splitter.setMinimumSize(QtCore.QSize(0, 401))
        self.splitter.setStyleSheet("#splitter::handle{background: 1px solid  rgba(200, 200, 200,100);}")
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setLineWidth(10)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setObjectName("splitter")
        self.raw_video = Label_click_Mouse(self.splitter)
        self.raw_video.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.raw_video.sizePolicy().hasHeightForWidth())
        self.raw_video.setSizePolicy(sizePolicy)
        self.raw_video.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.raw_video.setFont(font)
        self.raw_video.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.raw_video.setStyleSheet("color: rgb(218, 218, 218);\n"
"")
        self.raw_video.setText("")
        self.raw_video.setScaledContents(False)
        self.raw_video.setAlignment(QtCore.Qt.AlignCenter)
        self.raw_video.setObjectName("raw_video")
        self.out_video = Label_click_Mouse(self.splitter)
        self.out_video.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.out_video.sizePolicy().hasHeightForWidth())
        self.out_video.setSizePolicy(sizePolicy)
        self.out_video.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.out_video.setFont(font)
        self.out_video.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.out_video.setStyleSheet("color: rgb(218, 218, 218);\n"
"")
        self.out_video.setText("")
        self.out_video.setScaledContents(False)
        self.out_video.setAlignment(QtCore.Qt.AlignCenter)
        self.out_video.setObjectName("out_video")
        self.verticalLayout_6.addWidget(self.splitter)
        self.frame = QtWidgets.QFrame(self.groupBox)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QFormLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setMinimumSize(QtCore.QSize(371, 61))
        self.label_8.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.frame)
        self.label_9.setMinimumSize(QtCore.QSize(371, 61))
        self.label_9.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_9)
        self.input_poseLabel = QtWidgets.QLabel(self.frame)
        self.input_poseLabel.setMinimumSize(QtCore.QSize(371, 41))
        self.input_poseLabel.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.input_poseLabel.setText("")
        self.input_poseLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.input_poseLabel.setObjectName("input_poseLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.input_poseLabel)
        self.output_poseLabel = QtWidgets.QLabel(self.frame)
        self.output_poseLabel.setMinimumSize(QtCore.QSize(371, 41))
        self.output_poseLabel.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 18px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.output_poseLabel.setText("")
        self.output_poseLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.output_poseLabel.setObjectName("output_poseLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.output_poseLabel)
        self.verticalLayout_6.addWidget(self.frame)
        self.horizontalLayout_6.addWidget(self.groupBox)
        self.verticalLayout_5.addWidget(self.mainBoard)
        self.status = QtWidgets.QWidget(self.widget)
        self.status.setMinimumSize(QtCore.QSize(1101, 0))
        self.status.setObjectName("status")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.status)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.statuslabel = QtWidgets.QLabel(self.status)
        self.statuslabel.setMinimumSize(QtCore.QSize(981, 0))
        self.statuslabel.setStyleSheet("QLabel\n"
"{\n"
"    font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: light;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.statuslabel.setText("")
        self.statuslabel.setObjectName("statuslabel")
        self.horizontalLayout_2.addWidget(self.statuslabel)
        spacerItem2 = QtWidgets.QSpacerItem(116, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout_5.addWidget(self.status)
        self.horizontalLayout_7.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "通用型姿态检测beta"))
        self.define_check.setText(_translate("MainWindow", "定义"))
        self.predict_check.setText(_translate("MainWindow", "预测"))
        self.currentModel.setText(_translate("MainWindow", "当前模型"))
        self.label_7.setText(_translate("MainWindow", "  输入选择"))
        self.saveButton.setText(_translate("MainWindow", "结果保存路径"))
        self.label_2.setText(_translate("MainWindow", "检测结果"))
        self.label_8.setText(_translate("MainWindow", "                             输入动作"))
        self.label_9.setText(_translate("MainWindow", "                           预测动作"))
from MouseLabel import Label_click_Mouse
import apprcc_rc
