# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1243, 458)
        MainWindow.setStyleSheet("%background: #161616")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imagewin = ImageView(self.centralwidget)
        self.imagewin.setObjectName("imagewin")
        self.horizontalLayout_2.addWidget(self.imagewin)
        self.specwin = PlotWidget(self.centralwidget)
        self.specwin.setObjectName("specwin")
        self.horizontalLayout_2.addWidget(self.specwin)
        self.horizontalLayout_2.setStretch(0, 40)
        self.horizontalLayout_2.setStretch(1, 60)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.slider = QtWidgets.QSlider(self.centralwidget)
        self.slider.setMaximum(10)
        self.slider.setPageStep(1)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.horizontalLayout_3.addWidget(self.slider)
        self.updateSpectrum = QtWidgets.QPushButton(self.centralwidget)
        self.updateSpectrum.setObjectName("updateSpectrum")
        self.horizontalLayout_3.addWidget(self.updateSpectrum)
        self.updateImage = QtWidgets.QPushButton(self.centralwidget)
        self.updateImage.setObjectName("updateImage")
        self.horizontalLayout_3.addWidget(self.updateImage)
        self.Reset = QtWidgets.QPushButton(self.centralwidget)
        self.Reset.setObjectName("Reset")
        self.horizontalLayout_3.addWidget(self.Reset)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        self.gridLayout.setRowStretch(1, 95)
        self.gridLayout.setRowStretch(2, 5)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "View Hyperspectral Image"))
        self.updateSpectrum.setText(_translate("MainWindow", " Update Spectrum "))
        self.updateImage.setText(_translate("MainWindow", " Update Image "))
        self.Reset.setText(_translate("MainWindow", "RESET"))

from pyqtgraph import ImageView, PlotWidget
