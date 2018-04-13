import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.Qt import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory

from skhyper.view.hsi.form import mainwindow
from skhyper.process._properties import data_shape


class HSIDialog(QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, data, parent=None):
        super(HSIDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('View Hyperspectral Data')
        self.data = data
        self.shape = None
        self.dimensions = None

        self.slider.valueChanged.connect(self.funcUpdateLayer)
        self.updateImage.clicked.connect(self.funcUpdateImage)
        self.updateSpectrum.clicked.connect(self.funcUpdateSpectrum)
        self.Reset.clicked.connect(self.funcReset)

        # --- Setting image/plot settings -----------------------
        self.spec_lo = 0
        self.spec_hi = 0

        self.pgLRI = pg.LinearRegionItem()
        self.specwin.addItem(self.pgLRI)

        self.pgLRI.sigRegionChanged.connect(self.spec_region_updated)

        self.plotline = self.specwin.plot()
        # -------------------------------------------------------

        self.loadData()

    def loadData(self):
        if self.data is None:
            self.slider.setEnabled(False)

        elif type(self.data) != np.ndarray:
            raise TypeError('Data must be a numpy array.')

        else:
            shape, dimensions = data_shape(self.data)
            self.shape = shape
            self.dimensions = dimensions
            # self.slider.setMaximum(self.shape[2]-1)

            if dimensions != 3 and dimensions != 4:
                raise TypeError('Expected data to have dimensions of 3 or 4 only.')

            if dimensions == 3:
                self.slider.setEnabled(False)
                self.dataImage(self.data)
                self.dataSpectrum(self.data)

            elif dimensions == 4:
                self.slider.setValue(0)
                self.slider.setMaximum(self.shape[2]-1)
                self.dataImage(self.data[:, :, 0, :])
                self.dataSpectrum(self.data[:, :, 0, :])

    def spec_region_updated(self, regionItem):
        self.spec_lo, self.spec_hi = regionItem.getRegion()

    def funcUpdateLayer(self):
        self.dataImage(self.data[:, :, int(self.slider.value()), :])
        self.dataSpectrum(self.data[:, :, int(self.slider.value()), :])

    def funcUpdateImage(self):
        if self.dimensions == 3:
            self.dataImage(self.data[:, :, int(self.spec_lo):int(self.spec_hi)])

        elif self.dimensions == 4:
            self.dataImage(self.data[:, :, int(self.slider.value())-1, int(self.spec_lo):int(self.spec_hi)])

    def funcUpdateSpectrum(self):
        # Obtaining coordinates of ROI graphic in the image plot
        image_coord_handles = self.imagewin.roi.getState()
        posimage = image_coord_handles['pos']
        sizeimage = image_coord_handles['size']

        posx = int(posimage[0])
        sizex = int(sizeimage[0])
        posy = int(posimage[1])
        sizey = int(sizeimage[1])
        xmin = posx
        xmax = posx + sizex
        ymin = posy
        ymax = posy + sizey

        if self.dimensions == 3:
            self.dataSpectrum(self.data[ymin:ymax, xmin:xmax, :])

        elif self.dimensions == 4:
            self.dataSpectrum(self.data[ymin:ymax, xmin:xmax, int(self.slider.value())-1, :])

    def funcReset(self):
        self.loadData()

    def dataImage(self, data):
        self.imagewin.setImage(np.squeeze(np.mean(data, 2)))

    def dataSpectrum(self, data):
        self.plotline.setData(np.squeeze(np.mean(np.mean(data, 1), 0)))


def hsiPlot(data):
    app = QApplication(sys.argv)

    # Setting the dark-themed Fusion style for the GUI
    app.setStyle(QStyleFactory.create('Fusion'))
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(23, 23, 23))
    dark_palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    dark_palette.setColor(QPalette.Base, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, QColor(200, 200, 200))
    dark_palette.setColor(QPalette.Button, QColor(33, 33, 33))
    dark_palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(33, 33, 33))
    dark_palette.setColor(QPalette.Disabled, QPalette.Button, QColor(20, 20, 20))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))

    app.setPalette(dark_palette)

    form = HSIDialog(data)
    form.show()
    app.exec_()
    # sys.exit(app.exec_())


# If the file is run directly and not imported, this runs the main function
if __name__ == '__main__':
    hsiPlot(None)
# sys.exit(1)
