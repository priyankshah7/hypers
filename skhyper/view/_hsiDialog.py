import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.Qt import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory

from skhyper import process
from skhyper.view._form import mainwindow


class HSIDialog(QMainWindow, mainwindow.Ui_MainWindow):
    """ Hyperspectral data viewer

    Displays the hyperspectral data. Features include:
    - Multiple layers (z-axis) by scrolling through the layers.
    - Viewing the image pertaining to chosen spectral bands.
    - Viewing the spectra averaged over a chosen region in the image.

    This class should not be imported directly. Instead either:
    - Import hsiPlot from skhyper.view and use on a Process object
    - Call the `view()` method of a Process object

    """
    def __init__(self, X, parent=None):
        super(HSIDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('View Hyperspectral Data')

        if not isinstance(X, process.Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        self.data = X

        self.shape = None
        self.dimensions = None

        self.slider.valueChanged.connect(self.update_layer)
        self.updateImage.clicked.connect(self.update_image)
        self.updateSpectrum.clicked.connect(self.update_spectrum)
        self.Reset.clicked.connect(self.reset)

        # --- Setting image/plot settings -----------------------
        self.spec_lo = 0
        self.spec_hi = 0

        self.pgLRI = pg.LinearRegionItem()
        self.specwin.addItem(self.pgLRI)

        self.pgLRI.sigRegionChanged.connect(self.spec_region_updated)

        self.plotline = self.specwin.plot()
        # -------------------------------------------------------

        self.load_data()

    def load_data(self):
        if self.data is None:
            self.slider.setEnabled(False)

        else:
            self.shape = self.data.shape
            self.dimensions = self.data.n_dimension
            # self.slider.setMaximum(self.shape[2]-1)

            if self.dimensions == 3:
                self.slider.setEnabled(False)
                self.data_image(self.data)
                self.data_spectrum(self.data)

            elif self.dimensions == 4:
                self.slider.setValue(0)
                self.slider.setMaximum(self.shape[2]-1)
                self.data_image(self.data[:, :, 0, :])
                self.data_spectrum(self.data[:, :, 0, :])

    def spec_region_updated(self, regionItem):
        self.spec_lo, self.spec_hi = regionItem.getRegion()

    def update_layer(self):
        self.data_image(self.data[:, :, int(self.slider.value()), :])
        self.data_spectrum(self.data[:, :, int(self.slider.value()), :])

    def update_image(self):
        if self.dimensions == 3:
            self.data_image(self.data[:, :, int(self.spec_lo):int(self.spec_hi)])

        elif self.dimensions == 4:
            self.data_image(self.data[:, :, int(self.slider.value()) - 1, int(self.spec_lo):int(self.spec_hi)])

    def update_spectrum(self):
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
            self.data_spectrum(self.data[ymin:ymax, xmin:xmax, :])

        elif self.dimensions == 4:
            self.data_spectrum(self.data[ymin:ymax, xmin:xmax, int(self.slider.value()) - 1, :])

    def reset(self):
        self.load_data()

    def data_image(self, data):
        self.imagewin.setImage(np.squeeze(np.mean(data, 2)))

    def data_spectrum(self, data):
        self.plotline.setData(np.squeeze(np.mean(np.mean(data, 1), 0)))


def hsiPlot(X):
    """Hyperspectral data viewer

    Displays the hyperspectral data. This is one of two methods to do this.

    Parameters
    ----------
    X : object, Process instance
        The object X containing the hyperspectral array.

    """
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

    form = HSIDialog(X)
    form.show()
    app.exec_()
    # sys.exit(app.exec_())


# If the file is run directly and not imported, this runs the main function
if __name__ == '__main__':
    hsiPlot(None)
# sys.exit(1)
