import sys
import numpy as np
from PyQt5.Qt import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory

from hyperanalysis.view.image.form import mainwindow
from hyperanalysis.process.properties import data_shape
from hyperanalysis.utils._exception import HyperanalysisError


class ImageDialog(QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, data, parent=None):
        super(ImageDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('View Image')
        self.data = data
        self.slider.valueChanged.connect(self.updateImage)

        if data is None:
            self.slider.setEnabled(False)

        elif type(data) != np.ndarray:
            raise HyperanalysisError('Data must be a numpy array.')

        else:
            shape, dimensions = data_shape(data)
            self.shape = shape
            self.dimensions = dimensions

            if dimensions != 2 and dimensions != 3:
                raise HyperanalysisError('Expected data to have dimensions of 2 or 3 only.')

            if dimensions == 2:
                self.slider.setEnabled(False)
                self.imagewin.setImage(data)

            elif dimensions == 3:
                self.slider.setValue(0)
                self.slider.setMaximum(self.shape[2])
                self.imagewin.setImage(np.squeeze(data[:, :, 0]))

    def updateImage(self):
        self.imagewin.setImage(np.squeeze(self.data[:, :, int(self.slider.value())-1]))


def imagePlot(data):
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

    form = ImageDialog(data)
    form.show()
    app.exec_()


# If the file is run directly and not imported, this runs the main function
if __name__ == '__main__':
    imagePlot(None)
# sys.exit(1)
