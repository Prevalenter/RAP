import os

import PyQt5.QtWidgets as qt

import sys
sys.path.append('..')
from ui.MainWidget import MainWidget
from ui.recoder import FTRecoderWidget

from ui.menu.data_collect.diffusion_peg_in_hole import DiffusionPegInHoleWidget

def create_action(obj_parent, text, slot = None, checkable = False, icon = None):
    action = qt.QAction(text, obj_parent)
    if slot:
        action.triggered.connect(slot)
    action.setCheckable(checkable)
    if icon:
        action.setIcon(icon)
    return action

class FrmMain(qt.QMainWindow):
    def __init__(self, parent=None):
        super(FrmMain, self).__init__(parent)
        self.__initUI__()

    def __initUI__(self):
        self.setWindowTitle('Robot Assembly Platform')
        # self.mdi_area = qt.QMdiArea(self)
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)

        self.menu_data = qt.QMenu('Data')
        self.menu_data.addAction(create_action(self, 'Peg in Hole',
                                        slot= lambda : self.main_widget.peg_in_hole_widget.show()))

        # DiffusionPegInHoleWidget().show()

        self.menu_monitor = qt.QMenu('Monitor')
        # print(self.main_widget.para_magager)
        # for k in self.main_widget.para_magager:
        # self.menu_monitor.addAction(create_action(self, 'FT monitor',
        #                                        slot= lambda : FTRecoderWidget().show()))
        self.menu_monitor.addAction(create_action(self, 'FT monitor',
                                               slot= lambda : self.main_widget.ft_recoder.show()))


        self.menuBar().addMenu(self.menu_data)
        self.menuBar().addMenu(self.menu_monitor)

        # self.menu_set = qt.QMenu('Setting')
        # # print(self.main_widget.para_magager)
        # for k in self.main_widget.para_magager:
        #     self.menu_set.addAction(create_action(self, f'{k} Set',
        #                                            slot=self.on_setting(k)))
        # self.menuBar().addMenu(self.menu_set)

    def on_setting(self, k):
        # print('on setting')
        def f():
            print('key is: ', k)
            self.main_widget.para_magager[k].show()
        return f



if __name__ == '__main__':
    import sys

    app = qt.QApplication(sys.argv)
    frmMain = FrmMain()
    frmMain.showMaximized()
    sys.exit(app.exec_())