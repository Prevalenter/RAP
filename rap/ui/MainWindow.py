import PyQt5.QtWidgets as qt

import sys
sys.path.append('..')
from ui.MainWidget import MainWidget


def create_action(obj_parent, text, slot = None, checkable = False, icon = None):
    action = qt.QAction(text, obj_parent)
    if slot:
        action.triggered.connect(slot)
    action.setCheckable(checkable)
    if icon:
        action.setIcon(icon)
    return action

class FrmMain(qt.QMainWindow):
    '''
    demo for pca algorithm
    '''

    def __init__(self, parent=None):
        super(FrmMain, self).__init__(parent)
        # 界面布局
        self.__initUI__()

    # 函数前增加 __的目的是使得函数为私有，后面也加__ 完全是为了对称起来好看
    def __initUI__(self):
        self.setWindowTitle('Robot Assembly Platform')
        # self.mdi_area = qt.QMdiArea(self)
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)

        #
        self.menu_help = qt.QMenu('Files')
        self.menu_help.addAction(create_action(self, 'Setting', self.on_setting))

        self.menuBar().addMenu(self.menu_help)

    def on_setting(self):
        # print('on setting')
        self.main_widget.para_set.show()



if __name__ == '__main__':
    import sys

    app = qt.QApplication(sys.argv)
    frmMain = FrmMain()
    frmMain.showMaximized()
    sys.exit(app.exec_())