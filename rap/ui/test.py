import PyQt5.QtWidgets as qt

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
        self.setWindowTitle('pca algorithm')
        self.mdi_area = qt.QMdiArea(self)
        self.setCentralWidget(self.mdi_area)

        # menuFile
        self.menu_file = qt.QMenu('File')
        self.menu_file.addAction(create_action(self, 'pca', slot=self.__slt_pca__))
        self.menu_file.addSeparator()
        self.menu_file.addAction(create_action(self, 'close', slot=self.close))

        # menuView
        self.menu_view = qt.QMenu('View')
        self.menu_view.addAction(create_action(self, 'tile', slot=self.mdi_area.tileSubWindows))
        self.menu_view.addAction(create_action(self, 'cascade', slot=self.mdi_area.cascadeSubWindows))
        self.menu_view.addSeparator()
        self.menu_view.addAction(create_action(self, 'previous', slot=self.mdi_area.activatePreviousSubWindow))
        self.menu_view.addAction(create_action(self, 'next', slot=self.mdi_area.activateNextSubWindow))
        self.menu_view.addSeparator()
        self.menu_view.addAction(create_action(self, 'close current', slot=self.mdi_area.closeActiveSubWindow))
        self.menu_view.addAction(create_action(self, 'close all', slot=self.mdi_area.closeAllSubWindows))

        # menuHelp
        self.menu_help = qt.QMenu('Help')
        self.menu_help.addAction(create_action(self, 'manual'))
        self.menu_help.addSeparator()
        self.menu_help.addAction(create_action(self, 'about'))

        self.menuBar().addMenu(self.menu_file)
        self.menuBar().addMenu(self.menu_view)
        self.menuBar().addMenu(self.menu_help)

    # 后面章节补充
    def __slt_pca__(self):
        pass


if __name__ == '__main__':
    import sys

    app = qt.QApplication(sys.argv)
    frmMain = FrmMain()
    frmMain.showMaximized()
    sys.exit(app.exec_())