from connect.client import Client
from ui.plt import Interface

class AIAR(object):
    """docstring for AIAR"""
    def __init__(self):
        super(AIAR, self).__init__()
        self.interface = Interface()
        client = Client()
        client.run()


if __name__ == '__main__':

    AIAR()
