# coding: utf-8
# tcp stream server
import socket
import logging
import time
import datetime as dt
from threading import Thread, currentThread
# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)s %(threadName)s |%(message)s",
)


class Server:
    """ socket 服务端 """
    def __init__(self, host='localhost', port=12345):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((host, port))
        self.msg = None

    def read(self, conn: socket.socket = None):
        """ 从tcp连接里面读取数据 """
        while True:
            try:
                data = conn.recv(1024).decode()
            except Exception as e:
                logging.info("recv failed: %s", e)
                return
            logging.info("[R %s]<< %s", currentThread().getName(), data)
            msg = data
            # print('in server read: ', self.msg == "", len(self.msg))
            doing(msg)
            self.write("task finish: "+msg)
            time.sleep(1)

    def write(self, msg):
        """ 向tcp连接里面写入数据 """
        # while True:
        # msg = f"{dt.datetime.now()} - {msg}"
        logging.info("[W %s]>> %s", currentThread().getName(), msg)
        try:
            self.conn.send(msg.encode())
        except Exception as e:
            logging.info("send failed: %s", e)
            return
        time.sleep(1)

    def serve(self):
        """ 开启服务 """
        self._sock.listen()
        logging.info("Serving...")
        while True:
            logging.info("Waiting for connection...")
            conn, addr = self._sock.accept()
            self.conn = conn
            logging.info("Recived new conn: %s from %s", conn, addr)
            # 开启读写线程处理当前连接
            Thread(target=self.read, args=(conn, )).start()
            # Thread(target=self.write, args=(conn, )).start()

def doing(cmd_msg):
    print("doing begin: ", cmd_msg)
    time.sleep(0.5)
    print("doing end ", cmd_msg)

if __name__ == '__main__':
    # doing("go")
    s = Server()
    s.serve()


