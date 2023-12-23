# coding: utf-8
# tcp stream client
import socket
import logging
import time
import os
from threading import Thread, currentThread

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)s %(threadName)s |%(message)s",
)

class Client:
    """ socket 客户端 """
    def __init__(self, host='192.168.3.5', port=12340, parent=None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._addr = (host, port)

        self.parent = parent

        self.is_writing = False


    def read(self):
        """ 向连接中接受数据 """
        while True:
            try:
                # if self.is_writing: continue
                data = self._sock.recv(1024).decode()
                # print(len(data))
                # logging.info("[R %s]<< %s", currentThread().getName(), data)
                if self.parent!=None: 
                    self.parent.set_msg_rcv(data)
                # print('set_msg_rcv down')
                # time.sleep(0.1)

            except Exception as e:
                logging.info("recv failed: %s", e)
                return

    def write(self, msg):
        """ 向连接中发送随机数 """
        # while True:
        # if self.is_writing:
        #     print('is_writing, return ')
        #     return

        # self.is_writing = True
        logging.info("[W %s]>> %s", currentThread().getName(), msg)

        try:
            self._sock.send(msg.encode())
        except Exception as e:
            logging.info("send failed: %s", e)
            return
        # time.sleep(2)
        # print("write down")
        # self.is_writing = False

    def cmd_initial(self):
        cmd_list =  [["00,cl", 0.5],
                     ["00,ds", 0.5],
                     ["00,md", 0.5],
                     ["00,en", 3],
                     # ["00,BZ", 4],
                     ["00,FL", 1.5 ],
                     # ["01,{1,2,3,4,5,6}", 1.5 ],
                     # ["00,XYZ --p={0.45,0.10,0.30,3.14,0.5,3.0} --t_total=2 --t_interval=2", 3],
                     # ["00,XYZ --p={0.4,0.15,0.50,3.14,0.5,3.0} --t_total=2 --t_interval=2", 3]
                     ]

        for cmd_str, t_sleep in cmd_list:
            print("in send: ", cmd_str)
            self.write(cmd_str)
            time.sleep(t_sleep)

    def run(self, block=True):
        """ 开启连接 """
        self._sock.connect(self._addr)
        logging.info("New connection: %s", self._sock)
        r = Thread(target=self.read)
        r.start()
        w = Thread(target=self.write)

        # self.cmd_initial()
        cmd = Thread(target=self.cmd_initial)
        cmd.start()
        # cmd.join()

        # w.start()
        if block: r.join()
        # w.join()



if __name__ == '__main__':
    client = Client()
    client.run()
