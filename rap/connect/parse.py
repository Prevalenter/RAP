import numpy as np

def parse_recv(msg):
    cmd_type, cmd_data = msg.split(": ")

    data = []
    for i in cmd_data.split(' '):
        data.append(float(i))
    data = np.array(data)

    return cmd_type, data


if __name__ == '__main__':
    cmd_type, data = parse_recv("cur angles: 0 1.2314 2.4628 3.6942 4.9256 6.157")
    # cmd_type, data = parse_recv("task finish: 0")
    print(cmd_type, data)
