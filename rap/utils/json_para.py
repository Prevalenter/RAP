import json

def save_para(path, para):
    with open(path, 'w') as result_file:
        json.dump(para, result_file, indent=2)

def load_para(path):
    with open(path, 'r') as result_file:
        save_dict = json.load(result_file)
    return save_dict


if __name__ == '__main__':
    para = load_para("../data/para.json")
    print(para)

    para['compliance']['M'] = [0.01, 0.01, 0.01, 0, 0, 0]

    print(type(para['compliance']['M']))

    save_para('../data/para_save.json', para)

