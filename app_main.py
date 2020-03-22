# -*- coding: utf-8 -*-
# import idcardocr
from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage

import findidcard
import recognizeidcard
import tkinter as tk
import cv2


def process(img_name):
    try:
        idfind = findidcard.findidcard()
        idcard_img = idfind.find(img_name)
        result_dict = recognizeidcard.idcardocr(idcard_img)
        result_dict['error'] = 0
    except Exception as e:
        result_dict = {'error': 1}
        print(e)
    return result_dict


if __name__ == '__main__':

    idcardimagepath = 'images/gfm.jpg'

    # 实例化object，建立窗口window
    window = tk.Tk()

    # 给窗口的可视化起名字
    window.title('身份证字符识别工具：高傅敏')

    # 设定窗口的大小(长 * 宽)
    window.geometry('700x700')  # 这里的乘是小x

    # 在图形界面上设定标签
    title = tk.Label(window, text='身份证字符识别工具', bg='green', font=('Arial', 12), width=30, height=2).pack(side="top",
                                                                                                        fill='x')
    # 说明： bg为背景，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
    img = Image.open(idcardimagepath)
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(window, image=photo)
    img_label.pack()


    name_str = tk.StringVar()
    nation_str = tk.StringVar()
    sex_str = tk.StringVar()
    birth_str = tk.StringVar()
    address_str = tk.StringVar()
    idnum_str = tk.StringVar()

    name_label = tk.Label(window, textvariable=name_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')
    nation_label = tk.Label(window, textvariable=nation_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')
    sex_label = tk.Label(window, textvariable=sex_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')
    birth_label = tk.Label(window, textvariable=birth_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')
    address_label = tk.Label(window, textvariable=address_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')
    idnum_label = tk.Label(window, textvariable=idnum_str, bg='green', font=('Arial', 12), width=30, height=2).pack(fill='x')



    info = process(idcardimagepath)
    error = info['error']
    if error == 0:
        name = info['name']
        nation = info['nation']
        sex = info['sex']
        birth = info['birth']
        address = info['address']
        idnum = info['idnum']
        print('*' * 30)
        print('姓名:   ' + name)
        print('民族:    ' + nation)
        print('性别:    ' + sex)
        print('生日:  ' + birth)
        print('地址:   ' + address)
        print('公民身份证号码:  ' + idnum)
        print('*' * 30)

        name_str.set(name)
        nation_str.set(nation)
        sex_str.set(sex)
        birth_str.set(birth)
        address_str.set(address)
        idnum_str.set(idnum)
    else:
        print(info)

    #主窗口循环显示
    window.mainloop()
    # 注意，loop因为是循环的意思，window.mainloop就会让window不断的刷新，如果没有mainloop,就是一个静态的window,传入进去的值就不会有循环，mainloop就相当于一个很大的while循环，有个while，每点击一次就会更新一次，所以我们必须要有循环
    # 所有的窗口文件都必须有类似的mainloop函数，mainloop是窗口文件的关键的关键。
