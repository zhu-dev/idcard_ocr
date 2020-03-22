### 第二代身份证识别系统

##### 1.环境搭建

+ 安装`Tesseract-OCR`

  ```java
  1.安装包在工程environment文件夹下：tesseract-ocr-setup-4.00.00dev.exe
  2.建议安装路径：D:\Tesseract-OCR
  3.安装完，将environment文件夹下的environment\language 文件夹下的语言包复制到
      Tesseract-OCR下的\tessdata目录
  ```

  

+ 配置系统环境变量看word截图，environment文件夹里

  + `path: `D:\Tesseract-OCR

  + 新建：`TESSDATA_PREFIX`  `D:\Tesseract-OCR\tessdata`

  + 在命令行窗口验证：win+r 输入cmd,进入命令行

    + 输入tesseract 回车，是否有相关信息

      

#### 2.工程配置

+ 将工程`idcard`直接拖入`pycharm`,注意是跟`environment `并列的那个（记得点确定，每一步操作）
+ 配置编译器，使用你的本机编译器
+ 打开工程后，点击`requirements.txt`文件夹，上方有提示安装依赖包，点击等待下载安装完
+ 点击`app_main.py`,修改你的身份证图片路径，然后run