# style_transfer
该项目复现了Leon A. Gatys的论文《A Neural Algorithm of Artistic Style》中的工作。

论文链接：
https://arxiv.org/pdf/1508.06576v2.pdf

运行程序：
在style_transfer.py文件中指定STYLE为输入的风格图片，指定CONTENT为输入的内容图片,CONTENT图片的IMAGE_HEIGHT与IMAGE_WIDTH，
以及输出图像的路径OUT_DIR与训练的中间文件保存路径CHECKPOINT_DIR，然后run style_transfer.py的main函数，即可在OUT_DIR文件夹下得到与CONTENT图片大小一致的输出图片。