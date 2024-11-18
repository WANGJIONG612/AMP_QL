# Ocu_loong

#### 介绍
基于SerialStudio开源工程，按照高频数据收发需求进行修改，适配机器人的调试上位机

#### 软件架构

软件架构未做修改，与[Serial-Studio](http://github.com/Serial-Studio/Serial-Studio)一致，针对高频数据收发造成的软件崩溃进行修改，临时采用QWidget代替QML窗口显示曲线以实现曲线缩放与数据显示功能。

#### 使用说明

与[Serial-Studio](http://github.com/Serial-Studio/Serial-Studio)一致，根据数据协议对应修改json文件，实现数据显示与在线画图。
根据openloong数据协议，制定了对应的long.json

