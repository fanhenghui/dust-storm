<<<<<<< HEAD
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：server.py

import socket               # 导入 socket 模块

s = socket.socket()         # 创建 socket 对象
host = socket.gethostname() # 获取本地主机名
port = 12345                # 设置端口
s.bind((host, port))        # 绑定端口

s.listen(5)                 # 等待客户端连接
while True:
    c, addr = s.accept()     # 建立客户端连接。
    print '连接地址：', addr
    c.send('欢迎访问菜鸟教程！')
    c.close()                # 关闭连接
=======
#!/usr/bin/python  
import socket  
import signal  
import errno  
from time import sleep   
  
  
def HttpResponse(header,whtml):  
    f = file(whtml)  
    contxtlist = f.readlines()  
    context = ''.join(contxtlist)  
    response = "%s %d\n\n%s\n\n" % (header,len(context),context)  
    return response  
  
def sigIntHander(signo,frame):  
    print 'get signo# ',signo  
    global runflag  
    runflag = False  
    global lisfd  
    lisfd.shutdown(socket.SHUT_RD)  
  
strHost = "172.20.52.163"  
HOST = strHost #socket.inet_pton(socket.AF_INET,strHost)  
PORT = 20014  
  
httpheader = '''''\ 
HTTP/1.1 200 OK 
Context-Type: text/html 
Server: Python-slp version 1.0 
Context-Length: '''  
  
lisfd = socket.socket(socket.AF_INET,socket.SOCK_STREAM)  
lisfd.bind((HOST, PORT))  
lisfd.listen(2)  
  
signal.signal(signal.SIGINT,sigIntHander)  
  
runflag = True  
while runflag:  
    try:  
        confd,addr = lisfd.accept()  
    except socket.error as e:  
        if e.errno == errno.EINTR:  
            print 'get a except EINTR'  
        else:  
            raise  
        continue  
  
    if runflag == False:  
        break;  
  
    print "connect by ",addr  
    data = confd.recv(1024)  
    if not data:  
        break  
    print data  
    confd.send(HttpResponse(httpheader,'index.html'))  
    confd.close()  
else:  
    print 'runflag#',runflag  
  
print 'Done'  
>>>>>>> 443f43a30f8b5137c8a3b28b0f4378cb1f05baf8
