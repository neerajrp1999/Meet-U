import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication, \
    QTableWidgetItem, QPushButton, QHeaderView, \
    QAbstractItemView, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from firebase_admin_tester import *
import tkinter as tk
from tkinter import *
from stream_ import CameraClient,AudioReceiver,AudioSender,StreamingServer
import threading
import time
import socket
from firebase_admin_tester import *
from tkinter import Label,Button,Tk 


class caller():
    def __init__(self,user,another_user,another_user_name):
        self.another_user_name=another_user_name
        self.another_user=another_user
        self.user=user
        self.feed=True

        self.root = Tk()
        self.root.geometry('640x580')
        self.FeedLabel=Label(self.root,text="Calling....   wait for response....",width=500,height=500)
        self.FeedLabel.pack(side="top")
        self.btn = Button(self.root, text = 'Drop Call', command = lambda:self.CancelCall())
        self.btn.pack(side="bottom") 
        self.startWork()        
        self.root.mainloop()
    
    def CancelCall(self):
        CallDataUpdate(self.another_user,"5")
        self.root.destroy()

    def startWork(self):
        sendCallRequest(self.user,self.another_user,self.another_user_name)
        def checkStatus(another_user):
            gotStatus=False
            while not gotStatus:
                status=getStatus(another_user)
                if(status=="1"):
                    continue
                elif(status=="2"):
                    self.FeedLabel.config(text="Ringing the , wait for respones..")
                elif(status=="3"):
                    self.FeedLabel.config(text="Connecting to user, just wait..")
                    #self.root.destroy()
                    #VideoChatCaller.caller(getIpAddress(self.another_user),self.FeedLabel,self.btn)
                    self.caller1(getIpAddress(self.another_user),self.FeedLabel,self.btn,self.root)
                    break
                elif(status=="4"):
                    self.FeedLabel.config(text="Call Decline..")
                    break
        import threading
        callroomThread=threading.Thread(target=checkStatus,args=(self.another_user,))
        callroomThread.start()
    def caller1(self,ip_of_send,panel,btn,root):
        btn.configure(command = lambda:stop())
        root.protocol("WM_DELETE_WINDOW", lambda:stop())
        
        IPAddr = socket.gethostbyname(socket.gethostname())

        receiving_audio =AudioReceiver(IPAddr,8081)#your ip
        receiving_video =StreamingServer(IPAddr,8082,panel)#your ip
        
        sending_audio=AudioSender(ip_of_send,8081)
        sending_video=CameraClient(ip_of_send,8082)
        def stop():
            #self.window.destroy()
            root.destroy()
            print("drop")
            sending_video.stop_stream()
            sending_audio.stop_stream()
            receiving_audio.stop_server()
            receiving_video.stop_server()
        
        t1=threading.Thread(target=receiving_audio.start_server)
        t1.start()
        t2=threading.Thread(target=receiving_video.start_server)
        t2.start()  
        t3=threading.Thread(target=sending_audio.start_stream)
        t3.start()
        t4=threading.Thread(target=sending_video.start_stream)
        t4.start()

if __name__=="__main__":
    caller("","","")