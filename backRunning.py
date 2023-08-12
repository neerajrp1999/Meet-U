# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 23:15:10 2023

@author: neera
"""
import time
import schedule
from firebase_admin_tester import *
import tkinter as tk
from tkinter import *
import VideoChatCaller
from stream_ import CameraClient,AudioReceiver,AudioSender,StreamingServer
import threading
import time
import socket
from firebase_admin_tester import *
import tkinter
from tkinter import Label,Button,Tk

UserID=""
canReceiveCall=True
def setUserID(id):
    global UserID
    UserID=id
def caller1(ip_of_send,panel,btn,root):
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
            from main_window import MainWindow
            main_window = MainWindow(user_id=UserID)
            main_window.show()
        
        t1=threading.Thread(target=receiving_audio.start_server)
        t1.start()
        t2=threading.Thread(target=receiving_video.start_server)
        t2.start()  
        t3=threading.Thread(target=sending_audio.start_stream)
        t3.start()
        t4=threading.Thread(target=sending_video.start_stream)
        t4.start()
        

def BackgroundFunction():
    if(getReceivingCall_Call(UserID)):
        global canReceiveCall
        canReceiveCall=False
        caller=getReceivingCall_CallData(UserID)
        ReceivingCall_CallDataUpdate(UserID,str(2))

        def yes():
            ReceivingCall_CallDataUpdate(UserID,str(3))
            FeedLabel.config(text="Connecting to user, just wait..")
            print("yes")
            btn1.destroy()
            btn.configure(text="Drop Call")
            btn.pack(side="bottom")
            caller1(getIpAddress(caller[0]),FeedLabel,btn,root)
        def no():
            ReceivingCall_CallDataUpdate(UserID,str(4))
            global canReceiveCall
            canReceiveCall=True
            root.destroy()
        root = Tk()
        root.geometry('640x580')
        FeedLabel=Label(root,text=str(caller[1])+'('+str(caller[0])+") are calling....\nAre you sure you want to recieve this call?",width=500,height=500)
        FeedLabel.pack(side="top")
        btn = Button(root, text = 'Accept Call', command = lambda:yes())
        btn.place(x=255, y=400) 
        btn1 = Button(root, text = 'Drop Call', command = lambda:no())
        btn1.place(x=355, y=400)
                
        root.mainloop()
            
    
  
schedule.every(5).seconds.do(BackgroundFunction)

def backgroundRunStart():
    while canReceiveCall:
        schedule.run_pending()
        time.sleep(3)


#backgroundRunStart()


"""

import time
import schedule
from firebase_admin_tester import *
import tkinter as tk
from tkinter import *
import VideoChatCaller

UserID=""
canReceiveCall=True
def setUserID(id):
    global UserID
    UserID=id
def BackgroundFunction():
    if(getReceivingCall_Call(UserID)):
        global canReceiveCall
        canReceiveCall=False
        caller=getReceivingCall_CallData(UserID)
        ReceivingCall_CallDataUpdate(UserID,str(2))

        class ca:
            def __init__(self):
                self.root = Tk()
                self.root.geometry('640x480')
                self.FeedLabel=Label(self.root,text=str(caller[1])+'('+str(caller[0])+") are calling....\nAre you sure you want to recieve this call?",width=550,height=500)
                self.btn = Button(self.root, text = 'Accept Call', command = lambda:self.yes())
                #msg_box =  messagebox.askquestion('confirmation', str(caller[1])+'('+str(caller[0])+') are calling....\nAre you sure you want to recieve this call?')
                self.FeedLabel.pack(side="top")
        
                self.btn.pack(side="bottom") 
                self.btn1 = Button(self.root, text = 'Drop Call', command = lambda:self.no())
                self.btn1.pack(side="bottom") 
                        
                VideoChatCaller.caller(getIpAddress(caller[0]),self.FeedLabel)
                self.root.mainloop()
            def yes(self):
                    ReceivingCall_CallDataUpdate(UserID,str(3))
                    print("yes")
            
            def no(self):
                    ReceivingCall_CallDataUpdate(UserID,str(4))
                    canReceiveCall=True
        s=ca()
            
    print("Geeksforgeeks")
  
schedule.every(5).seconds.do(BackgroundFunction)

def backgroundRunStart():
    while canReceiveCall:
        schedule.run_pending()
        time.sleep(3)

"""