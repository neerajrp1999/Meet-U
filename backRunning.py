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

        root = Tk()
        root.geometry('640x480')
        FeedLabel=Label(root,text="str(caller[1])+'('+str(caller[0])+') are calling....\nAre you sure you want to recieve this call?",width=550,height=500)
        FeedLabel.pack(side="top")

        #msg_box =  messagebox.askquestion('confirmation', str(caller[1])+'('+str(caller[0])+') are calling....\nAre you sure you want to recieve this call?')
        def yes():
            ReceivingCall_CallDataUpdate(UserID,str(3))
            print("yes")
            
            VideoChatCaller.caller(getIpAddress(caller[0]),FeedLabel)
        def no():
            ReceivingCall_CallDataUpdate(UserID,str(4))
            canReceiveCall=True
        
        btn = Button(root, text = 'Accept Call', command = lambda:yes())
        btn.pack(side="bottom") 
        btn = Button(root, text = 'Drop Call', command = lambda:no())
        btn.pack(side="bottom") 
                
        root.mainloop()
            
    print("Geeksforgeeks")
  
schedule.every(20).seconds.do(BackgroundFunction)

def backgroundRunStart():
    while canReceiveCall:
        schedule.run_pending()
        time.sleep(3)
