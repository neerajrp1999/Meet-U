import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication, \
    QTableWidgetItem, QPushButton, QHeaderView, \
    QAbstractItemView, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from firebase_admin_tester import *
import VideoChatCaller
from tkinter import *       
class caller():
    def __init__(self,user,another_user,another_user_name):
        self.another_user_name=another_user_name
        self.another_user=another_user
        self.user=user
        self.feed=True

        self.root = Tk()
        self.root.geometry('640x480')
        self.FeedLabel=Label(self.root,text="Calling....   wait for response....",width=550,height=500)
        self.FeedLabel.pack(side="top")
        btn = Button(self.root, text = 'Drop Call', command = lambda:self.CancelCall())
        btn.pack(side="bottom") 
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
                    self.FeedLabel.config(text="Connecting to user just wait..")
                    #self.root.destroy()
                    VideoChatCaller.caller(getIpAddress(self.another_user),self.FeedLabel)
                    break
                elif(status=="4"):
                    self.FeedLabel.config(text="Call Decline..")
                    break
        import threading
        callroomThread=threading.Thread(target=checkStatus,args=(self.another_user,))
        callroomThread.start()

if __name__=="__main__":
    caller("","","")