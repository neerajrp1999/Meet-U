from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication, \
    QTableWidgetItem, QPushButton, QHeaderView, \
    QAbstractItemView, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
import waitingCallerRoom
from ui.main_window_ui import Ui_MainWindow
from firebase_admin_tester import *
import backRunning
import threading

#import socket   
#hostname=socket.gethostname()   
#IPAddr=socket.gethostbyname(hostname)  
#print(IPAddr)

class MainWindow(QMainWindow):
    def __init__(self, user_id):
        super(MainWindow, self).__init__()

        self.USER_ID = user_id
        backRunning.setUserID(user_id)
        thread = threading.Thread(target=backRunning.backgroundRunStart)
        thread.start()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.contact_verify=False

        ## initialize widget in app
        # menu widget and main window
        self.contact_list_leftmenu = self.ui.contact_list_leftmenu_pushButton
        self.add_new_contact_leftmenu = self.ui.add_new_contact_leftmenu_pushButton
        #self.history_leftmenu = self.ui.history_leftmenu_pushButton

        self.pages = self.ui.stackedWidget

        self.search_lineEdit_page1 = self.ui.search_lineEdit_contact_list
        self.table_contact_list = self.ui.tableWidget_contact_list_page1
        #self.tableWidget_call_history_page3 = self.ui.tableWidget_call_history_page3

        self.ui.search_lineEdit_contact_list.textChanged.connect(self.on_SearchBtn_contact_list_page_clicked)
        #self.ui.search_lineEdit_page3.textChanged.connect(self.on_showSearchBtn_page3_clicked)

        #page2
        self.gmail_lineEdit_addnewcontact_page2 = self.ui.gmail_lineEdit_addnewcontact_page2
        self.name_lineEdit_addnewcontact_page2 = self.ui.name_lineEdit_addnewcontact_page2

        # initialize QTableWidget
        self.table_contact_list.setRowCount(0)
        self.table_contact_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_contact_list.setColumnWidth(0, 20)
        self.table_contact_list.setColumnWidth(1, 20)

        self.contact_list_data=getContactListOfUser(self.USER_ID)
        #self.contact_list_data=[{'nnnfgn.gmail.com': 'hh'}, {'nnnfgn.gma1il.com': 'hh'}, {'neerajrp1999.2@gmail.com': 'radhe'}]
        self.call_history_list_data=[]
        updateIpAddress(self.USER_ID)

        self.addDataInContactTable(self.contact_list_data)
        # show password list page when start app
        self.pages.setCurrentIndex(0)
        
        self.contact_list_leftmenu.toggled.connect(
            lambda: self.do_change_page(self.contact_list_leftmenu))
        self.add_new_contact_leftmenu.toggled.connect(
            lambda: self.do_change_page(self.add_new_contact_leftmenu))
        #self.history_leftmenu.toggled.connect(
        #    lambda: self.do_change_page(self.history_leftmenu))

        # search password list by condition(username and website)
        #self.search_lineEdit.textChanged.connect(self.on_SearchBtn_contact_list_page_clicked)
        #self.website_show.textChanged.connect(self.on_SearchBtn_contact_list_page_clicked)

    # Main window ///////////////////////////////////////////////////////////////
    @pyqtSlot()
    def on_logout_leftmenu_pushButton_clicked(self):
        #logout
        # create QMessageBox
        msgBox = QMessageBox(self)
        msgBox.setWindowIcon(QIcon("./static/icon/webcam-4-128.ico"))
        msgBox.setIconPixmap(QPixmap("./static/icon/question-mark-7-48.ico"))
        msgBox.setWindowTitle('LogOut?')
        msgBox.setText("Are you sure to LogOut and Exit?")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            self.close()
        else:
            return

    def do_change_page(self, btn):
        btn_text = btn.text().strip()
        if btn_text == self.contact_list_leftmenu.text().strip():
            self.pages.setCurrentIndex(0)
            #do refresh
            self.on_SearchBtn_contact_list_page_clicked()
        elif btn_text == self.add_new_contact_leftmenu.text().strip():
            self.pages.setCurrentIndex(1)
        else:
            self.pages.setCurrentIndex(2)
            #do refresh
    
    
    @pyqtSlot()
    def on_AddContactBtn_addnewcontact_page2_clicked(self):
        name=self.name_lineEdit_addnewcontact_page2.text().strip()
        mail=self.gmail_lineEdit_addnewcontact_page2.text().strip()
        if not self.check(mail):
            self.warning_messagebox("G-Mail ID Is Invalid..")
            return
        if len(name)==0:
            self.warning_messagebox("Name lenght must be greater than 0..")
            return
        if not IsUserAlreadyExist(mail):
            self.warning_messagebox("User does not exist in database..")
            return
        if not self.contact_verify:
            self.warning_messagebox("Verify contact first..")
            return
        AddContactListOfUser(self.USER_ID,mail,name)
        self.done_messagebox("Contact Added Successful..")
        self.name_lineEdit_addnewcontact_page2.clear()
        self.gmail_lineEdit_addnewcontact_page2.clear()
        self.contact_list_data=getContactListOfUser(self.USER_ID)
        
    @pyqtSlot()
    def on_verifyContactBtn_addnewcontact_page2_clicked(self):
        name=self.name_lineEdit_addnewcontact_page2.text().strip()
        mail=self.gmail_lineEdit_addnewcontact_page2.text().strip()
        if not self.check(mail):
            self.warning_messagebox("G-Mail ID Is Invalid..")
            return
        if len(name)==0:
            self.warning_messagebox("Name lenght must be greater than 0..")
            return
        if not IsUserAlreadyExist(mail):
            self.warning_messagebox("User does not exist in database..")
            return
        self.done_messagebox("Contact Verified Successful..")
        self.contact_verify=True
        #AddContactListOfUser(self.USER_ID,mail,name)

    def done_messagebox(self, content):
        msgBox = QMessageBox(self)
        msgBox.setWindowIcon(QIcon("./static/icon/key-6-128.ico"))
        msgBox.setIconPixmap(QPixmap("./static/icon/verify-50.png"))
        msgBox.setWindowTitle("Info")
        msgBox.setText(content)
        msgBox.setStandardButtons(QMessageBox.Close)
        msgBox.exec_()

    @pyqtSlot()
    def on_ResetBtn_addnewcontact_page2_clicked(self):
        self.name_lineEdit_addnewcontact_page2.clear()
        self.gmail_lineEdit_addnewcontact_page2.clear()
        
    @pyqtSlot()
    def on_RefreshBtn_contact_list_page_clicked(self):
        #refresh
        self.search_lineEdit_page1.clear()
        self.contact_list_data=getContactListOfUser(self.USER_ID)
        #self.addDataInContactTable(self.contact_list_data)
        
    def call(self):
        pass
    """
    @pyqtSlot()
    def on_showSearchBtn_page3_clicked(self):
        search_string=self.ui.search_lineEdit_page3.text().strip()
        temp=[]
        for row_data in self.call_history_list_data:
            #not done
            gmail=list(row_data.keys())[0]
            if(search_string.lower() in gmail.lower() 
               or 
               search_string.lower() in str(row_data.get(gmail)).lower()):
                temp.append(row_data)
        self.addDataInContactTable(temp)
    """
    @pyqtSlot()
    def on_SearchBtn_contact_list_page_clicked(self):
        search_string=self.search_lineEdit_page1.text().strip()
        temp=[]
        for row_data in self.contact_list_data:
            gmail=list(row_data.keys())[0]
            if(search_string.lower() in gmail.lower() 
               or 
               search_string.lower() in str(row_data.get(gmail)).lower()):
                temp.append(row_data)
        self.addDataInContactTable(temp)

    def addDataInContactTable(self, data):
        self.table_contact_list.setRowCount(len(data))
        for i,row_data in enumerate(data):
            gmail=list(row_data.keys())[0]
            self.table_contact_list.setItem(i, 0,QTableWidgetItem(str((i+1))))
            self.table_contact_list.setItem(i, 1,QTableWidgetItem(str(row_data.get(gmail))))
            self.table_contact_list.setItem(i, 2,QTableWidgetItem(str(gmail)))
            self.table_contact_list.setCellWidget(i, 3, self.createWidget("Call",self.call_contact))
            self.table_contact_list.setCellWidget(i, 4, self.createWidget("Delete",self.delete_contact))
    """
    def addDataInCallHistoryTable(self, data):
        self.tableWidget_call_history_page3.setRowCount(len(data))
        for i,row_data in enumerate(data):
            gmail=list(row_data.keys())[0]
            self.tableWidget_call_history_page3.setItem(i, 0,QTableWidgetItem(str((i+1))))
            self.tableWidget_call_history_page3.setItem(i, 1,QTableWidgetItem(str(row_data.get(gmail))))
            self.tableWidget_call_history_page3.setItem(i, 2,QTableWidgetItem(str("Date")))
            self.tableWidget_call_history_page3.setItem(i, 3,QTableWidgetItem(str("Call Type")))
            self.tableWidget_call_history_page3.setItem(i, 4,QTableWidgetItem(str(gmail)))
            self.tableWidget_call_history_page3.setCellWidget(i, 5, self.createWidget("Delete",self.delete_contact))
    """

    def createWidget(self,name,click_method):
        self.btn = QPushButton()
        self.btn.setObjectName(name)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./static/icon/"+name+".png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn.setIcon(icon)
        self.btn.setFixedWidth(100)
        self.btn.clicked.connect(click_method)
        layout = QHBoxLayout()
        layout.addWidget(self.btn)
        layout.setAlignment(Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    def call_contact(self):
        button = self.sender()
        row = self.table_contact_list.indexAt(button.parent().pos()).row()
        gmail=list(self.contact_list_data[row].keys())[0]
        name=self.contact_list_data[row].get(gmail)
        waitingCallerRoom.caller(self.USER_ID,gmail,name)
        #self.Root = waitingCallerRoom.MainWindowCV(self.USER_ID,gmail,name)
        #self.Root.show()
        #print(row)
    
    #Delete data from QTableWidget and database
    def delete_contact(self):
        # Create QMessageBox
        msgBox = QMessageBox(self)
        msgBox.setWindowIcon(QIcon("./static/icon/key-6-128.ico"))
        msgBox.setIconPixmap(QPixmap("./static/icon/question-mark-7-48.ico"))
        msgBox.setWindowTitle("Delete?")
        msgBox.setText("Are you sure to DELETE this cotact?")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            button = self.sender()
            row = None
            if button:
                try:
                    row = self.table_contact_list.indexAt(button.parent().pos()).row()
                    gmail=list(self.contact_list_data[row].keys())[0]
                    name=self.contact_list_data[row].get(gmail)
                    RemoveContactListOfUser(self.USER_ID,gmail,name)
                    self.done_messagebox("Contact deleted successfully...")
                    self.contact_list_data=getContactListOfUser(self.USER_ID)
                    self.on_RefreshBtn_contact_list_page_clicked()
                except Exception as E:
                    content = f"Something is wrong: {E}"
                    self.warning_messagebox(context=content)
        else:
            return
        
    ## common functions
    def warning_messagebox(self, context, type=None):
        # create QMessageBox
        msgBox = QMessageBox(self)
        msgBox.setWindowIcon(QIcon("./static/icon/key-6-128.ico"))
        if type == "Success":
            msgBox.setIconPixmap(QPixmap("./static/icon/check-mark-2-48.ico"))
        else:
            msgBox.setIconPixmap(QPixmap("./static/icon/exclamation-48.ico"))
        msgBox.setWindowTitle("Warning")
        msgBox.setText(context)
        msgBox.setStandardButtons(QMessageBox.Close)
        msgBox.exec_()

    def check(self,email):
        import re
        regex = r'\b[A-Za-z0-9._%+-]+@gmail.com\b'
        return True if re.fullmatch(regex, email) else False

if __name__ == '__main__':
    pass