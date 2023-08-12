from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import waitingCallerRoom

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1144, 593)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icon/key-6-128.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("#logoFrame {\n"
"    image: url(:/icon/icon/key-48.ico);\n"
"    border-bottom: 1px solid #fff;\n"
"}\n"
"\n"
"/* Style for menu widget  */\n"
"#menuWidget {\n"
"    background-color: #353535;\n"
"}\n"
"\n"
"/* Style for QPushButton in menu widget  */\n"
"#menuWidget QPushButton {\n"
"    text-align: left;\n"
"    padding-left: 15px;\n"
"    border: none;\n"
"    background-color: #fff;\n"
"    font-size: 14px;\n"
"    border: 5px solid #353535;\n"
"    height: 25px;\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"#menuWidget QPushButton:hover {\n"
"    background-color: #e3e3e3;\n"
"}\n"
"\n"
"\n"
"#menuWidget QPushButton:checked {\n"
"    background-color: #0055ff;\n"
"    color: #fff;\n"
"}\n"
"\n"
"/* Style for main widget */\n"
"#stackedWidget {\n"
"    background-color: #f4f5f8;\n"
"}\n"
"\n"
"/* Style for QLabel in main widget */\n"
"#stackedWidget QLabel {\n"
"    font-size: 14px;\n"
"    font-family: \"Segoe UI Semibold\";\n"
"}\n"
"\n"
"/* Style for QLineEdit and QSpinBox in main widget */\n"
"#stackedWidget QLineEdit, #stackedWidget QSpinBox {\n"
"    border: 1px solid #353535;\n"
"    border-radius: 3px;\n"
"    padding: 5px 10px;\n"
"    font-size: 12px;\n"
"}\n"
"\n"
"/* style for  QSpinBox*/\n"
"#stackedWidget QSpinBox::up-arrow {\n"
"    image: url(:/icon/icon/arrow-146-48.ico);\n"
"    width: 12px;\n"
"    height: 12px;\n"
"}\n"
"\n"
"#stackedWidget QSpinBox::down-arrow {\n"
"    image: url(:/icon/icon/arrow-208-48.ico);\n"
"    width: 12px;\n"
"    height: 12px;\n"
"}\n"
"\n"
"#stackedWidget QSpinBox::down-button, \n"
"#stackedWidget QSpinBox::up-button {\n"
"    border: none;\n"
"    width: 30px;\n"
"}\n"
"\n"
"#stackedWidget QSpinBox::down-button:hover, \n"
"#stackedWidget QSpinBox::up-button:hover {\n"
"    background-color: rgb(176, 176, 176);\n"
"}\n"
"\n"
"#stackedWidget QSpinBox::down-button:pressed, \n"
"#stackedWidget QSpinBox::up-button:pressed {\n"
"    background-color: rgb(78, 88, 121);\n"
"}\n"
"\n"
"/* Style for QPushButton in main widget */\n"
"#stackedWidget QPushButton {\n"
"    border: none;\n"
"    font-size: 16px;\n"
"    border-radius: 3px;\n"
"    color: #fff;\n"
"    padding: 5px 10px;\n"
"}\n"
"\n"
"#stackedWidget QPushButton:pressed {\n"
"    padding-left: 20px;\n"
"}\n"
"\n"
"/* Style for sepcial QPushButton in main widget */\n"

"#CancelBTN,\n"
"#verifyContactBtn_addnewcontact_page2, \n"
"#showSearchBtn_page3,\n"
"#SearchBtn_contact_list_page\n"
"{\n"
"    background-color: #3554d1;\n"
"}\n"
"\n"

"#CancelBTN:hover, #CancelBTN:pressed, \n"
"#verifyContactBtn_addnewcontact_page2:hover,  #verifyContactBtn_addnewcontact_page2:pressed, \n"
"#SearchBtn_contact_list_page:hover,  #SearchBtn_contact_list_page:pressed, \n"
"#showSearchBtn_page3:hover,  #showSearchBtn_page3:pressed \n"
"{\n"
"    background-color: #0000ff;\n"
"}\n"
"\n"
"#AddContactBtn_addnewcontact_page2, \n"
"#ResetBtn_addnewcontact_page2\n"
"{\n"
"    background-color: #00aa7f;\n"
"}\n"
"\n"
"#AddContactBtn_addnewcontact_page2:hover, #AddContactBtn_addnewcontact_page2:pressed,\n"
"#ResetBtn_addnewcontact_page2:hover, #ResetBtn_addnewcontact_page2:pressed \n"
"{\n"
"    background-color: #00aa00;\n"
"}\n"
"\n"
" #showRefreshBtn_page3,\n"
" #RefreshBtn_contact_list_page\n"
"{\n"
"    background-color: #E59866;\n"
"}\n"
"\n"
"\n"
"#showRefreshBtn_page3:hover, #showRefreshBtn_page3:pressed,\n"
"#RefreshBtn_contact_list_page:hover, #RefreshBtn_contact_list_page:pressed\n"
"{\n"
"    background-color: #DC7633;\n"
"}\n"
"\n"
"/* Style for QCheckBox in main widget */\n"
"QCheckBox {\n"
"    spacing: 10px;\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 15px;\n"
"    height: 15px;\n"
"}\n"
"\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url(:/icon/icon/checkbox_unchecked.ico);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    image: url(:/icon/icon/checkbox_checked.ico);\n"
"}\n"
"\n"
"\n"
"/* Style for QFrame */\n"
"#resultFrame {\n"
"    border-radius: 5px;\n"
"    background-color: #fff;\n"
"}\n"
"\n"
"/* Style for QTableWidget */\n"
"#tableWidget_contact_list_page1 QHeaderView, #tableWidget_contact_list_page1   {\n"
"    border:0px;\n"
"    background-color:  #fff;\n"
"    border-radius:5px;\n"
"    text-align:left;\n"
"}\n"
"\n"
"#tableWidget_contact_list_page1 QHeaderView::section{\n"
"    font-family:\"Times New Roman\", \"Times\", \"serif\";\n"
"    font-size: 12px;\n"
"    font-weight: bold;\n"
"    text-align:left;\n"
"    border-radius:14px;\n"
"    border-bottom:1px solid #353535;\n"
"    border-top:1px solid #353535;\n"
"    padding-left: 10px;\n"
"}\n"
"\n"
"/* Style for QTableWidget */\n"
"#tableWidget_call_history_page3 QHeaderView, #tableWidget_call_history_page3   {\n"
"    border:0px;\n"
"    background-color:  #fff;\n"
"    border-radius:5px;\n"
"    text-align:left;\n"
"}\n"
"\n"
"#tableWidget_call_history_page3 QHeaderView::section{\n"
"    font-family:\"Times New Roman\", \"Times\", \"serif\";\n"
"    font-size: 12px;\n"
"    font-weight: bold;\n"
"    text-align:left;\n"
"    border-radius:14px;\n"
"    border-bottom:1px solid #353535;\n"
"    border-top:1px solid #353535;\n"
"    padding-left: 10px;\n"
"}\n"
"\n"
"#tableWidget_contact_list_page1::item:selected {\n"
"    background-color: #55aaff;\n"
"}\n"
"\n"
"#tableWidget_call_history_page3::item:selected {\n"
"    background-color: #55aaff;\n"
"}\n"
"\n"
"#tableWidget_contact_list_page1_page3::item{\n"
"    border-bottom:1px solid #d0c6ff;\n"
"    padding-left: 10px;\n"
"    color: black;\n"
"}\n"
"\n"
"#tableWidget_contact_list_page1::item{\n"
"    border-bottom:1px solid #d0c6ff;\n"
"    padding-left: 10px;\n"
"    color: black;\n"
"}\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.menuWidget = QtWidgets.QWidget(self.centralwidget)
        self.menuWidget.setMinimumSize(QtCore.QSize(200, 0))
        self.menuWidget.setMaximumSize(QtCore.QSize(200, 16777215))
        self.menuWidget.setObjectName("menuWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.menuWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 5)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.logoFrame = QtWidgets.QFrame(self.menuWidget)
        self.logoFrame.setMinimumSize(QtCore.QSize(0, 50))
        self.logoFrame.setMaximumSize(QtCore.QSize(16777215, 50))
        self.logoFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.logoFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.logoFrame.setObjectName("logoFrame")
        self.verticalLayout.addWidget(self.logoFrame)
        self.contact_list_leftmenu_pushButton = QtWidgets.QPushButton(self.menuWidget)
        self.contact_list_leftmenu_pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.contact_list_leftmenu_pushButton.setMaximumSize(QtCore.QSize(16777215, 35))
        
        self.contact_list_leftmenu_pushButton.setCheckable(True)
        self.contact_list_leftmenu_pushButton.setChecked(True)
        self.contact_list_leftmenu_pushButton.setAutoExclusive(True)
        self.contact_list_leftmenu_pushButton.setObjectName("contact_list_leftmenu_pushButton")
        self.verticalLayout.addWidget(self.contact_list_leftmenu_pushButton)
        self.add_new_contact_leftmenu_pushButton = QtWidgets.QPushButton(self.menuWidget)
        self.add_new_contact_leftmenu_pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.add_new_contact_leftmenu_pushButton.setMaximumSize(QtCore.QSize(16777215, 35))
        
        self.add_new_contact_leftmenu_pushButton.setCheckable(True)
        self.add_new_contact_leftmenu_pushButton.setAutoExclusive(True)
        self.add_new_contact_leftmenu_pushButton.setObjectName("add_new_contact_leftmenu_pushButton")
        self.verticalLayout.addWidget(self.add_new_contact_leftmenu_pushButton)
        """
        self.history_leftmenu_pushButton = QtWidgets.QPushButton(self.menuWidget)
        self.history_leftmenu_pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.history_leftmenu_pushButton.setMaximumSize(QtCore.QSize(16777215, 35))
        
        self.history_leftmenu_pushButton.setCheckable(True)
        self.history_leftmenu_pushButton.setAutoExclusive(True)
        self.history_leftmenu_pushButton.setObjectName("history_leftmenu_pushButton")
        self.verticalLayout.addWidget(self.history_leftmenu_pushButton)
        """
        spacerItem = QtWidgets.QSpacerItem(20, 574, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.logout_leftmenu_pushButton = QtWidgets.QPushButton(self.menuWidget)
        self.logout_leftmenu_pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.logout_leftmenu_pushButton.setMaximumSize(QtCore.QSize(16777215, 35))
        self.logout_leftmenu_pushButton.setStyleSheet("#logout_leftmenu_pushButton {\n"
"    color: #943e3e;\n"
"    text-align: center;\n"
"    padding-left: 0;\n"
"}\n"
"\n"
"#logout_leftmenu_pushButton:pressed {\n"
"    padding-left: 10px;\n"
"}\n"
"")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/icon/logout-48.ico"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.logout_leftmenu_pushButton.setIcon(icon4)
        self.logout_leftmenu_pushButton.setObjectName("logout_leftmenu_pushButton")
        self.verticalLayout.addWidget(self.logout_leftmenu_pushButton)
        self.gridLayout.addWidget(self.menuWidget, 0, 0, 1, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_4.setContentsMargins(50, 50, 50, 50)
        self.gridLayout_4.setVerticalSpacing(20)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setHorizontalSpacing(10)
        self.gridLayout_3.setVerticalSpacing(15)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.search_label_contact_list = QtWidgets.QLabel(self.page)
        self.search_label_contact_list.setObjectName("search_label_contact_list")
        self.gridLayout_3.addWidget(self.search_label_contact_list, 0, 0, 1, 1)
        self.search_lineEdit_contact_list = QtWidgets.QLineEdit(self.page)
        self.search_lineEdit_contact_list.setObjectName("search_lineEdit_contact_list")
        self.gridLayout_3.addWidget(self.search_lineEdit_contact_list, 0, 1, 1, 1)
        self.RefreshBtn_contact_list_page = QtWidgets.QPushButton(self.page)
        self.RefreshBtn_contact_list_page.setMinimumSize(QtCore.QSize(150, 30))
        self.RefreshBtn_contact_list_page.setMaximumSize(QtCore.QSize(150, 32))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/icon/redo-5-128.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.RefreshBtn_contact_list_page.setIcon(icon5)
        self.RefreshBtn_contact_list_page.setObjectName("RefreshBtn_contact_list_page")
        self.gridLayout_3.addWidget(self.RefreshBtn_contact_list_page, 0, 2, 1, 1)
        
        self.SearchBtn_contact_list_page = QtWidgets.QPushButton(self.page)#,clicked=lambda:self.call(MainWindow)
        self.SearchBtn_contact_list_page.setMinimumSize(QtCore.QSize(150, 30))
        self.SearchBtn_contact_list_page.setMaximumSize(QtCore.QSize(150, 32))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/icon/search-7-128.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SearchBtn_contact_list_page.setIcon(icon6)
        self.SearchBtn_contact_list_page.setObjectName("SearchBtn_contact_list_page")
        self.gridLayout_3.addWidget(self.SearchBtn_contact_list_page, 0, 3, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.resultFrame = QtWidgets.QFrame(self.page)
        self.resultFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.resultFrame.setObjectName("resultFrame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.resultFrame)
        self.gridLayout_2.setObjectName("gridLayout_2")


        self.tableWidget_contact_list_page1 = QtWidgets.QTableWidget(self.resultFrame)
        self.tableWidget_contact_list_page1.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tableWidget_contact_list_page1.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tableWidget_contact_list_page1.setShowGrid(False)
        self.tableWidget_contact_list_page1.setGridStyle(QtCore.Qt.NoPen)
        self.tableWidget_contact_list_page1.setObjectName("tableWidget_contact_list_page1")
        self.tableWidget_contact_list_page1.setColumnCount(5)

        header = self.tableWidget_contact_list_page1.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)

        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_contact_list_page1.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_contact_list_page1.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_contact_list_page1.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_contact_list_page1.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_contact_list_page1.setHorizontalHeaderItem(4, item)
        
        self.tableWidget_contact_list_page1.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget_contact_list_page1.horizontalHeader().setDefaultSectionSize(150)
        self.tableWidget_contact_list_page1.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_contact_list_page1.verticalHeader().setVisible(False)
        self.tableWidget_contact_list_page1.verticalHeader().setCascadingSectionResizes(False)
        self.gridLayout_2.addWidget(self.tableWidget_contact_list_page1, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.resultFrame, 1, 0, 1, 1)
        self.stackedWidget.addWidget(self.page)
        
        #page2
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_6.setContentsMargins(100, 50, 100, -1)
        self.gridLayout_6.setVerticalSpacing(20)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.ResetBtn_addnewcontact_page2 = QtWidgets.QPushButton(self.page_2)
        self.ResetBtn_addnewcontact_page2.setMinimumSize(QtCore.QSize(0, 30))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icon/x-mark-3-48.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ResetBtn_addnewcontact_page2.setIcon(icon)
        self.ResetBtn_addnewcontact_page2.setObjectName("ResetBtn_addnewcontact_page2")
        self.horizontalLayout.addWidget(self.ResetBtn_addnewcontact_page2)
        self.verifyContactBtn_addnewcontact_page2 = QtWidgets.QPushButton(self.page_2)
        self.verifyContactBtn_addnewcontact_page2.setMinimumSize(QtCore.QSize(0, 30))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/icon/user-48.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.verifyContactBtn_addnewcontact_page2.setIcon(icon1)
        self.verifyContactBtn_addnewcontact_page2.setObjectName("verifyContactBtn_addnewcontact_page2")
        self.horizontalLayout.addWidget(self.verifyContactBtn_addnewcontact_page2)
        self.AddContactBtn_addnewcontact_page2 = QtWidgets.QPushButton(self.page_2)
        self.AddContactBtn_addnewcontact_page2.setMinimumSize(QtCore.QSize(0, 30))
        icon2 = QtGui.QIcon() 
        icon2.addPixmap(QtGui.QPixmap(":/icon/icon/login-64.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.AddContactBtn_addnewcontact_page2.setIcon(icon2)
        self.AddContactBtn_addnewcontact_page2.setObjectName("AddContactBtn_addnewcontact_page2")
        self.horizontalLayout.addWidget(self.AddContactBtn_addnewcontact_page2)
        self.gridLayout_6.addLayout(self.horizontalLayout,3, 0, 1, 3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(50)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout_6.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setHorizontalSpacing(10)
        self.gridLayout_5.setVerticalSpacing(15)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.title_label_addnewcontact_page2 = QtWidgets.QLabel(self.page_2)
        self.title_label_addnewcontact_page2.setObjectName("title_label_addnewcontact_page2")
        self.gridLayout_5.addWidget(self.title_label_addnewcontact_page2, 0, 0, 1, 2)
        
        self.name_label_addnewcontact_page2 = QtWidgets.QLabel(self.page_2)
        self.name_label_addnewcontact_page2.setObjectName("name_label_addnewcontact_page2")
        self.gridLayout_5.addWidget(self.name_label_addnewcontact_page2, 1, 0, 1, 1)
        self.name_lineEdit_addnewcontact_page2 = QtWidgets.QLineEdit(self.page_2)
        self.name_lineEdit_addnewcontact_page2.setObjectName("name_lineEdit_addnewcontact_page2")
        self.gridLayout_5.addWidget(self.name_lineEdit_addnewcontact_page2, 1, 1, 1, 2)
        self.gmailid_label_addnewcontact_page2 = QtWidgets.QLabel(self.page_2)
        self.gmailid_label_addnewcontact_page2.setObjectName("gmailid_label_addnewcontact_page2")
        self.gridLayout_5.addWidget(self.gmailid_label_addnewcontact_page2, 2, 0, 1, 1)
        self.gmail_lineEdit_addnewcontact_page2 = QtWidgets.QLineEdit(self.page_2)
        self.gmail_lineEdit_addnewcontact_page2.setObjectName("gmail_lineEdit_addnewcontact_page2")
        self.gridLayout_5.addWidget(self.gmail_lineEdit_addnewcontact_page2, 2, 1, 1, 2)
        
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 2)
        spacerItem3 = QtWidgets.QSpacerItem(744, 426, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem3, 5, 0, 1, 2)
        self.stackedWidget.addWidget(self.page_2)
        
        """
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_8.setContentsMargins(100, 50, 100, -1)
        self.gridLayout_8.setVerticalSpacing(20)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setHorizontalSpacing(10)
        self.gridLayout_7.setVerticalSpacing(20)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_6 = QtWidgets.QLabel(self.page_3)
        self.label_6.setObjectName("label_6")
        self.gridLayout_7.addWidget(self.label_6, 0, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_7.addWidget(self.lineEdit_6, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.page_3)
        self.label_7.setObjectName("label_7")
        self.gridLayout_7.addWidget(self.label_7, 1, 0, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_7.addWidget(self.lineEdit_7, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.page_3)
        self.label_8.setObjectName("label_8")
        self.gridLayout_7.addWidget(self.label_8, 2, 0, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout_7.addWidget(self.lineEdit_8, 2, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.page_3)
        self.label_9.setObjectName("label_9")
        self.gridLayout_7.addWidget(self.label_9, 3, 0, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_7.addWidget(self.lineEdit_9, 3, 1, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 508, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_8.addItem(spacerItem4, 2, 1, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        """
        
        """
        self.page_page3 = QtWidgets.QWidget()
        self.page_page3.setObjectName("page_page3")
        self.gridLayout_4_page3 = QtWidgets.QGridLayout(self.page_page3)
        self.gridLayout_4_page3.setContentsMargins(50, 50, 50, 50)
        self.gridLayout_4_page3.setVerticalSpacing(20)
        self.gridLayout_4_page3.setObjectName("gridLayout_4")
        self.gridLayout_3_page3 = QtWidgets.QGridLayout()
        self.gridLayout_3_page3.setHorizontalSpacing(10)
        self.gridLayout_3_page3.setVerticalSpacing(15)
        self.gridLayout_3_page3.setObjectName("gridLayout_3_page3")
        self.search_label_page3 = QtWidgets.QLabel(self.page_page3)
        self.search_label_page3.setObjectName("search_label_page3")
        self.gridLayout_3_page3.addWidget(self.search_label_page3, 0, 0, 1, 1)
        self.search_lineEdit_page3 = QtWidgets.QLineEdit(self.page_page3)
        self.search_lineEdit_page3.setObjectName("search_lineEdit_page3")
        self.gridLayout_3_page3.addWidget(self.search_lineEdit_page3, 0, 1, 1, 1)
        self.showRefreshBtn_page3 = QtWidgets.QPushButton(self.page_page3)
        self.showRefreshBtn_page3.setMinimumSize(QtCore.QSize(150, 30))
        self.showRefreshBtn_page3.setMaximumSize(QtCore.QSize(150, 32))
        icon5_page3 = QtGui.QIcon()
        icon5_page3.addPixmap(QtGui.QPixmap(":/icon/icon/redo-5-128.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.showRefreshBtn_page3.setIcon(icon5_page3)
        self.showRefreshBtn_page3.setObjectName("showRefreshBtn_page3")
        self.gridLayout_3_page3.addWidget(self.showRefreshBtn_page3, 0, 2, 1, 1)
        
        self.showSearchBtn_page3 = QtWidgets.QPushButton(self.page_page3)
        self.showSearchBtn_page3.setMinimumSize(QtCore.QSize(150, 30))
        self.showSearchBtn_page3.setMaximumSize(QtCore.QSize(150, 32))
        icon6_page3 = QtGui.QIcon()
        icon6_page3.addPixmap(QtGui.QPixmap(":/icon/icon/search-7-128.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.showSearchBtn_page3.setIcon(icon6_page3)
        self.showSearchBtn_page3.setObjectName("showSearchBtn_page3")
        self.gridLayout_3_page3.addWidget(self.showSearchBtn_page3, 0, 3, 1, 1)
        self.gridLayout_4_page3.addLayout(self.gridLayout_3_page3, 0, 0, 1, 1)
        self.resultFrame_page3 = QtWidgets.QFrame(self.page_page3)
        self.resultFrame_page3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultFrame_page3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.resultFrame_page3.setObjectName("resultFrame_page3")
        self.gridLayout_2_page3 = QtWidgets.QGridLayout(self.resultFrame_page3)
        self.gridLayout_2_page3.setObjectName("gridLayout_2_page3")
        self.tableWidget_call_history_page3 = QtWidgets.QTableWidget(self.resultFrame_page3)
        self.tableWidget_call_history_page3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tableWidget_call_history_page3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tableWidget_call_history_page3.setShowGrid(False)
        self.tableWidget_call_history_page3.setGridStyle(QtCore.Qt.NoPen)
        #self.tableWidget_call_history_page3.setRowCount(2)
        self.tableWidget_call_history_page3.setObjectName("tableWidget_call_history_page3")
        self.tableWidget_call_history_page3.setColumnCount(6)
        item_page3 = QtWidgets.QTableWidgetItem()
        item_page3.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(0, item_page3)
        item_page3 = QtWidgets.QTableWidgetItem()
        item_page3.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(1, item_page3)
        item_page3 = QtWidgets.QTableWidgetItem()
        item_page3.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(2, item_page3)
        item_page3 = QtWidgets.QTableWidgetItem()
        item_page3.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(3, item_page3)
        item_page3 = QtWidgets.QTableWidgetItem()
        item_page3.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(4, item_page3)
        item_page3 = QtWidgets.QTableWidgetItem()
        self.tableWidget_call_history_page3.setHorizontalHeaderItem(5, item_page3)
        self.tableWidget_call_history_page3.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget_call_history_page3.horizontalHeader().setDefaultSectionSize(150)
        self.tableWidget_call_history_page3.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_call_history_page3.verticalHeader().setVisible(False)
        self.tableWidget_call_history_page3.verticalHeader().setCascadingSectionResizes(False)
        self.gridLayout_2_page3.addWidget(self.tableWidget_call_history_page3, 0, 0, 1, 1)
        self.gridLayout_4_page3.addWidget(self.resultFrame_page3, 1, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_page3)
        """
        
        
        
        #////////////////////
        
        
        
        
        self.gridLayout.addWidget(self.stackedWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def call(self,MainWindow):
        #self.app=waitingCallerRoom.MainWindowCV()
        #self.app.show()
        pass
        
        MainWindow.setVisible(False)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MeetU"))
        self.contact_list_leftmenu_pushButton.setText(_translate("MainWindow", "Contact List"))
        self.add_new_contact_leftmenu_pushButton.setText(_translate("MainWindow", "ADD New Contact"))
        #self.history_leftmenu_pushButton.setText(_translate("MainWindow", "History"))
        self.logout_leftmenu_pushButton.setText(_translate("MainWindow", "LogOut"))
        self.search_label_contact_list.setText(_translate("MainWindow", "Search"))
        self.RefreshBtn_contact_list_page.setText(_translate("MainWindow", "Refresh"))
        self.SearchBtn_contact_list_page.setText(_translate("MainWindow", "Search"))
        item = self.tableWidget_contact_list_page1.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ID"))
        item = self.tableWidget_contact_list_page1.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Name"))
        item = self.tableWidget_contact_list_page1.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "G-Mail"))
        item = self.tableWidget_contact_list_page1.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Call"))
        item = self.tableWidget_contact_list_page1.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Delete"))
        __sortingEnabled = self.tableWidget_contact_list_page1.isSortingEnabled()
        self.tableWidget_contact_list_page1.setSortingEnabled(False)
        self.tableWidget_contact_list_page1.setSortingEnabled(__sortingEnabled)
        
        self.verifyContactBtn_addnewcontact_page2.setText(_translate("MainWindow", "Verify User"))
        self.ResetBtn_addnewcontact_page2.setText(_translate("MainWindow", "Reset"))
        self.AddContactBtn_addnewcontact_page2.setText(_translate("MainWindow", "ADD"))
        self.title_label_addnewcontact_page2.setText(_translate("MainWindow", "ADD NEW CONTACT"))
        self.name_label_addnewcontact_page2.setText(_translate("MainWindow", "Name"))
        self.gmailid_label_addnewcontact_page2.setText(_translate("MainWindow", "G-Mail ID"))
        
        """
        self.search_label_page3.setText(_translate("MainWindow", "Search"))
        self.showRefreshBtn_page3.setText(_translate("MainWindow", "Refresh"))
        self.showSearchBtn_page3.setText(_translate("MainWindow", "Search"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(0)
        item_page3.setText(_translate("MainWindow", "ID"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(1)
        item_page3.setText(_translate("MainWindow", "Name"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(2)
        item_page3.setText(_translate("MainWindow", "Date"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(3)
        item_page3.setText(_translate("MainWindow", "Call Type"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(4)
        item_page3.setText(_translate("MainWindow", "G-Mail"))
        item_page3 = self.tableWidget_call_history_page3.horizontalHeaderItem(5)
        item_page3.setText(_translate("MainWindow", "Delete"))
        __sortingEnabled_page3 = self.tableWidget_call_history_page3.isSortingEnabled()
        self.tableWidget_call_history_page3.setSortingEnabled(False)
        self.tableWidget_call_history_page3.setSortingEnabled(__sortingEnabled_page3)
        """

from static import resource_rc