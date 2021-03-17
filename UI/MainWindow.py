# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1342, 1068)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1321, 1011))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.formLayoutWidget = QtWidgets.QWidget(self.tab)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1291, 81))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.path_to_data = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.path_to_data.setObjectName("path_to_data")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.path_to_data)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.classifier_type = QtWidgets.QComboBox(self.formLayoutWidget)
        self.classifier_type.setObjectName("classifier_type")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.classifier_type)
        self.preprocess_type = QtWidgets.QComboBox(self.formLayoutWidget)
        self.preprocess_type.setObjectName("preprocess_type")
        self.preprocess_type.addItem("")
        self.preprocess_type.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.preprocess_type)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 100, 1291, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_select_input_data = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_select_input_data.setObjectName("btn_select_input_data")
        self.horizontalLayout.addWidget(self.btn_select_input_data)
        self.btn_load_data = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_load_data.setObjectName("btn_load_data")
        self.horizontalLayout.addWidget(self.btn_load_data)
        self.btn_fit = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_fit.setObjectName("btn_fit")
        self.horizontalLayout.addWidget(self.btn_fit)
        self.status = QtWidgets.QLabel(self.tab)
        self.status.setGeometry(QtCore.QRect(10, 160, 761, 16))
        self.status.setObjectName("status")
        self.line = QtWidgets.QFrame(self.tab)
        self.line.setGeometry(QtCore.QRect(10, 140, 1291, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.table_file_info = QtWidgets.QTableWidget(self.tab)
        self.table_file_info.setGeometry(QtCore.QRect(10, 180, 281, 371))
        self.table_file_info.setObjectName("table_file_info")
        self.table_file_info.setColumnCount(0)
        self.table_file_info.setRowCount(0)
        self.train_predictions_matrix = QtWidgets.QTableWidget(self.tab)
        self.train_predictions_matrix.setGeometry(QtCore.QRect(10, 560, 1291, 411))
        self.train_predictions_matrix.setObjectName("train_predictions_matrix")
        self.train_predictions_matrix.setColumnCount(0)
        self.train_predictions_matrix.setRowCount(0)
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(300, 180, 991, 371))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_4.setFont(font)
        self.label_4.setScaledContents(True)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 1291, 21))
        self.label_5.setObjectName("label_5")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 40, 1281, 109))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 1, 0, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox.setMinimum(1)
        self.spinBox.setProperty("value", 20)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 3, 1, 1, 1)
        self.btn_choose_output_bmp_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_output_bmp_data_path.setObjectName("btn_choose_output_bmp_data_path")
        self.gridLayout.addWidget(self.btn_choose_output_bmp_data_path, 1, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 3, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 0, 1, 1)
        self.output_bmp_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.output_bmp_data_path.setObjectName("output_bmp_data_path")
        self.gridLayout.addWidget(self.output_bmp_data_path, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.btn_choose_input_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_input_data_path.setObjectName("btn_choose_input_data_path")
        self.gridLayout.addWidget(self.btn_choose_input_data_path, 0, 2, 1, 1)
        self.input_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.input_data_path.setObjectName("input_data_path")
        self.gridLayout.addWidget(self.input_data_path, 0, 1, 1, 1)
        self.output_png_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.output_png_data_path.setObjectName("output_png_data_path")
        self.gridLayout.addWidget(self.output_png_data_path, 2, 1, 1, 1)
        self.btn_choose_output_png_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_output_png_data_path.setObjectName("btn_choose_output_png_data_path")
        self.gridLayout.addWidget(self.btn_choose_output_png_data_path, 2, 2, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.tab_3)
        self.progressBar.setGeometry(QtCore.QRect(10, 220, 1281, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 260, 1281, 651))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 0, 1, 1)
        self.table_input_info = QtWidgets.QTableWidget(self.gridLayoutWidget_2)
        self.table_input_info.setObjectName("table_input_info")
        self.table_input_info.setColumnCount(0)
        self.table_input_info.setRowCount(0)
        self.gridLayout_2.addWidget(self.table_input_info, 1, 0, 1, 1)
        self.table_test_info = QtWidgets.QTableWidget(self.gridLayoutWidget_2)
        self.table_test_info.setObjectName("table_test_info")
        self.table_test_info.setColumnCount(0)
        self.table_test_info.setRowCount(0)
        self.gridLayout_2.addWidget(self.table_test_info, 1, 1, 1, 1)
        self.table_train_info = QtWidgets.QTableWidget(self.gridLayoutWidget_2)
        self.table_train_info.setObjectName("table_train_info")
        self.table_train_info.setColumnCount(0)
        self.table_train_info.setRowCount(0)
        self.gridLayout_2.addWidget(self.table_train_info, 1, 2, 1, 1)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 170, 1281, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_convert_png = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.checkBox_convert_png.setChecked(False)
        self.checkBox_convert_png.setObjectName("checkBox_convert_png")
        self.horizontalLayout_2.addWidget(self.checkBox_convert_png)
        self.btn_start_devider = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.btn_start_devider.setObjectName("btn_start_devider")
        self.horizontalLayout_2.addWidget(self.btn_start_devider)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1342, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификатор"))
        self.label.setText(_translate("MainWindow", "Путь к файлу с данными:"))
        self.path_to_data.setText(_translate("MainWindow", "C:\\_Programming\\BloodClassifierClassic\\Data\\cells_all.csv"))
        self.label_3.setText(_translate("MainWindow", "Способ классификации:"))
        self.preprocess_type.setItemText(0, _translate("MainWindow", "Стандартизация"))
        self.preprocess_type.setItemText(1, _translate("MainWindow", "Нормализация"))
        self.label_2.setText(_translate("MainWindow", "Способ предобработки данных:"))
        self.btn_select_input_data.setText(_translate("MainWindow", "Выбрать файл"))
        self.btn_load_data.setText(_translate("MainWindow", "Загрузить данные в Систему"))
        self.btn_fit.setText(_translate("MainWindow", "Обучить"))
        self.status.setText(_translate("MainWindow", "Статус загрузки данных:"))
        self.label_4.setText(_translate("MainWindow", "Тут будут данные по точностям обученной модели"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Обучение моделей"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Работа с обученными моделями"))
        self.label_5.setText(_translate("MainWindow", "Подготовщик данных работает с исходными данными в формате .BMP, которые рассортированы по пакам, название которых соответствует классу объекта. Для выделения признаков необходимо пользоваться программой 4Show."))
        self.label_7.setText(_translate("MainWindow", "Папка для сохранения данных в формате BMP:"))
        self.btn_choose_output_bmp_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_8.setText(_translate("MainWindow", "Процент тестовой выборки:"))
        self.label_12.setText(_translate("MainWindow", "Папка для сохранения данных в формате PNG:"))
        self.output_bmp_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\bmp_devided_data"))
        self.label_6.setText(_translate("MainWindow", "Путь к исходному набору данных:"))
        self.btn_choose_input_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.input_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\IshodData"))
        self.output_png_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\png_devided_data"))
        self.btn_choose_output_png_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_10.setText(_translate("MainWindow", "Данные в тестовом наборе"))
        self.label_11.setText(_translate("MainWindow", "Данные в тренировочном наборе"))
        self.label_9.setText(_translate("MainWindow", "Данные в исходном наборе"))
        self.checkBox_convert_png.setText(_translate("MainWindow", "Осуществлять конвертацию в PNG"))
        self.btn_start_devider.setText(_translate("MainWindow", "Запуск разделения данных"))
        self.pushButton_2.setText(_translate("MainWindow", "Очистить форму"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Подготовка данных (деление на тестовую и обучающие выборки)"))
