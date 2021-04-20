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
        MainWindow.resize(1280, 1024)
        MainWindow.setMinimumSize(QtCore.QSize(1280, 1024))
        MainWindow.setMaximumSize(QtCore.QSize(1280, 1024))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1261, 971))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 100, 1231, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
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
        self.line.setGeometry(QtCore.QRect(10, 140, 1241, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.table_file_info = QtWidgets.QTableWidget(self.tab)
        self.table_file_info.setGeometry(QtCore.QRect(10, 180, 281, 371))
        self.table_file_info.setObjectName("table_file_info")
        self.table_file_info.setColumnCount(0)
        self.table_file_info.setRowCount(0)
        self.train_predictions_matrix = QtWidgets.QTableWidget(self.tab)
        self.train_predictions_matrix.setGeometry(QtCore.QRect(10, 560, 1231, 371))
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
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(10, 10, 1241, 80))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.classifier_type = QtWidgets.QComboBox(self.gridLayoutWidget_5)
        self.classifier_type.setObjectName("classifier_type")
        self.gridLayout_5.addWidget(self.classifier_type, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 1, 0, 1, 1)
        self.btn_select_input_data = QtWidgets.QPushButton(self.gridLayoutWidget_5)
        self.btn_select_input_data.setObjectName("btn_select_input_data")
        self.gridLayout_5.addWidget(self.btn_select_input_data, 0, 2, 1, 1)
        self.preprocess_type = QtWidgets.QComboBox(self.gridLayoutWidget_5)
        self.preprocess_type.setObjectName("preprocess_type")
        self.preprocess_type.addItem("")
        self.preprocess_type.addItem("")
        self.gridLayout_5.addWidget(self.preprocess_type, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 2, 0, 1, 1)
        self.path_to_data = QtWidgets.QLineEdit(self.gridLayoutWidget_5)
        self.path_to_data.setObjectName("path_to_data")
        self.gridLayout_5.addWidget(self.path_to_data, 0, 1, 1, 1)
        self.btn_open_path_to_data = QtWidgets.QPushButton(self.gridLayoutWidget_5)
        self.btn_open_path_to_data.setObjectName("btn_open_path_to_data")
        self.gridLayout_5.addWidget(self.btn_open_path_to_data, 0, 3, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 10, 1241, 80))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.selector_model = QtWidgets.QComboBox(self.gridLayoutWidget_4)
        self.selector_model.setObjectName("selector_model")
        self.gridLayout_4.addWidget(self.selector_model, 1, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 1, 0, 1, 1)
        self.path_to_test_data = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.path_to_test_data.setObjectName("path_to_test_data")
        self.gridLayout_4.addWidget(self.path_to_test_data, 0, 1, 1, 1)
        self.btn_open_path_to_test_data = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.btn_open_path_to_test_data.setObjectName("btn_open_path_to_test_data")
        self.gridLayout_4.addWidget(self.btn_open_path_to_test_data, 0, 3, 1, 1)
        self.btn_select_path_to_test_data = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.btn_select_path_to_test_data.setObjectName("btn_select_path_to_test_data")
        self.gridLayout_4.addWidget(self.btn_select_path_to_test_data, 0, 2, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_17.setObjectName("label_17")
        self.gridLayout_4.addWidget(self.label_17, 0, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_19.setObjectName("label_19")
        self.gridLayout_4.addWidget(self.label_19, 2, 0, 1, 1)
        self.preprocess_type_test = QtWidgets.QComboBox(self.gridLayoutWidget_4)
        self.preprocess_type_test.setEnabled(True)
        self.preprocess_type_test.setObjectName("preprocess_type_test")
        self.preprocess_type_test.addItem("")
        self.preprocess_type_test.addItem("")
        self.gridLayout_4.addWidget(self.preprocess_type_test, 2, 1, 1, 1)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(10, 100, 1241, 41))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btn_load_test_data = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        self.btn_load_test_data.setObjectName("btn_load_test_data")
        self.horizontalLayout_4.addWidget(self.btn_load_test_data)
        self.btn_test = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        self.btn_test.setObjectName("btn_test")
        self.horizontalLayout_4.addWidget(self.btn_test)
        self.test_predictions_matrix = QtWidgets.QTableWidget(self.tab_2)
        self.test_predictions_matrix.setGeometry(QtCore.QRect(10, 550, 1231, 371))
        self.test_predictions_matrix.setObjectName("test_predictions_matrix")
        self.test_predictions_matrix.setColumnCount(0)
        self.test_predictions_matrix.setRowCount(0)
        self.status_2 = QtWidgets.QLabel(self.tab_2)
        self.status_2.setGeometry(QtCore.QRect(10, 150, 761, 16))
        self.status_2.setObjectName("status_2")
        self.table_file_info_2 = QtWidgets.QTableWidget(self.tab_2)
        self.table_file_info_2.setGeometry(QtCore.QRect(10, 170, 281, 371))
        self.table_file_info_2.setObjectName("table_file_info_2")
        self.table_file_info_2.setColumnCount(0)
        self.table_file_info_2.setRowCount(0)
        self.label_20 = QtWidgets.QLabel(self.tab_2)
        self.label_20.setGeometry(QtCore.QRect(300, 170, 991, 371))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.label_20.setFont(font)
        self.label_20.setScaledContents(True)
        self.label_20.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_20.setObjectName("label_20")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 1291, 21))
        self.label_5.setObjectName("label_5")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 40, 1231, 109))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.output_png_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.output_png_data_path.setObjectName("output_png_data_path")
        self.gridLayout.addWidget(self.output_png_data_path, 2, 1, 1, 1)
        self.input_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.input_data_path.setObjectName("input_data_path")
        self.gridLayout.addWidget(self.input_data_path, 0, 1, 1, 1)
        self.btn_choose_output_bmp_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_output_bmp_data_path.setObjectName("btn_choose_output_bmp_data_path")
        self.gridLayout.addWidget(self.btn_choose_output_bmp_data_path, 1, 2, 1, 1)
        self.btn_choose_input_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_input_data_path.setObjectName("btn_choose_input_data_path")
        self.gridLayout.addWidget(self.btn_choose_input_data_path, 0, 2, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox.setMinimum(1)
        self.spinBox.setProperty("value", 20)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 3, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 3, 0, 1, 1)
        self.output_bmp_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.output_bmp_data_path.setObjectName("output_bmp_data_path")
        self.gridLayout.addWidget(self.output_bmp_data_path, 1, 1, 1, 1)
        self.btn_choose_output_png_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_choose_output_png_data_path.setObjectName("btn_choose_output_png_data_path")
        self.gridLayout.addWidget(self.btn_choose_output_png_data_path, 2, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 0, 1, 1)
        self.btn_open_input_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_open_input_data_path.setObjectName("btn_open_input_data_path")
        self.gridLayout.addWidget(self.btn_open_input_data_path, 0, 3, 1, 1)
        self.btn_open_output_bmp_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_open_output_bmp_data_path.setObjectName("btn_open_output_bmp_data_path")
        self.gridLayout.addWidget(self.btn_open_output_bmp_data_path, 1, 3, 1, 1)
        self.btn_open_output_png_data_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_open_output_png_data_path.setObjectName("btn_open_output_png_data_path")
        self.gridLayout.addWidget(self.btn_open_output_png_data_path, 2, 3, 1, 1)
        self.checkBox_convert_png = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.checkBox_convert_png.setChecked(False)
        self.checkBox_convert_png.setObjectName("checkBox_convert_png")
        self.gridLayout.addWidget(self.checkBox_convert_png, 3, 2, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.tab_3)
        self.progressBar.setGeometry(QtCore.QRect(10, 220, 1231, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 260, 1231, 671))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 1, 1, 1)
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
        self.table_input_info = QtWidgets.QTableWidget(self.gridLayoutWidget_2)
        self.table_input_info.setObjectName("table_input_info")
        self.table_input_info.setColumnCount(0)
        self.table_input_info.setRowCount(0)
        self.gridLayout_2.addWidget(self.table_input_info, 1, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 2, 1, 1)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 170, 1231, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_start_devider = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.btn_start_devider.setObjectName("btn_start_devider")
        self.horizontalLayout_2.addWidget(self.btn_start_devider)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_2.setEnabled(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.tab_4)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 10, 1241, 109))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.btn_choose_neiron_test_data_path = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.btn_choose_neiron_test_data_path.setObjectName("btn_choose_neiron_test_data_path")
        self.gridLayout_3.addWidget(self.btn_choose_neiron_test_data_path, 1, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 1, 0, 1, 1)
        self.btn_choose_neiron_train_data_path = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.btn_choose_neiron_train_data_path.setObjectName("btn_choose_neiron_train_data_path")
        self.gridLayout_3.addWidget(self.btn_choose_neiron_train_data_path, 0, 2, 1, 1)
        self.neiron_test_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.neiron_test_data_path.setObjectName("neiron_test_data_path")
        self.gridLayout_3.addWidget(self.neiron_test_data_path, 1, 1, 1, 1)
        self.neiron_train_data_path = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.neiron_train_data_path.setObjectName("neiron_train_data_path")
        self.gridLayout_3.addWidget(self.neiron_train_data_path, 0, 1, 1, 1)
        self.btn_open_neiron_train_data_path = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.btn_open_neiron_train_data_path.setObjectName("btn_open_neiron_train_data_path")
        self.gridLayout_3.addWidget(self.btn_open_neiron_train_data_path, 0, 3, 1, 1)
        self.btn_open_neiron_test_data_path = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.btn_open_neiron_test_data_path.setObjectName("btn_open_neiron_test_data_path")
        self.gridLayout_3.addWidget(self.btn_open_neiron_test_data_path, 1, 3, 1, 1)
        self.epoch = QtWidgets.QSpinBox(self.gridLayoutWidget_3)
        self.epoch.setSuffix("")
        self.epoch.setMinimum(1)
        self.epoch.setMaximum(1000)
        self.epoch.setProperty("value", 10)
        self.epoch.setObjectName("epoch")
        self.gridLayout_3.addWidget(self.epoch, 3, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 3, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_16.setObjectName("label_16")
        self.gridLayout_3.addWidget(self.label_16, 2, 0, 1, 1)
        self.btn_open_neiron_weights_path = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.btn_open_neiron_weights_path.setObjectName("btn_open_neiron_weights_path")
        self.gridLayout_3.addWidget(self.btn_open_neiron_weights_path, 2, 3, 1, 1)
        self.select_neiron_weights = QtWidgets.QComboBox(self.gridLayoutWidget_3)
        self.select_neiron_weights.setObjectName("select_neiron_weights")
        self.gridLayout_3.addWidget(self.select_neiron_weights, 2, 1, 1, 1)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 170, 1241, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btn_start_neiron_train = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.btn_start_neiron_train.setObjectName("btn_start_neiron_train")
        self.horizontalLayout_3.addWidget(self.btn_start_neiron_train)
        self.btn_neiron_test = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.btn_neiron_test.setObjectName("btn_neiron_test")
        self.horizontalLayout_3.addWidget(self.btn_neiron_test)
        self.btn_neiron_additional_train = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.btn_neiron_additional_train.setObjectName("btn_neiron_additional_train")
        self.horizontalLayout_3.addWidget(self.btn_neiron_additional_train)
        self.tableWidget = QtWidgets.QTableWidget(self.tab_4)
        self.tableWidget.setGeometry(QtCore.QRect(10, 210, 1241, 681))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(10, 130, 1241, 31))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.use_gpu = QtWidgets.QCheckBox(self.horizontalLayoutWidget_5)
        self.use_gpu.setObjectName("use_gpu")
        self.horizontalLayout_5.addWidget(self.use_gpu)
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_5)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_5.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_5)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_5.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_5)
        self.checkBox_3.setObjectName("checkBox_3")
        self.horizontalLayout_5.addWidget(self.checkBox_3)
        self.pushButton = QtWidgets.QPushButton(self.tab_4)
        self.pushButton.setGeometry(QtCore.QRect(1130, 910, 121, 23))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.class_selector = QtWidgets.QComboBox(self.tab_5)
        self.class_selector.setGeometry(QtCore.QRect(20, 30, 251, 22))
        self.class_selector.setObjectName("class_selector")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.class_selector.addItem("")
        self.label_21 = QtWidgets.QLabel(self.tab_5)
        self.label_21.setGeometry(QtCore.QRect(20, 10, 47, 13))
        self.label_21.setObjectName("label_21")
        self.cell = QtWidgets.QLabel(self.tab_5)
        self.cell.setGeometry(QtCore.QRect(290, 340, 381, 16))
        self.cell.setObjectName("cell")
        self.sings_table = QtWidgets.QTableWidget(self.tab_5)
        self.sings_table.setGeometry(QtCore.QRect(20, 60, 256, 871))
        self.sings_table.setObjectName("sings_table")
        self.sings_table.setColumnCount(0)
        self.sings_table.setRowCount(0)
        self.image_cell = QtWidgets.QLabel(self.tab_5)
        self.image_cell.setGeometry(QtCore.QRect(290, 30, 300, 300))
        self.image_cell.setAlignment(QtCore.Qt.AlignCenter)
        self.image_cell.setObjectName("image_cell")
        self.prev_cell = QtWidgets.QPushButton(self.tab_5)
        self.prev_cell.setGeometry(QtCore.QRect(290, 380, 75, 23))
        self.prev_cell.setObjectName("prev_cell")
        self.next_cell = QtWidgets.QPushButton(self.tab_5)
        self.next_cell.setGeometry(QtCore.QRect(400, 380, 75, 23))
        self.next_cell.setObjectName("next_cell")
        self.cell_class = QtWidgets.QLabel(self.tab_5)
        self.cell_class.setGeometry(QtCore.QRect(290, 360, 381, 16))
        self.cell_class.setObjectName("cell_class")
        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификатор"))
        self.btn_load_data.setText(_translate("MainWindow", "Загрузить данные в Систему"))
        self.btn_fit.setText(_translate("MainWindow", "Обучить"))
        self.status.setText(_translate("MainWindow", "Статус загрузки данных:"))
        self.label_4.setText(_translate("MainWindow", "Тут будут данные по точностям обученной модели"))
        self.label.setText(_translate("MainWindow", "Путь к файлу с данными:"))
        self.label_3.setText(_translate("MainWindow", "Способ классификации:"))
        self.btn_select_input_data.setText(_translate("MainWindow", "Выбрать файл"))
        self.preprocess_type.setItemText(0, _translate("MainWindow", "Стандартизация"))
        self.preprocess_type.setItemText(1, _translate("MainWindow", "Нормализация"))
        self.label_2.setText(_translate("MainWindow", "Способ предобработки данных:"))
        self.path_to_data.setText(_translate("MainWindow", "C:\\_Programming\\BloodClassifierClassic\\ProjectData\\Data\\train_multiclass.csv"))
        self.btn_open_path_to_data.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Обучение моделей (классика)"))
        self.label_18.setText(_translate("MainWindow", "Выберите модель для загрузки:"))
        self.path_to_test_data.setText(_translate("MainWindow", "C:\\_Programming\\BloodClassifierClassic\\ProjectData\\Data\\test_multiclass.csv"))
        self.btn_open_path_to_test_data.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.btn_select_path_to_test_data.setText(_translate("MainWindow", "Выбрать файл"))
        self.label_17.setText(_translate("MainWindow", "Путь к файлу с данными:"))
        self.label_19.setText(_translate("MainWindow", "Способ предобработки изображений(автовыбор при выборе модели):"))
        self.preprocess_type_test.setItemText(0, _translate("MainWindow", "Стандартизация"))
        self.preprocess_type_test.setItemText(1, _translate("MainWindow", "Нормализация"))
        self.btn_load_test_data.setText(_translate("MainWindow", "Загрузить данные в Систему"))
        self.btn_test.setText(_translate("MainWindow", "Тест"))
        self.status_2.setText(_translate("MainWindow", "Статус загрузки данных:"))
        self.label_20.setText(_translate("MainWindow", "Тут будут данные по точностям обученной модели"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Работа с обученными моделями (классика)"))
        self.label_5.setText(_translate("MainWindow", "Подготовщик данных работает с исходными данными в формате .BMP, которые рассортированы по пакам, название которых соответствует классу объекта. Для выделения признаков необходимо пользоваться программой 4Show."))
        self.label_7.setText(_translate("MainWindow", "Папка для сохранения данных в формате BMP:"))
        self.label_6.setText(_translate("MainWindow", "Путь к исходному набору данных:"))
        self.output_png_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\png_devided_data"))
        self.input_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\IshodData"))
        self.btn_choose_output_bmp_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.btn_choose_input_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_8.setText(_translate("MainWindow", "Процент тестовой выборки:"))
        self.output_bmp_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\bmp_devided_data"))
        self.btn_choose_output_png_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_12.setText(_translate("MainWindow", "Папка для сохранения данных в формате PNG:"))
        self.btn_open_input_data_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.btn_open_output_bmp_data_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.btn_open_output_png_data_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.checkBox_convert_png.setText(_translate("MainWindow", "Осуществлять конвертацию в PNG "))
        self.label_10.setText(_translate("MainWindow", "Данные в тестовом наборе"))
        self.label_9.setText(_translate("MainWindow", "Данные в исходном наборе"))
        self.label_11.setText(_translate("MainWindow", "Данные в тренировочном наборе"))
        self.btn_start_devider.setText(_translate("MainWindow", "Запуск разделения данных"))
        self.pushButton_2.setText(_translate("MainWindow", "Очистить форму"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Подготовка данных (деление на тестовую и обучающие выборки)"))
        self.btn_choose_neiron_test_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.label_14.setText(_translate("MainWindow", "Путь к тестовому датасету:"))
        self.btn_choose_neiron_train_data_path.setText(_translate("MainWindow", "Выбрать путь"))
        self.neiron_test_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\png_devided_data\\test"))
        self.neiron_train_data_path.setText(_translate("MainWindow", "C:\\_Programming\\_DataSets\\Multiclass\\png_devided_data\\train"))
        self.btn_open_neiron_train_data_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.btn_open_neiron_test_data_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.label_15.setText(_translate("MainWindow", "Количество эпох:"))
        self.label_13.setText(_translate("MainWindow", "Путь к обучающему датасету:"))
        self.label_16.setText(_translate("MainWindow", "Веса:"))
        self.btn_open_neiron_weights_path.setText(_translate("MainWindow", "Открыть в проводнике"))
        self.btn_start_neiron_train.setText(_translate("MainWindow", "Запустить обучение"))
        self.btn_neiron_test.setText(_translate("MainWindow", "Протестировать"))
        self.btn_neiron_additional_train.setText(_translate("MainWindow", "Дообучить модель"))
        self.use_gpu.setText(_translate("MainWindow", "Использование ГПУ"))
        self.checkBox.setText(_translate("MainWindow", "Тестировать каждую эпоху"))
        self.checkBox_2.setText(_translate("MainWindow", "Тестировать обучающую и тестовую выборку"))
        self.checkBox_3.setText(_translate("MainWindow", "Тестировать в разрезе каждого класса"))
        self.pushButton.setText(_translate("MainWindow", "Сохранить в CSV"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Нейросеть"))
        self.class_selector.setItemText(0, _translate("MainWindow", "All"))
        self.class_selector.setItemText(1, _translate("MainWindow", "Basophil"))
        self.class_selector.setItemText(2, _translate("MainWindow", "Blasts"))
        self.class_selector.setItemText(3, _translate("MainWindow", "Eosinophil"))
        self.class_selector.setItemText(4, _translate("MainWindow", "Lymphoblast"))
        self.class_selector.setItemText(5, _translate("MainWindow", "Lymphocyte"))
        self.class_selector.setItemText(6, _translate("MainWindow", "Megakaryocyte"))
        self.class_selector.setItemText(7, _translate("MainWindow", "Metamyelocyte"))
        self.class_selector.setItemText(8, _translate("MainWindow", "Monocyte"))
        self.class_selector.setItemText(9, _translate("MainWindow", "Myelocyte"))
        self.class_selector.setItemText(10, _translate("MainWindow", "Normoblasts"))
        self.class_selector.setItemText(11, _translate("MainWindow", "Plasma cell"))
        self.class_selector.setItemText(12, _translate("MainWindow", "Promyelocyte"))
        self.class_selector.setItemText(13, _translate("MainWindow", "Rod-shaped neutrophil"))
        self.class_selector.setItemText(14, _translate("MainWindow", "Segmented neutrophil"))
        self.label_21.setText(_translate("MainWindow", "Класс"))
        self.cell.setText(_translate("MainWindow", "Клетка:"))
        self.image_cell.setText(_translate("MainWindow", "Изображение"))
        self.prev_cell.setText(_translate("MainWindow", "Назад"))
        self.next_cell.setText(_translate("MainWindow", "Далее"))
        self.cell_class.setText(_translate("MainWindow", "Класс:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "База знаний"))
