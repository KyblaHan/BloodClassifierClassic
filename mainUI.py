# В данном файле описано управление основным интерфейсом программы и связи с остальными функцциями

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox
import PyQt5.QtCore

from UI.MainWindow import Ui_MainWindow  # импорт нашего сгенерированного файла
import sys
import classifier
import data_prep
from sklearn import metrics
import os
import neiron


# pyuic5 UI/MainWindow.ui -o UI/MainWindow.py


class mywindow(QtWidgets.QMainWindow):
    X = []
    y = []
    y_unic = []

    # начальная инициализация формы
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.tabWidget.setCurrentIndex(0)
        # ===1==================
        self.ui.btn_select_input_data.clicked.connect(self.btn_select_input_data_clicked)
        self.ui.btn_load_data.clicked.connect(self.btn_load_data_clicked)
        self.ui.btn_fit.clicked.connect(self.btn_fit_clicked)
        self.ui.btn_open_path_to_data.clicked.connect(self.btn_open_path_to_data_clicked)
        self.ui.btn_fit.setDisabled(True)
        for m in classifier.classifiers:
            self.ui.classifier_type.addItem(str(m))
        # ===2============================
        # ===3================================
        self.ui.btn_choose_input_data_path.clicked.connect(self.btn_choose_input_data_path_clicked)
        self.ui.btn_choose_output_bmp_data_path.clicked.connect(self.btn_choose_output_bmp_data_path_clicked)
        self.ui.btn_choose_output_png_data_path.clicked.connect(self.btn_choose_output_png_data_path_clicked)
        self.ui.btn_start_devider.clicked.connect(self.btn_start_devider_clicked)
        self.ui.btn_open_input_data_path.clicked.connect(self.btn_open_input_data_path_clicked)
        self.ui.btn_open_output_bmp_data_path.clicked.connect(self.btn_open_output_bmp_data_path_clicked)
        self.ui.btn_open_output_png_data_path.clicked.connect(self.btn_open_output_png_data_path_clicked)
        # ===4================================
        self.ui.btn_choose_neiron_train_data_path.clicked.connect(self.btn_choose_neiron_train_data_path_clicked)
        self.ui.btn_choose_neiron_test_data_path.clicked.connect(self.btn_choose_neiron_test_data_path_clicked)
        self.ui.btn_open_neiron_train_data_path.clicked.connect(self.btn_open_neiron_train_data_path_clicked)
        self.ui.btn_open_neiron_test_data_path.clicked.connect(self.btn_open_neiron_test_data_path_clicked)
        self.ui.btn_open_neiron_weights_path.clicked.connect(self.btn_open_neiron_weights_path_clicked)
        self.ui.btn_start_neiron_train.clicked.connect(self.btn_start_neiron_train_clicked)
        self.ui.btn_neiron_test.clicked.connect(self.btn_neiron_test_clicked)
        self.ui.btn_neiron_additional_train.clicked.connect(self.btn_neiron_additional_train_clicked)

    # ====Управление первой вкладкой====================================================
    def btn_open_path_to_data_clicked(self):
        path = self.ui.path_to_data.text()
        path = '\\'.join(path.split('\\')[:-1])
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_fit_clicked(self):
        expected, predicted = classifier.control_classifiers(self.X, self.y, self.ui.classifier_type.currentIndex(),
                                                             self.ui.preprocess_type.currentText())
        self.ui.label_4.setText(metrics.classification_report(expected, predicted))
        self.load_train_predictions_matrix_table(metrics.confusion_matrix(expected, predicted))

    def btn_select_input_data_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, "Выберите файл", filter="*.csv")
        self.ui.path_to_data.setText(file_path[0])

    def btn_load_data_clicked(self):

        if self.ui.preprocess_type.currentText() == "Стандартизация":
            preprocess_type = 2
            status_text = "стандартизованны"
        else:
            preprocess_type = 1
            status_text = "нормализованны"

        self.ui.btn_fit.setDisabled(False)

        self.X, self.y = classifier.load_and_preprocess_data(self.ui.path_to_data.text(), preprocess_type)

        self.load_info_table(classifier.get_load_file_stats(self.y))
        self.ui.status.setText("Статус загрузки данных: данные загружены и " + status_text)

    def load_train_predictions_matrix_table(self, data):
        self.ui.train_predictions_matrix.setColumnCount(len(self.y_unic))
        self.ui.train_predictions_matrix.setRowCount(len(self.y_unic))
        # заголовки для столбцов.
        self.ui.train_predictions_matrix.setHorizontalHeaderLabels(
            self.y_unic
        )
        # заголовки для строк.
        self.ui.train_predictions_matrix.setVerticalHeaderLabels(
            self.y_unic
        )
        row = 0
        for tup in data:
            col = 0
            for item in tup:
                cellinfo = QTableWidgetItem(str(item))
                # Только для чтения
                cellinfo.setFlags(
                    PyQt5.QtCore.Qt.ItemIsSelectable | PyQt5.QtCore.Qt.ItemIsEnabled
                )
                self.ui.train_predictions_matrix.setItem(row, col, cellinfo)
                col += 1
            row += 1

    def load_info_table(self, data):
        self.y_unic.clear()
        self.ui.table_file_info.setColumnCount(2)
        self.ui.table_file_info.setRowCount(len(data))
        # заголовки для столбцов.
        self.ui.table_file_info.setHorizontalHeaderLabels(
            ('Клетка', 'Количество')
        )
        row = 0
        for tup in data:
            col = 0
            for item in tup:
                cellinfo = QTableWidgetItem(item)
                # Только для чтения
                cellinfo.setFlags(
                    PyQt5.QtCore.Qt.ItemIsSelectable | PyQt5.QtCore.Qt.ItemIsEnabled
                )
                self.ui.table_file_info.setItem(row, col, cellinfo)
                if col == 0:
                    self.y_unic.append(item)
                col += 1
            row += 1

    # ====Управление второй вкладкой=================================================

    # ====Управление третьей вкладкой===================================================
    def btn_open_input_data_path_clicked(self):
        path = self.ui.input_data_path.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_open_output_bmp_data_path_clicked(self):
        path = self.ui.output_bmp_data_path.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_open_output_png_data_path_clicked(self):
        path = self.ui.output_png_data_path.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_choose_input_data_path_clicked(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями в формате BMP")
        self.ui.input_data_path.setText(file_path)
        self.load_table(self.ui.table_input_info, data_prep.get_folder_stats(self.ui.input_data_path.text()))

    def btn_choose_output_bmp_data_path_clicked(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения данных в формате BMP")
        self.ui.output_bmp_data_path.setText(file_path)

    def btn_choose_output_png_data_path_clicked(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения данных в формате PNG")
        self.ui.output_png_data_path.setText(file_path)

    def btn_start_devider_clicked(self):
        data_prep.devide_data(self.ui.input_data_path.text(), self.ui.output_bmp_data_path.text(),
                              self.ui.output_png_data_path.text(), self.ui.checkBox_convert_png.isChecked(),
                              int(self.ui.spinBox.value()), self.ui.progressBar
                              )

        self.load_table(self.ui.table_input_info,
                        data_prep.get_folder_stats(self.ui.input_data_path.text()))
        self.load_table(self.ui.table_test_info,
                        data_prep.get_folder_stats(self.ui.output_bmp_data_path.text() + "\\" + data_prep.TEST_FOLDER))
        self.load_table(self.ui.table_train_info,
                        data_prep.get_folder_stats(self.ui.output_bmp_data_path.text() + "\\" + data_prep.TRAIN_FOLDER))

    def load_table(self, table, data):
        table.setColumnCount(2)
        table.setRowCount(len(data))
        # заголовки для столбцов.
        table.setHorizontalHeaderLabels(
            ('Клетка', 'Количество')
        )
        row = 0
        for tup in data:
            col = 0
            for item in tup:
                cellinfo = QTableWidgetItem(item)
                # Только для чтения
                cellinfo.setFlags(
                    PyQt5.QtCore.Qt.ItemIsSelectable | PyQt5.QtCore.Qt.ItemIsEnabled
                )
                table.setItem(row, col, cellinfo)

                col += 1
            row += 1

    # ======================================4===========================================
    def btn_choose_neiron_train_data_path_clicked(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        self.ui.neiron_train_data_path.setText(file_path)

    def btn_choose_neiron_test_data_path_clicked(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        self.ui.neiron_test_data_path.setText(file_path)

    def btn_open_neiron_train_data_path_clicked(self):
        path = self.ui.neiron_train_data_path.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_open_neiron_test_data_path_clicked(self):
        path = self.ui.neiron_test_data_path.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_open_neiron_weights_path_clicked(self):
        path = "ProjectData//Weights//Neiron"
        path = os.path.realpath(path)
        os.startfile(path)

    def btn_start_neiron_train_clicked(self):
        print("!!")
        neiron.StartTrain(self.ui.neiron_train_data_path.text(), self.ui.use_gpu.isChecked(), self.ui.epoch.value())

    def btn_neiron_test_clicked(self):
        pass

    def btn_neiron_additional_train_clicked(self):
        pass


app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())
