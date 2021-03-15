from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
import PyQt5.QtCore

from UI.MainWindow import Ui_MainWindow  # импорт нашего сгенерированного файла
import sys
import classifier
from sklearn import metrics
# pyuic5 UI/MainWindow.ui -o UI/MainWindow.py


class mywindow(QtWidgets.QMainWindow):
    X = []
    y = []
    y_unic = []

    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btn_select_input_data.clicked.connect(self.btn_select_input_data_clicked)
        self.ui.btn_load_data.clicked.connect(self.btn_load_data_clicked)
        self.ui.btn_fit.clicked.connect(self.btn_fit_clicked)

        for m in classifier.methods:
            self.ui.classifier_type.addItem(m)

    def btn_fit_clicked(self):
        if self.ui.classifier_type.currentText() == "LogisticRegression":
            expected,predicted = classifier.logistic_regression(self.X, self.y)

        self.ui.label_4.setText(metrics.classification_report(expected, predicted))
        self.load_train_predictions_matrix_table(metrics.confusion_matrix(expected, predicted))



    def btn_select_input_data_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, "Выберите файл",filter="*.csv")
        # print(file_path)
        self.ui.path_to_data.setText(file_path[0])

    def btn_load_data_clicked(self):

        if self.ui.preprocess_type.currentText() == "Стандартизация":
            preprocess_type = 2
            status_text = "стандартизованны"
        else:
            preprocess_type = 1
            status_text = "нормализованны"


        self.X, self.y = classifier.load_and_preprocess_data(self.ui.path_to_data.text(), preprocess_type)
        self.y_unic = list(set(self.y))
        print(self.y_unic)

        self.load_info_table(classifier.get_load_file_stats(self.y))
        self.ui.status.setText("Статус загрузки данных: данные загружены и " + status_text)



    def load_train_predictions_matrix_table(self,data):
        self.ui.train_predictions_matrix.setColumnCount(len(self.y_unic))
        self.ui.train_predictions_matrix.setRowCount(len(self.y_unic))
        # заголовки для столбцов.
        self.ui.train_predictions_matrix.setHorizontalHeaderLabels(
            set(self.y_unic)
        )
        # заголовки для строк.
        self.ui.train_predictions_matrix.setVerticalHeaderLabels(
            set(self.y_unic)
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


    def load_info_table(self,data):
        self.ui.table_file_info.setColumnCount(2)
        self.ui.table_file_info.setRowCount(len(data))
        # заголовки для столбцов.
        self.ui.table_file_info.setHorizontalHeaderLabels(
            ('Клетка', 'Количество клеток')
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
                col += 1
            row += 1

app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())
