import classifier as cs

path_to_data_glob = "Data//cells_all.csv"

X,y = cs.load_and_preprocess_data(path_to_data_glob,2)

cs.logistic_regression(X,y)
