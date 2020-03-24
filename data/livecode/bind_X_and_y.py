# dataset with target
y_file_name = "./tmp/877bbf3a-6557-11ea-a946-0242ac110002/class_f50k_X_y_train.csv.1584123720.3251045.bin" 

target_col = "y" # target column name

y = dt.fread(y_file_name)

if y.shape[0] != X.shape[0]:
	raise ValueError("Datasets must have equal number of rows")

X.cbind(y[target_col])

return X 
