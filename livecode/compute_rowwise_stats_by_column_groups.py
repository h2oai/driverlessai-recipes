# Compute row-wise aggregates, usually applicable when dataset contains time
# series data as enumerated columns, e.g. "PAY_1, PAY_2, ..., PAY_6" in
# the Kaggle Credit Card Default dataset.
#
# Specification:
# Inputs:
#   X: datatable - primary data set with one or more groups of time series columns
# Parameters:
#   columns: list of column name prefixes - each column prefix defines a column group to compute rowwise aggregates:
#            sum, mean, standard deviation, max, min, rowfirst, rowlast. E.g Kaggle Credit Card Default dataset column
#            names are "PAY_AMT", "BILL_AMT", and "PAY_"
#   ranges: pairs of values that defines range 'from' and 'to' corresponding to each column group,
#           e.g. Kaggle Credit Card Default dataset ranges could be (1, 6), (1, 6), (2, 6)
# Output:
#   dataset augmented with computed rowwise statistics for each column group
from datatable import f, update, rowsum, rowmean, rowsd, rowmax, rowmin, rowfirst, rowlast

columns = ["PAY_AMT", "BILL_AMT", "PAY_"]
ranges = [(1, 6), (1, 6), (2, 6)]

ts_col_lists = [[(col + "%d") % (week) for week in range(desde, hasta)] for (col, (desde, hasta)) in
                zip(columns, ranges)]

ts_col_slices = [((col + "%d") % (desde), (col + "%d") % (hasta)) for (col, (desde, hasta)) in zip(columns, ranges)]

for c, s in zip(columns, ts_col_slices):
    update_map = {c + "_sum": rowsum(f[s[0]:s[1]]),
                  c + "_mean": rowmean(f[s[0]:s[1]]),
                  c + "_sd": rowsd(f[s[0]:s[1]]),
                  c + "_max": rowmax(f[s[0]:s[1]]),
                  c + "_min": rowmin(f[s[0]:s[1]]),
                  c + "_first": rowfirst(f[s[0]:s[1]]),
                  c + "_last": rowlast(f[s[0]:s[1]])
                  }
    X[:, update(**update_map)]

return {"CreditCard-train-aug.csv": X}