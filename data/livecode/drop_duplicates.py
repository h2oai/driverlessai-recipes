# Remove duplicate rows by grouping the same rows,
# sorting them and then selecting first (1) or last (-1)
# row from each group
from datatable import by, sort, f

# column(s) that define duplicate rows:
by_columns = ['county','state','fips']
# column(s) (e.g. time) to sort rows within each group
sort_columns = ['date']

# select last row from each group
# use 1 instead of -1 to select first row from each group
return X[-1, :, by(by_columns), sort(sort_columns)]
