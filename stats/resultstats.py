import csv
import os

import numpy as np

new_file_name = "/home/andreas/Documents/GitHub/reca/results/2017/collected/bittask-32-40.csv"

filenames = ["/home/andreas/Documents/GitHub/reca/results/2017/bittask-54-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-62-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-90-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-102-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-110-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-146-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-150-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-165-32-40-parts.csv",
             "/home/andreas/Documents/GitHub/reca/results/2017/bittask-195-32-40-parts.csv"
             ]

bit_headers = ["1 fully correct seq.", "1 mispredicted time steps"
               # , "2 fully correct seq.", "2 mispredicted time steps"
               # , "3 fully correct seq.", "3 mispredicted time steps"
               # , "4 fully correct seq.", "4 mispredicted time steps"
               # , "5 fully correct seq.", "5 mispredicted time steps"
               ]
jap_headers = ["1 misclassif"
    , "2 misclassif"
    , "3 misclassif"
    , "4 misclassif"
               # , "5 misclassif"
               ]

# outof_headers = [1480, 1480, 1480, 1480, 370]

headers = bit_headers
headers_per_layer = 2

file_ids = []
n_files = len(filenames)
n_headers = len(headers)
n_layers = n_headers / headers_per_layer
new_rows = []
file_stat = []

for filename in filenames:
    file_id = ""
    count = 0
    with open(filename) as csvfile:
        all_file_numbers = []
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            row_values = []
            try:
                for header_name in headers:
                    row_values.append(float(row[header_name]))
                    # row_values.append(100.0 - 100 * float(row[header_name]) / outof_headers[i])
                count += 1
            except (TypeError, ValueError):
                print "Yo! Halt!"
            all_file_numbers.append(row_values)
            if file_id == "":
                file_id = "%s" % row["Rule"]

        all_file_numbers = np.array(all_file_numbers).transpose().tolist()
        stat = []
        for population in all_file_numbers:
            # Should be as many populations as there are "headers"
            mean = np.mean(population)
            stddev = np.std(population)
            stddevneg = -stddev
            amin = np.amin(population)
            qs = (25, 50, 75)
            q1, q2, q3 = np.percentile(population, qs)
            amax = np.amax(population)

            stat.append([mean, mean + stddev, mean + stddevneg, amin, q1, q2, q3, amax])

        stat = np.array(stat).reshape((n_layers, headers_per_layer * len(stat[0]))).tolist()
        file_stat.append(stat)
    file_ids.append(file_id)

statistics_names = ["mean", "stddev", "stddevneg", "amin", "q1", "q2", "q3", "amax"]
n_statistics = len(statistics_names)

# file_stat = np.array(file_stat).reshape((n_layers, n_files * 8 * row_len / n_layers))
file_stat = np.array(file_stat).transpose((1, 2, 0))
file_stat = file_stat.reshape((n_layers, n_files * headers_per_layer * n_statistics))
file_stat = file_stat.tolist()

init_new_row = "Layer"
for group_name in headers[0:headers_per_layer]:
    for statistics_name in statistics_names:
        for file_id in file_ids:
            init_new_row += ",%s-%s-%s" % (group_name.replace(" ", "").replace(".", ""),
                                           file_id.replace(",", "-"),
                                           statistics_name)

if not os.path.isfile(new_file_name):
    fout = open(new_file_name, "a")
    fout.write(init_new_row + "\n")
    for i, layer_data in enumerate(file_stat):
        fout.write("%d," % (i + 1) + ",".join(map(str, layer_data)) + "\n")

    fout.close()
    print "Success. Wrote to file."
else:
    print "File existed already:", new_file_name
