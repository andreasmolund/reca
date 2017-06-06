# Given a csvfile generated from "resultstats", this script strips it from unwanted information

import csv
import getopt
import sys
import numpy as np


def main(args):
    opts, args = getopt.getopt(args[1:],
                               'f:c')
    file_name = '/home/andreas/Documents/GitHub/reca/results/2017/collected/bittask-30,30,30-30,20,20.csv'

    for o, a in opts:
        if o == '-f':
            file_name = a
    index = file_name.rfind('/') + 1
    path = file_name[0:index]
    file_name = file_name[index:]
    new_file_name = path + file_name + "-stripped.csv"

    header_substring = "mean"
    header_substring2 = "correct"
    header_substring_postfix = "stddevneg"
    header_substring_postfix2 = "correct"
    lines = []
    n_rows = 0

    # headers = bit_headers

    with open(path + file_name) as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')

        headers = reader.fieldnames
        h1 = []
        h2 = []
        for header in headers:
            if header_substring in header \
                    and header_substring2 in header:
                h1.append(header)
            if header_substring_postfix in header \
                    and header_substring_postfix2 in header:
                h2.append(header)

        lines.append(h1)

        for row in reader:
            line = []
            for i in range(len(h1)):
                decimal = float(row[h1[i]])
                cell_text = "$%.2f\\pm%.2f$" % (decimal, decimal - float(row[h2[i]]))
                line.append(cell_text)

            n_rows += 1
            lines.append(line)

    new_lines = []

    init_line = ["Rule"]
    [init_line.append("Layer %d" % (l + 1)) for l in range(n_rows)]
    new_lines.append(init_line)

    for col in range(len(lines[0])):
        new_line = []
        for row in range(n_rows + 1):
            new_line.append(lines[row][col])
        new_lines.append(new_line)
    lines = new_lines

    fout = open(new_file_name, "a")
    for line in lines:
        fout.write(",".join(map(str, line)) + "\n")
    fout.close()


if __name__ == '__main__':
    main(sys.argv)
