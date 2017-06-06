import csv
import getopt
import sys
import os

import numpy as np


def main(args):
    opts, args = getopt.getopt(args[1:],
                               'f:c')
    prefix = '/home/andreas/Documents/GitHub/reca/rawresults/abdaf1337'
    header_set = False

    for o, a in opts:
        if o == '-f':
            prefix = a
    index = prefix.rfind('/') + 1
    path = prefix[0:index]
    prefix = prefix[index:]
    new_file_name = path + prefix + "s.csv"

    lines = []
    prefixed = [filename for filename in os.listdir(path) if filename.startswith(prefix)]

    bit_headers = ["Tot. fit time"
                   , "1 fully correct seq.", "1 mispredicted time steps"
                   , "2 fully correct seq.", "2 mispredicted time steps"
                   , "3 fully correct seq.", "3 mispredicted time steps"
                   # , "4 fully correct seq.", "4 mispredicted time steps"
                   # , "5 fully correct seq.", "5 mispredicted time steps"
                   ]
    jap_headers = ["Tot. fit time"
                   , "1 out of", "1 misclassif"
                   , "2 out of", "2 misclassif"
                   , "3 out of", "3 misclassif"
                   , "4 out of", "4 misclassif"
                   # , "5 out of", "5 misclassif"
                   ]

    # headers = bit_headers

    if not os.path.isfile(new_file_name):
        fout = open(new_file_name, "a")

        for filename in prefixed:
            with open(path + filename) as f:
                if header_set:
                    f.next()

                for line in f:
                    fout.write(line)

                    if not header_set:
                        header_set = True
        fout.close()
    else:
        print "File existed already:", new_file_name

    count = 0
    # row_values = [0] * len(headers)

    with open(new_file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            try:
                # for i, header_name in enumerate(headers):
                #     row_values[i] += float(row[header_name])

                count += 1
            except (TypeError, ValueError):
                print "Yo!"

    # for i in xrange(len(row_values)):
    #     row_values[i] = row_values[i] / count

    print "tot", count


if __name__ == '__main__':
    main(sys.argv)
