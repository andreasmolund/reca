import csv
import getopt
import sys
import os

import numpy as np


def main(args):
    opts, args = getopt.getopt(args[1:],
                               'f:')
    prefix = '/home/andreas/Documents/GitHub/reca/results/japvow-110-20,20,20-20,20,20-part'

    for o, a in opts:
        if o == '-f':
            prefix = a
    index = prefix.rfind('/') + 1
    path = prefix[0:index]
    prefix = prefix[index:]

    prefixed = [filename for filename in os.listdir(path) if filename.startswith(prefix)]

    headers = ["Tot. fit time",
               "Fully correct seq.",
               "Mispredicted time steps","2 Fully correct seq.",
               "2 Mispredicted time steps",
               "3 Fully correct seq.",
               "3 Mispredicted time steps",
               "4 Fully correct seq.",
               "4 Mispredicted time steps"]
    row_values = [0] * len(headers)

    count = 0

    for filename in prefixed:
        with open(path + filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

            for row in reader:
                try:
                    for i, header_name in enumerate(headers):
                        row_values[i] += float(row[header_name])

                    count += 1
                except (TypeError, ValueError):
                    print "Yo!"

    for i in xrange(len(row_values)):
        row_values[i] = row_values[i] / count

    print row_values, len(prefixed), "a", count


if __name__ == '__main__':
    main(sys.argv)
