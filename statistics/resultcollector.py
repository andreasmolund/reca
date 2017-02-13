import csv
import getopt
import sys

from numpy.linalg.linalg import LinAlgError


def main(args):
    opts, args = getopt.getopt(args[1:],
                               'f:')
    filename = ''

    for o, a in opts:
        if o == '-f':
            filename = a

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

        rule = -1
        iterations = -1
        permutations = -1

        count = 0

        res1max = 0
        res1min = 32
        res1 = 0
        res1misbits = 0

        res2max = 0
        res2min = 32
        res2 = 0
        res2misbits = 0

        for row in reader:
            try:
                value = int(row['R1 correct'])
                if value == 32:
                    res1 += 1
                if value > res1max:
                    res1max = value
                if value < res1min:
                    res1min = value

                res2 += int(row['Point (success)'])
                res1misbits += int(row['R1 wrong bits'])

                value = int(row['R2 correct'])
                res2misbits += int(row['R2 wrong bits'])
                if value > res2max:
                    res2max = value
                if value < res2min:
                    res2min = value

                if rule == -1:
                    rule = int(row['Rule'])
                if iterations == -1:
                    iterations = int(row['I'])
                if permutations == -1:
                    permutations = int(row['R'])

                count += 1
            except (TypeError, ValueError):
                taiesvjewijvgf = 0

        print "(%d,%d,%d)\t\t1:%s,%d\t\t2:%s,%d" % (iterations,
                                                    permutations,
                                                    rule,
                                                    "%s/%s" % (res1, count),
                                                    res1misbits,
                                                    "%s/%s" % (res2, count),
                                                    res2misbits)


if __name__ == '__main__':
    main(sys.argv)
