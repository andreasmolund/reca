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

        res1max = 0
        res1min = 32
        res1 = 0

        res2max = 0
        res2min = 32
        res2 = 0

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

                value = int(row['R2 correct'])
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
            except (TypeError, ValueError):
                print "Prolly LinAlgError in da house"

        print "I %d, R %d, rule %d" % (iterations, permutations, rule)
        print "Reservoir 1: %d" % res1
        print "Reservoir 2: %d" % res2


if __name__ == '__main__':
    # print sys.argv
    main(sys.argv)
    # main(['aienv',
    #       '-f',
    #       '/home/andreas/Documents/GitHub/semester-project-code/project/results/2016-11-25T233347.325693-bitmem2res-i16-r8-rule-150.csv'])
