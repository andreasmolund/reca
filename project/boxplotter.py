import csv

import matplotlib.pyplot as plt
import numpy as np
path = '/home/andreas/Documents/GitHub/semester-project-code/project/results/'
file_name = 'bitmemorytask-2016-11-16T01:18:21.027681.csv'

flierprops = dict(marker='o', markerfacecolor='black', markersize=3,
                  linestyle='none')

with open("%s%s" % (path, file_name)) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    values = []
    for row in reader:
        value = row['Successful']
        values.append([int(value)])
    plt.boxplot(np.array(values), flierprops=flierprops)
    plt.show()

