# Module to generate configurations corresponding to different problems
# with labels/classes/fasit

import random as rn

import numpy as np
import random as rand


# np.random.seed(20161112)
import time


def japanese_vowels():

    # aetrain = np.zeros((9,  # Men
    #                     30,  # Blocks
    #                     12), dtype='int8')  # Coefficients
    aetrainblocks = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    aetestblocks = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    train_file_name = 'datasets/japvowels/ae.train'
    test_file_name = 'datasets/japvowels/ae.test'
    training_sets, training_labels = read_ae_file(train_file_name, aetrainblocks)
    testing_sets, testing_labels = read_ae_file(test_file_name, aetestblocks)

    return training_sets, training_labels, testing_sets, testing_labels


def read_ae_file(aefile_name, block_sizes):
    aefile = open(aefile_name, 'r')
    ae = []
    aelabels = []
    eof = False
    man_i = 0
    block_i = 0
    consecutive_empty_lines = 0
    block = []
    labelblock = []
    while not eof:
        line = aefile.readline().rstrip(' \n')
        if line == '':
            consecutive_empty_lines += 1
            if consecutive_empty_lines > 1:
                eof = True
            else:
                # Done with the block
                ae.append(np.array(block))
                aelabels.append(np.array(labelblock))
                block = []
                labelblock = []

                # Next block
                block_i += 1

                abdaf = block_sizes[man_i]
                # If no more blocks for this man
                if block_i == abdaf:
                    block_i = 0
                    man_i += 1
        else:
            coefficients = [float(feature) for feature in line.split(' ')]
            block.append(coefficients)

            label = [0b0] * 9
            label[man_i] = 1
            # label = man_i + 1

            labelblock.append(label)
            consecutive_empty_lines = 0
    aefile.close()
    return ae, aelabels

japanese_vowels_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def memory_task_5_bit(quantity, distractor_period):
    """

    :param quantity: the number of tasks to generate
    :param bits:
    :param distractor_period:
    :return: tasks, labels
    """
    tasks = []
    labels = []
    bits = 5

    for i in xrange(quantity):
        y = 2  # y being 0, 1, 2 means which of the output nodes y1, y2, y3 that is activated, respectively
        y1 = 0
        y2 = 0
        y3 = 1
        task = []
        set_labels = []
        for t in xrange(bits + distractor_period + bits):
            if t < bits:
                bit = (i & 2 ** t) / 2 ** t
                a1 = - bit + 1
                a2 = bit
            else:
                a1 = 0
                a2 = 0

            if bits <= t != bits + distractor_period - 1:
                # Distractor signal
                a3 = 1
            else:
                a3 = 0

            if t == bits + distractor_period - 1:
                # Cue signal
                a4 = 1
            else:
                a4 = 0

            task.append(np.asarray([a1, a2, a3, a4]))

            if t >= bits + distractor_period:
                y1 = task[t - (bits + distractor_period)][0]
                y2 = 1 - y1
                y3 = 0
                y = 0 if y1 == 1 else 1

            # if i == 7:
            #     print "%d %d %d %d -> %d" % (a1, a2, a3, a4, y)
            set_labels.append([y1, y2, y3])  # For multiple output nodes
            # set_labels.append(y)
        tasks.append(task)
        labels.append(set_labels)
    return np.array(tasks, dtype='int8'), np.array(labels, dtype='int8')


def memory_task_n_bit(dimensions, n_memory_time_steps, quantity, distractor_period):
    """
    :param dimensions: e.g. 5 for 20 bit task
    :param n_memory_time_steps: e.g. 10 for 2 bit task
    :param quantity:
    :param distractor_period:
    :return:
    """
    done_hashes = []
    tasks = []
    labels = []
    i = 0
    n_bits_distr = n_memory_time_steps + distractor_period
    if quantity > dimensions**n_memory_time_steps:
        raise ValueError("Illegal quantity. Must be <= %d" % dimensions**n_memory_time_steps)
    while i < quantity:
        n = 0
        task = []
        label = []
        for t in xrange(2 * n_memory_time_steps + distractor_period):
            if t < n_memory_time_steps:
                position = rand.randint(0, dimensions - 1)
                input_vector = [0] * (dimensions + 2)
                input_vector[position] = 1
                n += position * dimensions ** t
                output_vector = [0] * dimensions + [1, 0]
            elif t < n_bits_distr:
                if t == n_bits_distr - 1:
                    # Cue
                    input_vector = [0] * dimensions + [0, 1]
                else:
                    input_vector = [0] * dimensions + [1, 0]
                output_vector = [0] * dimensions + [1, 0]
            else:
                input_vector = [0] * dimensions + [1, 0]
                output_vector = task[t - n_bits_distr]
            task.append(input_vector)
            label.append(output_vector)
        if n in done_hashes:
            # Already existing task
            i -= 1
        else:
            tasks.append(task)
            labels.append(label)
            done_hashes.append(n)
        i += 1
    return np.array(tasks, dtype='int8'), np.array(labels, dtype='int8')

# tmp_20_tasks, tmp_20_labels = memory_task_n_bit(5, 10, 3, 2)
# for tsk, lbl in zip(tmp_20_tasks, tmp_20_labels):
#     for input_v, output_v in zip(tsk, lbl):
#         print "%s\t%s" % (input_v, output_v)
#     print '\n'


def temporal_parity(quantity, size, window_size=2, delay=0):
    tasks = []
    labels = []
    for q in xrange(quantity):
        task = []
        label = []

        bit_stream = np.random.randint(2, size=size, dtype='int8')

        for i in xrange(size + delay):
            from_i = i
            to_i = i + window_size
            task_element = bit_stream[from_i:to_i]
            if to_i > size:
                # Adding zeros if we have moved beyond the original bit_stream
                task_element = np.append(task_element, np.zeros(min(to_i - size, window_size), dtype='int8'))

            task.append(task_element.tolist())
            label_element = False if from_i < delay else sum(task[i - delay]) % 2 == 1
            if q == 1:
                print task_element, " --> ", label_element
            label_element = [1] if label_element else [0]
            label.append(label_element)
        tasks.append(task)
        labels.append(label)
    return tasks, labels


def density(quantity, size=5, on_probability=0.5):
    """Generates initial configurations for the density classification,
    along with correct classifications

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        zeros = 0
        for i in xrange(size):
            state = 1 if rn.random() < on_probability else 0
            if state == 1:
                ones += 1
            else:
                zeros += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones > zeros else 0)
    return configs, labels


def parity(quantity, size=4):
    """Generates initial configurations for the parity problem,
    along with the corresponding solutions

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        for i in xrange(size):
            state = rn.randint(0, 1)
            if state == 1:
                ones += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones % 2 == 0 else 0)
    return configs, labels
