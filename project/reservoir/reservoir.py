import marshal as dumper
import random as rn
from multiprocessing import Process, Queue

prefix = "dumpprocess"
filetype = "dump"


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, matter, iterations, verbose=False):
        """

        :param matter: the CA object
        :param iterations: the number of iterations
        :param verbose: set to true if you want to print the configurations
        """
        self.matter = matter
        self.iterations = iterations
        self.verbose = verbose

        print "%d iterations" % self.iterations

    @staticmethod
    def _transform_subset(configurations, iterations, matter, nr, queue):
        outputs = []
        for i in xrange(len(configurations)):

            config = configurations[i]
            concat = []

            # Iterate
            for _ in xrange(iterations):
                new_config = matter.step(config)
                # Concatenating this new configuration to the vector
                concat.extend(new_config)
                config = new_config
            outputs.append(concat)
        ouf = open("%s%d.%s" % (prefix, nr, filetype), 'wb')
        dumper.dump(outputs, ouf)
        ouf.close()
        queue.put(nr)

    def transform(self, configurations):
        """Lets the reservoir digest each of the configurations.
        No training here.

        :param configurations: a list of initial configurations
        :return: a list in which each element is the output of the translation of each configuration
        """

        out_q = Queue()
        n_processes = 4
        configs_per_thread = len(configurations) / n_processes
        processes = []
        for n in xrange(n_processes):
            start = n * configs_per_thread
            end = start + configs_per_thread
            process = Process(target=self._transform_subset,
                              args=(configurations[start:end],
                                    self.iterations,
                                    self.matter.copy(),
                                    n,
                                    out_q))
            processes.append(process)
            process.start()

        outputs = [None] * len(configurations)
        for _ in xrange(n_processes):
            item = out_q.get()
            inf = open("%s%d.%s" % (prefix, item, filetype), 'rb')
            readouts = dumper.load(inf)
            inf.close()
            for i, readout in enumerate(readouts):
                outputs[configs_per_thread * item + i] = readout

        # Wait for all sub computations to finish
        for process in processes:
            process.join()

        return outputs


def make_random_mapping(input_size, input_area, input_offset=0):
    """Generates a pseudo-random mapping from inputs to outputs.
    The encoding stage.

    :param input_size: the size that the inputs come in
    :param input_area: the area/size that the inputs are to be mapped to
    :param input_offset: a number if an offset is wanted, default 0
    :return: an array of mapped indexes
    """
    input_indexes = []
    for i in xrange(input_size):
        # Going through all states in the reservoir
        # Might be possible to improve
        index = rn.randint(0, input_area - 1)
        while index in input_indexes:
            index = rn.randint(0, input_area - 1)
        input_indexes.append(index)
    return [i + input_offset for i in input_indexes]

