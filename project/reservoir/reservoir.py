import marshal as dumper
import random as rn
from multiprocessing import Process, Queue
import sys
import ca.util as cutil

dump_path = "tmp/"
prefix = "dumpprocess"
filetype = "dump"


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, matter, iterations, verbose=1):
        """

        :param matter: the CA object
        :param iterations: the number of iterations
        :param verbose: 1 prints basic information, 2 prints the
        """
        self.matter = matter
        self.iterations = iterations
        self.verbose = verbose

        print "%d iterations" % self.iterations

    def transform(self, configurations, n_processes=4):
        """Lets the reservoir digest each of the configurations.
        No training here.

        :param configurations: a list of initial configurations,
                               and the quantity of it must be divisible on n_processes
        :param n_processes: divides the work into processes for better performance
        :return: a list in which each element is the output of the translation of each configuration
        """

        if self.verbose == 0:
            sys.stdout.write("Transforming... ")
            sys.stdout.flush()
        elif self.verbose > 1:
            print "Input:"
            for i, c in enumerate(configurations):
                cutil.print_config_1dim(c, postfix="(%d)" % i)

        # Starting processes to distribute work
        out_q = Queue()
        configs_per_thread = len(configurations) / n_processes
        processes = []
        for n in xrange(n_processes):
            start = n * configs_per_thread
            end = start + configs_per_thread
            process = Process(target=self._transform_subset,
                              args=(configurations[start:end],
                                    self.iterations,
                                    self.matter.copy(),
                                    start,
                                    out_q))
            processes.append(process)
            process.start()

        # Collecting data from the different processes
        outputs = [None] * len(configurations)
        for _ in xrange(n_processes):
            process_i = out_q.get()  # Process index; from where the process began
            in_file = open("%s%s%d.%s" % (dump_path, prefix, process_i, filetype), 'rb')

            for i in xrange(configs_per_thread):
                outputs[process_i + i] = dumper.load(in_file)

                if self.verbose > 1:
                    print "Output (%d):" % (process_i + i)
                    op = outputs[process_i + i]
                    casize = len(op) / self.iterations
                    for iteration in xrange(self.iterations):
                        cutil.print_config_1dim(op[iteration*casize:iteration*casize+casize])

            in_file.close()

        # Wait for all sub computations to finish
        for process in processes:
            process.join()

        if self.verbose == 1:
            sys.stdout.write("Done\n")
            sys.stdout.flush()
        elif self.verbose > 1:
            print "Concats:"
            for i, o in enumerate(outputs):
                cutil.print_config_1dim(o, postfix="(%d)" % i)

        return outputs

    @staticmethod
    def _transform_subset(configurations, iterations, matter, nr, queue):
        out_file = open("%s%s%d.%s" % (dump_path, prefix, nr, filetype), 'wb')
        for i in xrange(len(configurations)):
            concat = []
            config = configurations[i]
            # concat.extend(config)  # To include the initial configuration

            # Iterate
            for _ in xrange(iterations):
                new_config = matter.step(config)
                # Concatenating this new configuration to the vector
                concat.extend(new_config)
                config = new_config
            dumper.dump(concat, out_file)
        out_file.close()
        queue.put(nr)

    @staticmethod
    def _concat(elements, n_elements_per_concat):
        raise NotImplementedError("Not impl yet")


def normalized_addition(state_vector, input_vector):
    """Entries with value 2 (i.e. 1 + 1) become 1,
    with value 0 stay 0 (i.e. 0 + 0),
    and with value 1 (i.e. 0 + 1) are decided randomly.

    :param state_vector:
    :param input_vector:
    :return:
    """
    return 0


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

