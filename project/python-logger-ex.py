import sys
import getopt
import time
import gc
import random
import logging  # https://docs.python.org/2/howto/logging.html
import threading
import queue
import numpy as np
import math
from scipy import stats

import theano.tensor as T

from image.ann import Ann
import game.game_util as util
from game.visuals import GameWindow
import game.ai2048demo as demo

__plays_dir__ = "C:\\Users\\Andreas\\Documents\\NTNU\\2015 Semester 2\\AI Prog\\Project 3\\Module\\DeepLearning\\game\\"
__dim__ = 4
show_gameplay = False
logging.basicConfig(filename='dev-play-log.log', level=logging.INFO)
logging.info(
    "Datetime,Hidden layer spec,Learning rate,Activation function,Training cases,Epochs,Training time,Highest tile")


def update_gui(q):
    """
    The thread that updates the gui. Just add boards to the queue for schedule.
    :param q: Queue
    :return:
    """
    test_window = GameWindow()
    while True:
        item = q.get()
        test_window.update_view(item)


update_queue = queue.Queue()
gui_thread = threading.Thread(target=update_gui, args=[update_queue])
gui_thread.daemon = True
gui_thread.start()


# Code from mnist_basics.py by Valerij. Modified.


def load_flat_text_cases(filename, dir=__plays_dir__):
    f = open(dir + filename, "r")
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()
    x_l = list(map(int, lines[0]))
    x_t = [list(map(int, line)) for line in lines[1:]]
    return x_t, x_l


def _prepare(features):
    prepared = []

    for i in range(len(features)):
        prepared.append([])
        h_flat, h_mon = util.horizontal_utility(features[i])
        v_flat, v_mon = util.vertical_utility(features[i])

        h_mon_inc = max(0, h_mon)
        h_mon_dec = abs(min(0, h_mon))
        v_mon_inc = max(0, v_mon)
        v_mon_dec = abs(min(0, v_mon))
        greatest = max(h_mon_inc,
                       h_mon_dec,
                       v_mon_inc,
                       v_mon_dec)
        if greatest < 1:
            greatest = 1
        mon = [h_mon_inc / greatest,
               h_mon_dec / greatest,
               v_mon_inc / greatest,
               v_mon_dec / greatest]
        prepared[i] += mon

        # prepared[i] += horizontal_utility(features[i])
        # prepared[i] += vertical_utility(features[i])

        prepared[i] += h_flat, v_flat  # , max(0, h_mon), max(0, -h_mon), max(0, v_mon), max(0, -v_mon)
        greatest = max(h_flat, v_flat)
        if greatest < 1:
            greatest = 1
        flatness = [h_flat / greatest,
                    v_flat / greatest]
        # prepared[i] += flatness

        # for input in prepared:
        #     highest = 0
        #     for cell in input:
        #         if cell > highest:
        #             highest = cell
        #     for i in range(len(input)):
        #         input[i] /= highest if highest > 1 else 1

    return prepared


training_features = []
training_labels = []
training_files = ["f3-1024-4.txt", "f3-1024-3.txt", "f3-1024-2.txt", "f3-1024-5.txt"]

for file in training_files:
    tmp_f, tmp_l = load_flat_text_cases(file)
    training_features += _prepare(tmp_f)
    training_labels += tmp_l

pre_testing_features, testing_labels = load_flat_text_cases("f3-1024-4.txt")
testing_features = _prepare(pre_testing_features)

# averages_random = list
# averages_ai = list
# averages_p = list
# averages


def main(raw_args):
    hidden_layer_spec = [10]
    lr = 0.1
    af = T.nnet.sigmoid
    af_name = 'sigmoid'
    epochs = 20

    opts, args = getopt.getopt(raw_args[1:], "l:a:h:e:")
    for o, a in opts:
        if o == '-l':
            lr = float(a)
        elif o == '-e':
            epochs = int(a)
        elif o == '-a':
            if a == 'tanh':
                af = T.tanh
            elif a == 'sigmoid':
                af = T.nnet.sigmoid
            else:
                af = T.nnet.relu
            af_name = a
        elif o == '-h':
            # hidden_layer_spec = raw_args[i+1:]
            hidden_layer_spec = list(map(int, a.split()))

    brain = Ann(len(testing_features[0]), hidden_layer_spec, 4, lr, af)
    print("Building new ANN...")
    brain.build_ann()
    print("ANN built.\n\tActivation function", af_name, "\n\tLearning rate", lr, "\n\tHidden layer", hidden_layer_spec)
    # logging.info("New ANN built (%s)\n\tHidden layer: %s.\n\tLearning rate: %s\n\tActivation f: %s",
    #              time.asctime(),
    #              hidden_layer_spec,
    #              lr,
    #              af_name)

    errors, duration = brain.train(training_features, training_labels, epochs=epochs)
    print("Trained for", int(duration), "seconds.")
    # print(training_features[100:104])
    # print(training_labels[100:104])
    # print(testing_features[100:104])
    # print(testing_labels[100:104])
    print("Error from", errors[0], "to", errors[-1], ".")
    # logging.info("Trained.\n\tTraining cases: %s\n\tEpochs: %s\n\tSeconds: %s",
    #              len(training_features),
    #              epochs,
    #              int(duration))

    # test_cases = mnist.load_all_flat_cases('half_testing')
    # test_cases = mnist.load_flat_cases('demo_prep')
    test_output, error_rate = brain.test(testing_features, testing_labels)
    error_percent = 100 - error_rate * 100
    print("Error rate:", error_percent, "%")
    print("Fasit:\t\t", testing_labels[0:10], "...", testing_labels[-11:-1])
    print("Brain's:\t", test_output[0:10], "...", test_output[-11:-1])
    # logging.info("Tested.\n\tCases: %s\n\tError percent: %s",
    #              len(testing_features),
    #              format(error_percent, '.3f'))

    def play(games=1, randomize=False):
        high_tiles = []
        for i in range(games):
            # Starting to play...
            board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # board[random.randint(0, 15)] = 2 if random.random() > 0.1 else 4
            util.next_spawn(board)
            util.next_spawn(board)
            # game_window = GameWindow()
            # game_window.update_view(util.to_the_power(board))
            if show_gameplay:
                update_queue.put(util.to_the_power(board))
            if not randomize:
                brain_output = brain.test_one(_prepare([board])[0])
            else:
                brain_output = np.random.uniform(0.0, 1.0, 4)
            legal_move = True
            while legal_move:

                best_move = brain_output.argmax()
                if brain_output[best_move] <= -1:
                    break
                if best_move == 0:  # Up
                    legal_move = util.move_up(board)
                elif best_move == 1:  # Right
                    legal_move = util.move_right(board)
                elif best_move == 2:  # Down
                    legal_move = util.move_down(board)
                elif best_move == 3:  # Left
                    legal_move = util.move_left(board)

                if not legal_move:
                    # tmp = brain_output[:best_move] if best_move > 0 else []
                    # tmp[len(tmp):] = brain_output[best_move + 1:] if best_move < 3 else []
                    # brain_output = tmp
                    # np.delete(brain_output, brain_output.max())
                    brain_output[best_move] = -1
                    legal_move = True
                    continue

                # game_window.update_view(util.to_the_power(board))
                if show_gameplay:
                    update_queue.put(util.to_the_power(board))
                    time.sleep(0.1)

                util.next_spawn(board)

                # game_window.update_view(util.to_the_power(board))
                if show_gameplay:
                    update_queue.put(util.to_the_power(board))
                    time.sleep(0.05)

                if not randomize:
                    brain_output = brain.test_one(_prepare([board])[0])
                else:
                    brain_output = np.random.uniform(0.0, 1.0, 4)

                    # random_out = random.randint(0, 3)
            high_tiles.append(util.get_highest_tile(board))
            logging.info("%s,%s,%s,%s,%s,%s,%s,%s",
                         time.asctime(),
                         hidden_layer_spec,
                         lr,
                         af_name,
                         len(training_features),
                         epochs,
                         int(duration),
                         util.get_highest_tile(board))
            if show_gameplay:
                time.sleep(5)
        return high_tiles

    # random_play = play(2, True)
    #ai_play = play(1)
    random_play = play(50, True)
    ai_play = play(50)
    print("Random play:\n", random_play, "\nAI play:\n", ai_play)
    print()
    print(demo.welch(random_play, ai_play))
    #logging.info("%s", evaluation)


if __name__ == '__main__':
    networks = [['null', '-a', 'sigmoid', '-l', '0.1', '-h', '12 6', '-e', '20']]
    # network = [[]]
    if len(sys.argv) < 2:
        for network in networks:
            for i in range(1):
                main(network)
                # hidden_layer_spec = [10] lr = 0.1 af = T.nnet.sigmoid af_name = 'sigmoid' epochs = 20
                # gc.collect()
    else:
        main(sys.argv)
