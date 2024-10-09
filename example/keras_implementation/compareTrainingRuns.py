import matplotlib.pyplot as plt
import dill as pickle


class History:
    def __init__(self, loss, title):
        self.loss = loss
        self.title = title


def compare_training_runs(histories):
    # Plot the data
    for history in histories:
        plt.plot(history.loss['val_loss'], label=history.title)

    plt.title('validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.yscale('log')
    plt.legend()


if __name__ == "__main__":
    histories = []
    for i in range(1, 8):
        with open(f'01_algoRuns/run_{i:02d}/step_0001/2_16_16_16_2/training_history-000.pkl', 'rb') as f:
            history = pickle.load(f)

        with open(f'01_algoRuns/run_{i:02d}/log.out') as f:
            func = f.readlines()[30].split(' ')[-1]

        histories.append(History(history, func))

        #with open(f'01_algoRuns/run_09/step_{i:04d}/2_16_16_16_2/training_history-000.pkl', 'rb') as f:
        #    history2 = History(pickle.load(f), "relu")

        #with open(f'01_algoRuns/run_10/step_{i:04d}/2_16_16_16_2/training_history-000.pkl', 'rb') as f:
        #    history3 = History(pickle.load(f), "sigmoid")

        #with open(f'01_algoRuns/run_11/step_{i:04d}/2_16_16_16_2/training_history-000.pkl', 'rb') as f:
        #    history4 = History(pickle.load(f), "linear")

        #compare_training_runs([history1, history2, history3, history4])
        compare_training_runs(histories)
        plt.savefig(f'training_comparison.png')
        plt.close()