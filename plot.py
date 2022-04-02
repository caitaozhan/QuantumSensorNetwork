import matplotlib.pyplot as plt
from logger import Logger


class Plot:

    plt.rcParams['font.size'] = 50
    plt.rcParams['lines.linewidth'] = 5
    
    @staticmethod
    def hillclimbing(scores: list, constant: float = None):
        '''
        Args:
            scores -- a list of scores, each score is one evaluation value during an iteration
            constant -- value from a guess
        '''
        constant = [constant for _ in range(len(scores))]
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))
        ax.plot(scores, label='Hill Climbing')
        ax.plot(constant, label='Guess')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discrimination Success Probability')
        ax.set_ylim([0.7, 0.9])
        fig.legend(loc='upper center', ncol=2)
        fig.savefig('tmp.png')

    @staticmethod
    def vary_priors(data):
        pass


def vary_priors():
    logs = ['result-tmp/vary-prior']
    data = Logger.read_log(logs)
    print(data)


if __name__ == '__main__':
    # scores = [0.7899917641851644, 0.832989, 0.845341, 0.852194, 0.857062, 0.859864, 0.860891, 0.860928, 0.861471, 0.861524, 0.861533, 0.861536, 0.861537, 0.861541, 0.861543, 0.861763, 0.861993, 0.862333, 0.862427, 0.86244, 0.862667, 0.862849, 0.862851, 0.862853, 0.862854, 0.862854, 0.862855, 0.862856, 0.862856, 0.862857, 0.862857, 0.862857, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.86286, 0.862944, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862955, 0.862989, 0.863003, 0.863005]
    # constant = 0.794494
    # Plot.hillclimbing(scores, constant)


    vary_priors()