import matplotlib.pyplot as plt


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
        fig, ax = plt.subplots(1, 1, figsize=(40, 25))
        ax.plot(scores, label='Hill Climbing')
        ax.plot(constant, label='Guess')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discrimination Success Probability')
        fig.legend(loc='upper center')
        fig.savefig('tmp.png')
