import neat
import pickle
import sys
from getopt import getopt, GetoptError
from smb_runner import Runner
from mvp_reporter import MvpReporter


# TODO: join parallel_smb.py and single_worker_smb.py together,
#  since number of workers can also be optional as single worker
class ParallelRunner:
    population = None

    def eval_genomes(self, genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        runner = Runner(generation=self.population.generation)
        return runner.run(net.activate)


def print_arg_help():
    print(
        'Usage: <parallel_smb.py>\n'
        '    -c, --checkpoint        Checkpoint file to load\n'
        '    -m                      Use MvpReporter to show the best genome of each generation\n'
        '    -w, --workers           Number of workers/threads to use for parallel execution'
    )


def init_arguments():
    # init argument defaults
    checkpoint = None
    mvp_reporter = False
    workers = 6
    argv = sys.argv[1:]  # parsing the argument list

    try:
        opts, args = getopt(argv, 'c:w:mh', ['checkpoint=', 'workers=', 'help'])
        for option in opts:
            if option[0] == '--checkpoint' or option[0] == '-c':
                if option[1] == '':
                    raise GetoptError('empty --checkpoint option')
                checkpoint = option[1]
            elif option[0] == '-m':
                mvp_reporter = True
            elif option[0] == '--workers' or option[0] == '-w':
                try:
                    workers = int(option[1])
                except ValueError:
                    raise GetoptError('option -w expects an integer')
            else:
                print_arg_help()
                sys.exit()
    except GetoptError as optError:
        print('Something went wrong while interpreting cli options: ' + optError.msg)
        print_arg_help()
        sys.exit()

    return checkpoint, mvp_reporter, workers


def main():
    arg_checkpoint, arg_mvp_reporter, arg_parallel_workers = init_arguments()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config-feedforward-smb'
    )

    if arg_checkpoint:
        population = neat.Checkpointer.restore_checkpoint(arg_checkpoint)
        # population = neat.Population(config, (checkpoint.population, checkpoint.species, checkpoint.generation))
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(25, filename_prefix='neat-mario-cp-'))

    if arg_mvp_reporter:
        population.add_reporter(MvpReporter())

    runner = ParallelRunner()
    runner.population = population
    pe = neat.ParallelEvaluator(arg_parallel_workers, runner.eval_genomes)

    winner = population.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
