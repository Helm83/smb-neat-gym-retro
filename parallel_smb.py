import neat
import pickle
import sys
from getopt import getopt, GetoptError
from smb_runner import Runner
from mvp_reporter import MvpReporter


# TODO: add Checkpoints and MVPReporter as arguments instead of (un)commenting them
def eval_genomes(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    runner = Runner()
    return runner.run(net.activate)


def print_arg_help():
    msg = (
        'Usage: <parallel_smb.py>\n'
        '    -c, --checkpoint        Checkpoint file to load\n'
        '    -m                      Use MvpReporter to show the best genome of each generation\n'
        '    -w, --workers           Number of workers/threads to use for parallel execution'
    )
    print(msg)


def main():
    arg_checkpoint = None
    arg_mvp_reporter = False
    arg_parallel_workers = 6
    argv = sys.argv[1:]  # parsing the argument list

    try:
        opts, args = getopt(argv, 'c:w:mh', ['checkpoint=', 'workers=', 'help'])
        for option in opts:
            if option[0] == '--checkpoint' or option[0] == '-c':
                if option[1] == '':
                    raise GetoptError('empty --checkpoint option')
                arg_checkpoint = option[1]
            elif option[0] == '-m':
                arg_mvp_reporter = True
            elif option[0] == '--workers' or option[0] == '-w':
                try:
                    arg_parallel_workers = int(option[1])
                except ValueError:
                    print('-w option expects an integer')
                    return
            else:
                print_arg_help()
    except GetoptError as optError:
        print('Something went wrong while interpreting cli options: ' + optError.msg)
        return

    if arg_mvp_reporter:
        print('Using MvpReporter')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-smb')

    if arg_checkpoint:
        checkpoint = neat.Checkpointer.restore_checkpoint(arg_checkpoint)
        p = neat.Population(config, (checkpoint.population, checkpoint.species, checkpoint.generation))
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(25, filename_prefix='neat-mario-cp-'))

    if arg_mvp_reporter:
        p.add_reporter(MvpReporter())

    pe = neat.ParallelEvaluator(arg_parallel_workers, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
