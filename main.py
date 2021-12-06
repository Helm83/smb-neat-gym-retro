import neat
import pickle
import sys
from getopt import getopt, GetoptError
from smb_runner import Runner
from mvp_reporter import MvpReporter
from performance_reporter import PerformanceReporter


class NeatRunner:
    population = None
    workers = 1
    runners = []

    def eval_genomes_parallel(self, genome, config):
        if len(self.runners) < self.workers:
            self.runners.append(Runner(generation=self.population.generation))
        runner = self.runners[len(self.runners) - 1]
        net = neat.nn.RecurrentNetwork.create(genome, config)
        return runner.run(net.activate)

    def eval_genomes_single(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            runner = Runner(generation=self.population.generation)
            genome.fitness = runner.run(net.activate)


def print_arg_help():
    print(
        'Usage: <main.py>\n'
        '    -c, --checkpoint        Checkpoint file to load\n'
        '    -m                      Use MvpReporter to show the best genome of each generation\n'
        '    -w, --workers           Number of workers/threads to use for parallel execution'
    )


def init_arguments():
    # init argument defaults
    checkpoint = None
    mvp_reporter = False
    performance_file = None
    workers = 0
    argv = sys.argv[1:]  # parsing the argument list

    try:
        opts, args = getopt(argv, 'c:w:p:mh', ['checkpoint=', 'workers=', 'record-performance=', 'help'])
        for option in opts:
            if option[0] == '--checkpoint' or option[0] == '-c':
                if option[1] == '':
                    raise GetoptError('empty --checkpoint option')
                checkpoint = option[1]
            elif option[0] == '--record-performance' or option[0] == '-p':
                if option[1] == '':
                    raise GetoptError('empty --record-performance option')
                performance_file = option[1]
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

    return checkpoint, mvp_reporter, performance_file, workers


def main():
    arg_checkpoint, arg_mvp_reporter, arg_performance_file, arg_parallel_workers = init_arguments()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config-feedforward-smb'
    )

    if arg_checkpoint:
        checkpoint = neat.Checkpointer.restore_checkpoint(arg_checkpoint)
        population = neat.Population(config, (checkpoint.population, checkpoint.species, checkpoint.generation))
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(25, filename_prefix='neat-mario-cp-'))

    if arg_mvp_reporter:
        population.add_reporter(MvpReporter())

    if arg_performance_file:
        population.add_reporter(PerformanceReporter(arg_performance_file))

    runner = NeatRunner()
    runner.population = population

    if arg_parallel_workers > 0:
        runner.workers = arg_parallel_workers
        pe = neat.ParallelEvaluator(arg_parallel_workers, runner.eval_genomes_parallel)
        winner = population.run(pe.evaluate)
    else:
        winner = population.run(runner.eval_genomes_single)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
