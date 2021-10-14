import neat
import pickle
from smb_runner import Runner
from mvp_reporter import MvpReporter


# TODO: add Checkpoints and MVPReporter as arguments instead of (un)commenting them
def eval_genomes(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    runner = Runner()
    return runner.run(net.activate)


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-smb')

    # checkpoint = neat.Checkpointer.restore_checkpoint('neat-mario-cp')
    # p = neat.Population(config, (checkpoint.population, checkpoint.species, checkpoint.generation))
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(25, filename_prefix='neat-mario-cp-'))
    p.add_reporter(MvpReporter())

    pe = neat.ParallelEvaluator(6, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
