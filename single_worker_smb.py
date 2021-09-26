import neat
import pickle
from smb_runner import Runner
from mvp_reporter import MvpReporter


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        runner = Runner()
        genome.fitness = runner.run(net.activate)


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-smb')

    checkpoint = neat.Checkpointer.restore_checkpoint('neat-mario-cp')
    p = neat.Population(config, (checkpoint.population, checkpoint.species, checkpoint.generation))

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(MvpReporter())

    winner = p.run(eval_genomes)

    with open('winner_single_worker_smb.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
