import neat
import pickle
from smb_runner import Runner


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-smb')

    p = neat.Population(config)

    with open('winner.pkl', 'rb') as input_file:
        genome = pickle.load(input_file)

    net = neat.nn.RecurrentNetwork.create(genome, config)
    runner = Runner()
    runner.render_env = True
    runner.run(net.activate)


if __name__ == "__main__":
    main()
