import neat
import pickle
import sys
from getopt import getopt, GetoptError
from smb_runner import Runner


def print_arg_help():
    print(
        'Usage: <replay.py>\n'
        '    -f, --file        Replay file to load (required)\n'
        '    -s, --file        State to load'
    )


def init_arguments():
    replay_file = None
    state = None
    argv = sys.argv[1:]  # parsing the argument list
    try:
        opts, args = getopt(argv, 'f:s:h', ['file=', 'state=', 'help'])
        if not opts:
            raise GetoptError('-f, --file option ist required')
        for option in opts:
            if option[0] == '--file' or option[0] == '-f':
                if option[1] == '':
                    raise GetoptError('empty --file option')
                replay_file = option[1]
            elif option[0] == '--state' or option[0] == '-s':
                if option[1] == '':
                    raise GetoptError('empty --state option')
                state = option[1]
            else:
                print_arg_help()
                sys.exit()
        if replay_file is None:
            raise GetoptError('--file option ist required')
    except GetoptError as optError:
        print('Something went wrong while interpreting cli options: ' + optError.msg)
        print_arg_help()
        sys.exit()

    return replay_file, state


def main():
    arg_replay_file, arg_state = init_arguments()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config/config-feedforward-smb'
    )

    with open(arg_replay_file, 'rb') as input_file:
        genome = pickle.load(input_file)

    net = neat.nn.RecurrentNetwork.create(genome, config)
    runner = Runner()
    runner.render_env = True
    runner.run(net.activate)


if __name__ == "__main__":
    main()
