from neat.reporting import BaseReporter
from neat.nn import RecurrentNetwork
from smb_runner import Runner


class MvpReporter(BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        net = RecurrentNetwork.create(best_genome, config)
        runner = Runner()
        runner.render_env = True
        runner.render_ai_viewport = True
        runner.run(net.activate)
