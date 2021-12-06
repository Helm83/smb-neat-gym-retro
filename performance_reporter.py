from neat.reporting import BaseReporter
from neat.six_util import iterkeys
import csv


class PerformanceReporter(BaseReporter):
    file = ''

    def __init__(self, file):
        self.file = file

    def end_generation(self, config, population, species_set):
        if self.file == '':
            print('performance-file-path not set')
            return

        sids = list(iterkeys(species_set.species))
        fitnesses = []
        for sid in sids:
            fitnesses.append(species_set.species[sid].fitness or 0.0)

        fitnesses = sorted(fitnesses, reverse=True)[:5]

        if '/' not in self.file:
            self.file = 'performance_recordings/' + self.file
        with open(self.file, 'a') as f:
            write = csv.writer(f)
            write.writerow(fitnesses)
