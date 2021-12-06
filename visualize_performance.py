import matplotlib.pyplot as plt
import csv

with open('performance_recordings/performance_test_200_3.csv', 'r') as f:
    csv_reader = csv.reader(f)
    generations = []
    top_species = []
    for row in csv_reader:
        generations.append(csv_reader.line_num)
        top_species.append(row + ['0'] * (5 - len(row)))

fig, ax = plt.subplots()
for i in range(5):
    ax.plot(generations, [round(float(sp[i]), 2) for sp in top_species], label=f'Species No.{i+1}')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness progression of the top 5 species of each generation')
plt.legend()
plt.show()
