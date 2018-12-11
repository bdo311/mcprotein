
import latticeproteins as lp
# need to use myenv2 python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
# source activate myenv2
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import pandas as pd

codon_to_aa = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
    "UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",}
    
    
aa_to_codons = defaultdict(lambda: [])
for c in codon_to_aa:
    aa_to_codons[codon_to_aa[c]].append(c)

def mutate_codon(codon):
    codon = list(codon)
    pos = np.random.choice([0,1,2])
    bases = ['A', 'C', 'U', 'G']
    bases.remove(codon[pos])
    codon[pos] = np.random.choice(bases)
    return(''.join(codon))

# first we only accept mutations if they increase the fitness, using the same target configuration
# later we will consider different acceptance probabilities under MH
# may want to do branching trajectories in the future

def make_mutation_by_fitness(seq_mrna, fitness, conf):
    mutation = [0, ""]

    while True:         
        new_seq_mrna = deepcopy(seq_mrna)
        
        # choose one codon to make the mutation
        pos = np.random.choice(range(len(seq_mrna)))
        new_codon = mutate_codon(seq_mrna[pos])

        old_aa = codon_to_aa[seq_mrna[pos]]
        new_aa = codon_to_aa[new_codon]

        if new_aa != old_aa:  # nonsynonymous 
            # for adaptive we are allowing synonymous changes
            # for purifying we are not allowing synonymous changes
            if new_aa == 'STOP': continue
        
            new_seq_mrna[pos] = new_codon
            new_seq_pro = [codon_to_aa[c] for c in new_seq_mrna]
            new_fitness = lattice.fracfolded(new_seq_pro, target=conf)

            if new_fitness >= fitness:  # only take same or improved energy
                mutation = [pos, ''.join([old_aa, new_aa])]
                return((new_seq_mrna, new_fitness, mutation, pos, new_codon)) 

def make_mutation_by_purifying_fitness(seq_mrna, fitness, conf, delta=0.02):
    mutation = [0, ""]

    while True:         
        new_seq_mrna = deepcopy(seq_mrna)
        
        # choose one codon to make the mutation
        pos = np.random.choice(range(len(seq_mrna)))
        new_codon = mutate_codon(seq_mrna[pos])

        old_aa = codon_to_aa[seq_mrna[pos]]
        new_aa = codon_to_aa[new_codon]

        #if new_aa != old_aa:  # nonsynonymous 
            # for adaptive we are allowing synonymous changes
            # for purifying we are allowing synonymous changes
        if new_aa == 'STOP': continue
    
        new_seq_mrna[pos] = new_codon
        new_seq_pro = [codon_to_aa[c] for c in new_seq_mrna]
        new_fitness = lattice.fracfolded(new_seq_pro, target=conf)

        if np.abs(new_fitness - fitness) < delta:
            mutation = [pos, ''.join([old_aa, new_aa])]
            return((new_seq_mrna, new_fitness, mutation, pos, new_codon)) 

# by fitness
def get_trajectory_by_fitness_goal(seq_mrna, conf, goal_fitness=0.9):
    fitness = lattice.fracfolded(seq, target=conf)
    mutations = []
    fitnesses = []
    sequences = [[codon_to_aa[c] for c in seq_mrna]]
    mutation_positions = []
    mutation_new_codons = []
    
    repeat = 0
    while fitness < goal_fitness:
        seq_mrna, new_fitness, mutation, pos, new_codon = make_mutation_by_fitness(seq_mrna, fitness, conf)
        
        if new_fitness > fitness:
            repeat = 0
        else:
            repeat += 1
        if repeat > 50: break
            
        new_seq = [codon_to_aa[c] for c in seq_mrna]
        mutations.append(mutation)
        fitnesses.append(new_fitness)
        # fitness = new_fitness not including this line because we want a goal fitness
        sequences.append(new_seq)

        mutation_positions.append(pos)
        mutation_new_codons.append(new_codon)
        #print(''.join(new_seq), new_fitness)
    
        
    return(fitnesses, mutation_positions, mutation_new_codons)

def get_trajectory_by_purifying_fitness(seq_mrna, conf, fitness, steps=20):
    mutations = []
    fitnesses = []
    sequences = [[codon_to_aa[c] for c in seq_mrna]]
    mutation_positions = []
    mutation_new_codons = []
    
    for i in range(steps):
        seq_mrna, new_fitness, mutation, pos, new_codon = make_mutation_by_purifying_fitness(seq_mrna, fitness, conf, delta=0.0045)
            
        new_seq = [codon_to_aa[c] for c in seq_mrna]
        mutations.append(mutation)
        fitnesses.append(new_fitness)
        #fitness = new_fitness
        sequences.append(new_seq)

        mutation_positions.append(pos)
        mutation_new_codons.append(new_codon)
        #print(''.join(new_seq), new_fitness)
        
        #new_fitness = fitness
        
    return(fitnesses, mutation_positions, mutation_new_codons)

seq_length = 20
temperature = 1.0
lattice = lp.LatticeThermodynamics.from_length(seq_length, 1.0)

#seq = ['A', 'N', 'S', 'H', 'G', 'K', 'Y', 'L', 'F', 'I', 'E', 'R', 'I', 'L', 'E', 'K', 'V', 'T', 'I', 'K']  # for adaptive
#seq = ['L', 'M', 'L', 'R', 'R', 'K', 'V', 'F', 'D', 'F', 'Q', 'W', 'M', 'W', 'M', 'G', 'F', 'Q', 'K', 'I']  # for purifying
seq = ['G', 'C', 'F', 'C', 'M', 'K', 'C', 'G', 'N', 'G', 'C', 'K', 'F', 'K', 'K', 'F', 'K', 'F', 'K', 'W']  # 0.9
conf = lattice.native_conf(seq)
seq_mrna = [aa_to_codons[c][0] for c in seq]
fitness = lattice.fracfolded(seq, target=conf)

for i in range(1000):
    # (fitnesses, mutation_positions, mutation_new_codons) = get_trajectory_by_fitness_goal(seq_mrna, conf)
    (fitnesses, mutation_positions, mutation_new_codons) = get_trajectory_by_purifying_fitness(seq_mrna, conf, fitness, steps=20)
    #print(fitnesses)
    #print(mutation_positions)
    #print(mutation_new_codons)

    df = pd.DataFrame([mutation_positions, mutation_new_codons, fitnesses]).transpose()
    num = str(np.random.randint(1e8))
    df.to_csv("../data/trajectories_09_00045/traj_{}.csv".format(num))


