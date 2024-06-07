import numpy as np
import random

#constructor for the hopfield network class with data fields for size, weights, state, and threshold. 
#weights and threshold are initialized to zero, state(s) is initialized to 1/activated. 
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        self.state = np.ones(size)
        self.threshold = np.zeros(size)
    
    #takes an array of patterns with binary values as a parameter, then converts them to bipolar values of 1 and -1. 
    #weights are assigned using the outer product of the pattern with itself, diagnoal values in the matrix are set to zero to avoid self-connections
    def train(self, patterns):
        for pattern in patterns:
            bipolar_pattern = 2 * pattern - 1
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        np.fill_diagonal(self.weights, 0)
    
    #updates state of unit according to activation function in the lecture slides 
    def update_unit(self, index):
        net_input = np.dot(self.weights[index], self.state) - self.threshold[index]
        self.state[index] = 1 if net_input > 0 else -1

    #calculates net input for all units at once and updates them simultaneuosly/synchronously
    def sync_update(self):
        net_input = np.dot(self.weights, self.state) - self.threshold
        new_state = np.where(net_input > 0, 1, -1)
        self.state = new_state

    #randomly selects 5 units, then asynchronously updates them using previous update function
    def async_update(self, update_count=5):
        indices = np.random.choice(self.size, update_count, replace=False)
        for index in indices:
            self.update_unit(index)
 
    #runs network from an initial state and has a ceiling of 100 iterations. allows for all 3 methods of unit updating. records energy history and checks for convergence
    def run(self, initial_state, max_iterations=100, update_mode='async'):
        self.state = initial_state
        energy_history = []
        for _ in range(max_iterations):
            prev_state = self.state.copy()
            if update_mode == 'sync':
                self.sync_update()
            elif update_mode == 'random':
                self.async_update(update_count=5)
            else:
                for i in range(self.size):
                    self.update_unit(i)

            current_energy = self.energy()
            energy_history.append(current_energy)
            if np.array_equal(self.state, prev_state):
                break
        return energy_history
    
    #calculates energy using notation from lecture slides
    def energy(self):
        return -0.5 * np.dot(self.state, np.dot(self.weights, self.state))
    
    #creates a noisy pattern based off an existing pattern by flipping each bipolar value according to a probability passed into the method
    def apply_noise(self, pattern, noise_level):
        noisy_pattern = pattern.copy()
        for i in range(self.size):
            if random.random() < noise_level:
                noisy_pattern[i] = -noisy_pattern[i]
        return noisy_pattern

    #determines whether convergence has been acheived, returns a boolean value. if energy is identical consecutively, it will return True
    def has_converged(self, energy_history):
        return len(energy_history) > 1 and energy_history[-1] == energy_history[-2]
    
    #calculates hamming distance of two patterns by returning the sum of their differences divided by 2 
    def hamming_distance(self, pattern1, pattern2):
        return np.sum(pattern1 != pattern2) / 2

    #tests network on pattern passed into method at a passed noise level. applies noise to the pattern, then runs network and stores the energy history and hamming distance
    def test_network(self, pattern, noise_level, max_iterations=100):
        noisy_pattern = self.apply_noise(pattern, noise_level)
        energy_history = self.run(noisy_pattern, max_iterations)
        hamming_dist = self.hamming_distance(pattern, self.state)
        return hamming_dist, energy_history

#performs simulations/tests network on patterns passed as an argument for specified number of runs per pattern, and stores results
def perform_simulations(network, patterns, noise_levels, runs_per_pattern):
    results = {}
    for noise in noise_levels:
        results[noise] = {
            'hamming_distances': [],
            'iterations_count': [],
            'energy_histories': [],
            'failures': 0
        }
        for _ in range(runs_per_pattern):
            for pattern in patterns:
                hamming_dist, energy_history = network.test_network(pattern, noise)
                results[noise]['hamming_distances'].append(hamming_dist)
                results[noise]['iterations_count'].append(len(energy_history))
                results[noise]['energy_histories'].append(energy_history)
                if not network.has_converged(energy_history):
                    results[noise]['failures'] += 1
    return results

#summarizes data, then formats and prints
def print_results(results):
    for noise_level, data in results.items():
        print(f"noise level {noise_level}:")
        print(f"  average hamming distance = {np.mean(data['hamming_distances'])}")
        print(f"  average number of iterations = {np.mean(data['iterations_count'])}")
        print(f"  failures = {data['failures']}")
        for energies in data['energy_histories']:
            print(f"  energy history: {energies}")
        print()

#generates specified number of random bipolar patterns with activation probability of 0.5
def generate_random_patterns(size, num_patterns):
    return [np.where(np.random.rand(size) < 0.5, 1, -1) for _ in range(num_patterns)]

#creates base pattern with first half of nodes activated and second half not activated. creates noisy training patterns from base pattern with specified probability of bipolar value flip
def create_base_and_training_patterns(size, num_patterns, switch_prob):
    base_pattern = np.concatenate((np.ones(size // 2), -np.ones(size // 2)))
    training_patterns = []
    for _ in range(num_patterns):
        new_pattern = base_pattern.copy()
        for i in range(size):
            if random.random() < switch_prob:
                new_pattern[i] = -new_pattern[i]
        training_patterns.append(new_pattern)
    return base_pattern, training_patterns

#main
if __name__ == "__main__":
    size = 16
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    hopfield_net_part1 = HopfieldNetwork(size)
    
    #performing simulations in accordance with assignment instructions. broken up into 4 parts
    print("\nPART 1\n")
    walsh_patterns = [
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    ]
    hopfield_net_part1.train(walsh_patterns)
    part_a_results = perform_simulations(hopfield_net_part1, walsh_patterns, noise_levels, 5)
    print_results(part_a_results)

    #part 2: generate and test 3 random patterns
    print("\nPART 2\n")
    hopfield_net_part2 = HopfieldNetwork(size)
    random_patterns = generate_random_patterns(size, 3)
    hopfield_net_part2.train(random_patterns) 
    part_b_results = perform_simulations(hopfield_net_part2, random_patterns, noise_levels, 5)
    print("\nperforming simulations using 3 random patterns in which neurons have 0.5 probability of being active:\n")
    print_results(part_b_results)

    #testing 2 more random patterns one at a time
    for i in range(2):
        new_pattern = generate_random_patterns(size, 1)[0]
        hopfield_net_part2.train([new_pattern])
        print(f"\ntesting new random pattern {i+1} at noise level of 0.2:")
        new_pattern_results = perform_simulations(hopfield_net_part2, [new_pattern], [0.2], 5)
        print_results(new_pattern_results)


    #create new network for part 3 and train on new patterns made from base pattern
    print("\nPART 3\n")
    hopfield_net_part3 = HopfieldNetwork(size)
    base_pattern, training_patterns_part3 = create_base_and_training_patterns(size, 6, 0.125)
    hopfield_net_part3.train(training_patterns_part3)

    print("\ntesting part 3 network with part 1 patterns (zero noise):")
    p1_zero_noise = perform_simulations(hopfield_net_part3, walsh_patterns, [0], 5)
    print_results(p1_zero_noise)

    print("\ntesting part 3 network with part 2 patterns (zero noise):")
    p2_zero_noise = perform_simulations(hopfield_net_part3, random_patterns, [0], 5)
    print_results(p2_zero_noise)

    print("\nPART 4\n")
    hopfield_net_part4 = HopfieldNetwork(size)
    hopfield_net_part4.train(walsh_patterns) 

    initial_state = np.where(np.random.rand(size) < 0.5, 1, -1)
    print("synchronous update:")
    energy_history_sync = hopfield_net_part4.run(initial_state.copy(), max_iterations=100, update_mode = 'sync')
    print(energy_history_sync)

    print("asynchronous update (5 random nodes each time):")
    energy_history_async = hopfield_net_part4.run(initial_state.copy(), max_iterations=100, update_mode = 'random')
    print(energy_history_async)
