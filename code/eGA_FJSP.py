"""
Author: Chiu, Pao-Chang
Created time: 2024-08-24

Purpose:
This code is designed to address the flexible job-shop scheduling problem (FJSP) 
and to replicate the experimental results reported in the following paper.

Reference:
Zhang, G., Gao, L., & Shi, Y. (2011). An effective genetic algorithm for the flexible job-shop scheduling problem. 
Expert Systems with Applications, 38(4), 3563–3573.
"""

import time
import random
from typing import List, Tuple, Any
import pandas as pd
from settings import plot_schedule, plot_convergence

def read_excel(file_path: str) -> Any:
    df = pd.read_excel(file_path)

    # Try to convert 'Operation' column to numeric
    df["Operation"] = pd.to_numeric(df["Operation"], errors="coerce")

    jobs = df["Job"].max()
    machines = len(df.columns) - 2  # Subtract 'Job' and 'Operation' columns
    operations = []

    for job in range(1, jobs + 1):
        job_ops = []
        job_df = df[df["Job"] == job]
        for _, row in job_df.iterrows():
            machine_times = {}
            for m in range(1, machines + 1):
                if row[f"M{m}"] != "-":
                    machine_times[m - 1] = row[f"M{m}"]

            # Convert 'Operation' to int, use 0 if conversion fails
            operation = int(row["Operation"]) if pd.notna(row["Operation"]) else 0

            job_ops.append(Operation(job - 1, operation - 1, machine_times))
        operations.append(job_ops)

    return FJSP(jobs, machines, operations)


class Operation:
    def __init__(self, job: int, op_id: int, machine_times: dict):
        self.job = job
        self.op_id = op_id
        self.machine_times = machine_times


class FJSP:
    def __init__(self, jobs: int, machines: int, operations: List[List[Operation]]):
        self.jobs = jobs
        self.machines = machines
        self.operations = operations
        self.total_operations = sum(len(job_ops) for job_ops in operations)


class Chromosome:
    def __init__(self, ms: List[int], os: List[int]):
        self.ms = ms
        self.os = os


class GA:
    def __init__(
        self,
        fjsp: FJSP,
        pop_size: int = 100,
        generations: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.01,
    ):
        self.fjsp = fjsp
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Record the best makespan in each generation
        self.best_makespans = []

    def initialize(self) -> List[Chromosome]:
        population = []
        for _ in range(self.pop_size):
            if random.random() < 0.6:
                ms = self.global_selection()
            elif random.random() < 0.9:  # 0.6 + 0.3
                ms = self.local_selection()
            else:
                ms = self.random_selection()
            os = self.generate_operation_sequence()
            population.append(Chromosome(ms, os))
        return population

    def global_selection(self) -> List[int]:
        ms = []
        machine_times = [0] * self.fjsp.machines
        for job_ops in self.fjsp.operations:
            for op in job_ops:
                best_machine = min(
                    op.machine_times.items(), key=lambda x: x[1] + machine_times[x[0]]
                )[0]
                ms.append(best_machine)
                machine_times[best_machine] += op.machine_times[best_machine]
        return ms

    def local_selection(self) -> List[int]:
        ms = []
        for job_ops in self.fjsp.operations:
            machine_times = [0] * self.fjsp.machines
            for op in job_ops:
                best_machine = min(
                    op.machine_times.items(), key=lambda x: x[1] + machine_times[x[0]]
                )[0]
                ms.append(best_machine)
                machine_times[best_machine] += op.machine_times[best_machine]
        return ms

    def random_selection(self) -> List[int]:
        return [
            random.choice(list(op.machine_times.keys()))
            for job_ops in self.fjsp.operations
            for op in job_ops
        ]

    def generate_operation_sequence(self) -> List[int]:
        os = []
        for job, job_ops in enumerate(self.fjsp.operations):
            os.extend([job] * len(job_ops))
        random.shuffle(os)
        return os

    def tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        tournament = random.sample(population, 3)
        return min(tournament, key=self.evaluate)

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        # MS crossover
        if random.random() < 0.5:
            # Two-point crossover
            point1, point2 = sorted(random.sample(range(len(parent1.ms)), 2))
            child_ms = (
                parent1.ms[:point1] + parent2.ms[point1:point2] + parent1.ms[point2:]
            )
        else:
            # Uniform crossover
            child_ms = [
                p1 if random.random() < 0.5 else p2
                for p1, p2 in zip(parent1.ms, parent2.ms)
            ]

        # OS crossover (POX)
        jobs = list(set(parent1.os))
        selected_jobs = random.sample(jobs, len(jobs) // 2)
        child_os = [-1] * len(parent1.os)
        parent2_idx = 0
        for i, job in enumerate(parent1.os):
            if job in selected_jobs:
                child_os[i] = job
            else:
                while parent2.os[parent2_idx] in selected_jobs:
                    parent2_idx += 1
                child_os[i] = parent2.os[parent2_idx]
                parent2_idx += 1

        return Chromosome(child_ms, child_os)

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        # MS mutation
        for i in range(len(chromosome.ms)):
            if random.random() < self.mutation_rate:
                op = [op for job_ops in self.fjsp.operations for op in job_ops][i]
                chromosome.ms[i] = min(op.machine_times, key=op.machine_times.get)

        # OS mutation
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(chromosome.os)), 2)
            chromosome.os[i], chromosome.os[j] = chromosome.os[j], chromosome.os[i]

        return chromosome

    def decode(self, chromosome: Chromosome) -> List[Tuple[int, int, int, int, int]]:
        schedule = []
        machine_available_time = [0] * self.fjsp.machines
        job_next_op = [0] * self.fjsp.jobs

        for job in chromosome.os:
            op_id = job_next_op[job]
            machine = chromosome.ms[
                sum(len(job_ops) for job_ops in self.fjsp.operations[:job]) + op_id
            ]
            op = self.fjsp.operations[job][op_id]

            start_time = max(
                machine_available_time[machine],
                schedule[-1][4] if schedule and schedule[-1][0] == job else 0,
            )
            end_time = start_time + op.machine_times[machine]

            schedule.append((job, op_id, machine, start_time, end_time))
            machine_available_time[machine] = end_time
            job_next_op[job] += 1

        return schedule

    def evaluate(self, chromosome: Chromosome) -> int:
        schedule = self.decode(chromosome)
        return max(op[4] for op in schedule)

    def run(self) -> Tuple[List[Tuple[int, int, int, int, int]], int]:
        population = self.initialize()
        for _ in range(self.generations):
            new_population = []
            for _ in range(self.pop_size):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])
                child = self.mutation(child)
                new_population.append(child)
            population = new_population

            # Record the best makespan in the current generation
            best_makespan = min(self.evaluate(c) for c in population)
            self.best_makespans.append(best_makespan)

        best_chromosome = min(population, key=self.evaluate)
        best_schedule = self.decode(best_chromosome)
        return best_schedule, self.evaluate(best_chromosome)



# Main execution
if __name__ == "__main__":
    # Example small dataset
    # dataset = "dataset/Sample/eGA_FJSP_table1.xlsx" 
    # Benchmark dataset
    dataset = "dataset/Benchmark/FJSP/Brandimarte/Mk10.xlsx"
    fjsp = read_excel(dataset)

    start_time = time.time()

    ga = GA(fjsp, pop_size=300, generations=100)
    best_schedule, makespan = ga.run()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Best makespan: {makespan}")
    # print("Best schedule:")
    # for op in best_schedule:
    #     print(
    #         f"Job {op[0]+1} Operation {op[1]+1} on Machine {op[2]+1} from {op[3]} to {op[4]}"
    #     )
    print(f"Execution time: {execution_time: .2f} sec")

    # Visualize results
    # plot_schedule(dataset, best_schedule, makespan, save_fig=False)
    # plot_convergence(dataset, ga.best_makespans, save_fig=False)

