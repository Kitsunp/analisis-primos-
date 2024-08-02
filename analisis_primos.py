"""
Prime Number Formula Generator using Genetic Algorithms and Metaprogramming

This module implements a sophisticated system for generating and optimizing
mathematical formulas capable of identifying prime numbers. It utilizes
genetic algorithms, metaprogramming techniques, and symbolic mathematics
to evolve increasingly effective formulas.

The main components of this system are:
1. FormulaGenerator: Creates random mathematical formulas
2. GeneticAlgorithm: Evolves and optimizes the formulas
3. Various utility functions for formula evaluation and manipulation

The primary use case is to generate a formula that can identify prime numbers
up to a specified limit with high accuracy and efficiency.

This will generate an optimized formula for identifying primes up to 100,
evaluate it, and print the results.

Note: This is a computationally intensive process and may take significant
time for larger values of n.
"""
import random
import time
import sympy as sp
import inspect
from functools import wraps
from typing import Callable, List, Tuple
import numpy as np

def timer(func: Callable) -> Callable:
    """
    A decorator that measures and prints the execution time of a function.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: A wrapper function that executes the original function
                  and returns its result along with the execution time.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        execution_time = end - start
        print(f"{func.__name__} - Tiempo de ejecución: {execution_time:.6f} segundos")
        return result, execution_time
    return wrapper

def safe_eval(expr: sp.Expr, x: float, max_value: float = 10) -> float | None:
    """
    Safely evaluates a symbolic expression for a given value of x.

    This function attempts to evaluate the expression and returns the result
    only if it's a real number within the specified maximum absolute value.

    Args:
        expr (sp.Expr): The symbolic expression to evaluate.
        x (float): The value to substitute for the variable in the expression.
        max_value (float, optional): The maximum absolute value allowed for the result.
                                     Defaults to 10.

    Returns:
        float | None: The evaluated result if it's valid, or None if the evaluation
                      fails or produces an out-of-range result.
    """
    try:
        result = expr.evalf(subs={sp.Symbol('x'): x})
        if result.is_real and abs(result) < max_value:
            return float(result)
        return None
    except Exception as e:
        print(f"Error in safe_eval: {e}")
        return None

class FormulaGenerator:
    """
    A class for generating random mathematical formulas.

    This class provides methods to create complex mathematical expressions
    using a set of predefined operations and symbols.

    Attributes:
        x (sp.Symbol): The symbol used as the variable in the formulas.
        operations (List[Tuple[Callable, int]]): A list of tuples containing
            mathematical operations and their arity.

    Methods:
        create_term(depth: int = 0) -> sp.Expr:
            Recursively creates a term in the formula.
        generate_formula() -> sp.Lambda:
            Generates a complete formula as a lambda function.
    """

    def __init__(self):
        """Initializes the FormulaGenerator with a set of operations."""
        self.x = sp.Symbol('x')
        self.operations = [
            (sp.Add, 2),
            (sp.Mul, 2),
            (sp.sqrt, 1),
            (sp.log, 1),
            (sp.sin, 1),
            (sp.cos, 1),
            (sp.exp, 1)
        ]

    def create_term(self, depth: int = 0) -> sp.Expr:
        """
        Recursively creates a term in the formula.

        This method builds a mathematical expression by randomly choosing
        between simple terms (constants or variables) and complex terms
        (composed of operations on other terms).

        Args:
            depth (int, optional): The current depth in the recursion. Defaults to 0.

        Returns:
            sp.Expr: A sympy expression representing the created term.
        """
        if depth > 5 or random.random() < 0.3:
            return random.choice([sp.Integer(2), sp.Integer(3), sp.Integer(5), sp.Integer(7), sp.Integer(11), self.x])
        op, arity = random.choice(self.operations)
        if arity == 1:
            return op(self.create_term(depth + 1))
        return op(self.create_term(depth + 1), self.create_term(depth + 1))

    def generate_formula(self) -> sp.Lambda:
        """
        Generates a complete formula as a lambda function.

        This method creates a complex mathematical formula by calling create_term
        and wrapping the result in a lambda function.

        Returns:
            sp.Lambda: A sympy Lambda function representing the generated formula.
        """
        formula = self.create_term()
        return sp.Lambda(self.x, formula)

def dynamic_operation_generator() -> Callable[[], Tuple[Callable, int]]:
    """
    Creates a function that generates dynamic mathematical operations.

    This function returns another function that, when called, randomly selects
    and returns a mathematical operation along with its arity.

    Returns:
        Callable[[], Tuple[Callable, int]]: A function that generates a random
        mathematical operation and its arity.
    """
    operations = [
        "lambda a, b: a + b",
        "lambda a, b: a * b",
        "lambda a: sp.sqrt(abs(a))",
        "lambda a: sp.log(abs(a) + 1)",
        "lambda a: sp.sin(a)",
        "lambda a: sp.cos(a)",
        "lambda a: sp.exp(a)",
        "lambda a: sp.tan(a)",
        "lambda a: sp.atan(a)",
        "lambda a, b: a - b",
        "lambda a, b: a / (b + 0.001)",
    ]

    def create_operation():
        op_str = random.choice(operations)
        return eval(op_str), len(inspect.signature(eval(op_str)).parameters)

    return create_operation
def metaprogramming_formula_generator(max_depth: int = 7, complexity_factor: float = 0.7) -> sp.Lambda:
    """
    Generates formulas using metaprogramming techniques.

    This function creates complex mathematical formulas by dynamically
    combining various operations and terms.

    Args:
        max_depth (int, optional): The maximum depth of the formula tree. Defaults to 7.
        complexity_factor (float, optional): A factor influencing the complexity
            of the generated formula. Defaults to 0.7.

    Returns:
        sp.Lambda: A sympy Lambda function representing the generated formula.
    """
    x = sp.Symbol('x')
    create_operation = dynamic_operation_generator()

    def create_term(depth: int = 0) -> sp.Expr:
        if depth > max_depth or random.random() < complexity_factor ** depth:
            return random.choice([sp.Integer(2), sp.Integer(3), sp.Integer(5), sp.Integer(7), sp.Integer(11), x])
        op, arity = create_operation()
        if arity == 1:
            return op(create_term(depth + 1))
        return op(create_term(depth + 1), create_term(depth + 1))

    formula = create_term()
    return sp.Lambda(x, formula)


def evaluate_formula(formula: sp.Lambda, n: int) -> List[int]:
    """
    Evaluates a formula to find prime numbers up to a given limit.

    This function tests the given formula against all numbers up to 'n'
    and returns a list of numbers that the formula identifies as prime.

    Args:
        formula (sp.Lambda): The formula to evaluate.
        n (int): The upper limit for testing prime numbers.

    Returns:
        List[int]: A list of numbers identified as prime by the formula.
    """
    primes = []
    for i in range(2, n+1):
        if all(i % j != 0 for j in range(2, int(i**0.5) + 1)):
            result = safe_eval(formula.expr, i)
            if result is not None and abs(result) < 1e-10:
                primes.append(i)
    return primes


import random
import sympy as sp
import numpy as np
from typing import List, Tuple

class GeneticAlgorithm:
    """
    Implements a genetic algorithm for optimizing prime-generating formulas.

    This class provides methods to evolve a population of mathematical formulas
    with the goal of finding formulas that accurately identify prime numbers.

    Attributes:
        population_size (int): The number of formulas in each generation.
        mutation_rate (float): The probability of mutation for each formula.
        crossover_rate (float): The probability of crossover between formulas.
        formula_generator (FormulaGenerator): An instance used to generate random formulas.

    Methods:
        initialize_population() -> List[sp.Lambda]:
            Creates an initial population of random formulas.
        fitness(formula: sp.Lambda, n: int) -> int:
            Evaluates the fitness of a formula.
        select_parents(population: List[sp.Lambda], fitnesses: List[int]) -> Tuple[sp.Lambda, sp.Lambda]:
            Selects two parent formulas for reproduction.
        crossover(parent1: sp.Lambda, parent2: sp.Lambda) -> sp.Lambda:
            Performs crossover between two parent formulas.
        mutate(formula: sp.Lambda) -> sp.Lambda:
            Applies mutation to a formula.
        evolve(population: List[sp.Lambda], n: int) -> List[sp.Lambda]:
            Evolves the population of formulas for one generation.
    """

    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float):
        """
        Initializes the GeneticAlgorithm with specified parameters.

        Args:
            population_size (int): The number of formulas in each generation.
            mutation_rate (float): The probability of mutation for each formula.
            crossover_rate (float): The probability of crossover between formulas.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.formula_generator = FormulaGenerator()

    def initialize_population(self) -> List[sp.Lambda]:
        """
        Creates an initial population of random formulas.

        Returns:
            List[sp.Lambda]: A list of randomly generated formulas.
        """
        return [self.formula_generator.generate_formula() for _ in range(self.population_size)]

    def fitness(self, formula: sp.Lambda, n: int) -> int:
        """
        Evaluates the fitness of a formula.

        The fitness is determined by the number of prime numbers correctly
        identified by the formula up to the limit n.

        Args:
            formula (sp.Lambda): The formula to evaluate.
            n (int): The upper limit for testing prime numbers.

        Returns:
            int: The number of primes correctly identified by the formula.
        """
        return len(evaluate_formula(formula, n))

    def select_parents(self, population: List[sp.Lambda], fitnesses: List[int]) -> Tuple[sp.Lambda, sp.Lambda]:
        """
        Selects two parent formulas for reproduction.

        This method uses fitness-proportionate selection (roulette wheel selection)
        to choose parents.

        Args:
            population (List[sp.Lambda]): The current population of formulas.
            fitnesses (List[int]): The fitness scores corresponding to the population.

        Returns:
            Tuple[sp.Lambda, sp.Lambda]: Two selected parent formulas.
        """
        viable_parents = [p for p, f in zip(population, fitnesses) if f > 0]
        if len(viable_parents) < 2:
            return (self.formula_generator.generate_formula(), self.formula_generator.generate_formula())
        elif len(viable_parents) == 2:
            return tuple(viable_parents)
        else:
            total_fitness = sum(fitnesses)
            probabilities = [f / total_fitness for f in fitnesses if f > 0]
            return tuple(np.random.choice(viable_parents, size=2, p=probabilities, replace=False))

    def crossover(self, parent1: sp.Lambda, parent2: sp.Lambda) -> sp.Lambda:
        """
        Performs crossover between two parent formulas.

        This method combines parts of two parent formulas to create a new formula.

        Args:
            parent1 (sp.Lambda): The first parent formula.
            parent2 (sp.Lambda): The second parent formula.

        Returns:
            sp.Lambda: A new formula resulting from the crossover.
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, 5)
            new_formula = self.swap_subterms(parent1.expr, parent2.expr, crossover_point)
            return sp.Lambda(sp.Symbol('x'), new_formula)
        return random.choice([parent1, parent2])
    
    def swap_subterms(self, expr1: sp.Expr, expr2: sp.Expr, depth: int) -> sp.Expr:
        """
        Swaps subterms between two expressions at a given depth.

        This method is used as part of the crossover operation to combine
        parts of two parent formulas.

        Args:
            expr1 (sp.Expr): The first expression.
            expr2 (sp.Expr): The second expression.
            depth (int): The depth at which to perform the swap.

        Returns:
            sp.Expr: A new expression resulting from the swap.
        """
        if depth == 0 or random.random() < 0.5:
            return expr2
        if isinstance(expr1, sp.Symbol) or isinstance(expr1, sp.Integer):
            return expr1
        args = list(expr1.args)
        for i, arg in enumerate(args):
            args[i] = self.swap_subterms(arg, expr2, depth - 1)
        return expr1.func(*args)

    def mutate(self, formula: sp.Lambda) -> sp.Lambda:
        """
        Applies mutation to a formula.

        This method randomly alters a formula to introduce variation into
        the population.

        Args:
            formula (sp.Lambda): The formula to mutate.

        Returns:
            sp.Lambda: A new, possibly mutated formula.
        """
        if random.random() < self.mutation_rate:
            new_formula = self.formula_generator.generate_formula()
            return self.crossover(formula, new_formula)
        return formula

    def evolve(self, population: List[sp.Lambda], n: int) -> List[sp.Lambda]:
        """
        Evolves the population of formulas for one generation.

        This method applies the genetic operations (selection, crossover, mutation)
        to create a new generation of formulas.

        Args:
            population (List[sp.Lambda]): The current population of formulas.
            n (int): The upper limit for testing prime numbers (used in fitness evaluation).

        Returns:
            List[sp.Lambda]: A new population of evolved formulas.
        """
        fitnesses = [self.fitness(formula, n) for formula in population]
        new_population = []
        while len(new_population) < self.population_size:
            parents = self.select_parents(population, fitnesses)
            child = self.crossover(*parents)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

@timer
def meta_reinforcement_prime_generator(n: int, generations: int = 50, population_size: int = 500, print_formulas: bool = True) -> sp.Lambda:
    """
    Generates an optimized prime-generating formula using genetic algorithms.

    This function evolves a population of formulas over multiple generations
    to find an optimal formula for identifying prime numbers.

    Args:
        n (int): The upper limit for testing prime numbers.
        generations (int, optional): The number of generations to evolve. Defaults to 50.
        population_size (int, optional): The size of the formula population. Defaults to 50.
        print_formulas (bool, optional): Whether to print intermediate results. Defaults to True.

    Returns:
        sp.Lambda: The best formula found after evolution.
    """
    ga = GeneticAlgorithm(population_size=population_size, mutation_rate=0.3, crossover_rate=0.7)
    population = ga.initialize_population()

    best_formula = None
    best_score = 0

    for gen in range(generations):
        fitnesses = [ga.fitness(formula, n) for formula in population]
        
        best_idx = fitnesses.index(max(fitnesses))
        current_best_formula = population[best_idx]
        current_best_score = fitnesses[best_idx]

        if current_best_score > best_score:
            best_score = current_best_score
            best_formula = current_best_formula

        print(f"Generación {gen + 1}: Mejor puntuación = {best_score}")
        
        if print_formulas:
            formula_to_print = best_formula if best_formula is not None else random.choice(population)
            print(f"Mejor fórmula hasta ahora: {formula_to_print}")

        population = ga.evolve(population, n)
        
        population[0] = best_formula if best_formula is not None else current_best_formula

    return best_formula

def run_optimized_prime_formula(n: int):
    """
    Runs the optimized prime formula generator and evaluates its performance.

    This function generates an optimized formula for identifying prime numbers,
    evaluates it, and prints the results.

    Args:
        n (int): The upper limit for testing prime numbers.
    """
    print(f"\nGenerando fórmula matemática optimizada para primos hasta {n}:")

    formula = None
    attempts = 0
    max_attempts = 5

    while formula is None and attempts < max_attempts:
        formula, generation_time = meta_reinforcement_prime_generator(n, print_formulas=True)
        attempts += 1
        if formula is None:
            print(f"Intento {attempts} fallido. Reintentando...")

    if formula is None:
        print(f"No se pudo generar una fórmula válida después de {max_attempts} intentos.")
        return

    print(f"Fórmula final generada: {formula}")
    primes, evaluation_time = timer(evaluate_formula)(formula, n)

    print(f"Tiempo de generación: {generation_time:.6f} segundos")
    print(f"Tiempo de evaluación: {evaluation_time:.6f} segundos")
    print(f"Primeros 10 primos encontrados: {primes[:10]}")
    print(f"Últimos 10 primos encontrados: {primes[-10:]}")
    print(f"Total de primos encontrados: {len(primes)}")

if __name__ == "__main__":
    n = 100  # Can be changed as needed
    run_optimized_prime_formula(n)