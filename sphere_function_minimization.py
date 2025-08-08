import random
import math
from typing import List, Tuple, Callable

Vector = List[float]
Bounds = List[Tuple[float, float]]
Objective = Callable[[Vector], float]


def sphere_function(x: Vector) -> float:
    return sum(xi * xi for xi in x)

def clamp(x: Vector, bounds: Bounds) -> Vector:
    return [max(lo, min(xi, hi)) for xi, (lo, hi) in zip(x, bounds)]

def random_point(bounds: Bounds) -> Vector:
    return [random.uniform(lo, hi) for (lo, hi) in bounds]

def add_step(x: Vector, step: Vector, bounds: Bounds) -> Vector:
    return clamp([xi + si for xi, si in zip(x, step)], bounds)

def l2_dist(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def hill_climbing(
    func: Objective,
    bounds: Bounds,
    iterations: int = 1000,
    epsilon: float = 1e-6,
    init_step: float = 0.5,
    step_cooling: float = 0.98,
) -> Tuple[Vector, float]:
    """
    Жадібно приймаємо тільки покращення. Крок поступово зменшуємо.
    Зупинка: мале покращення функції або зміщення < epsilon, або вичерпання ітерацій.
    """
    dim = len(bounds)
    x = random_point(bounds)
    fx = func(x)
    step_size = init_step

    for _ in range(iterations):
        step = [random.gauss(0.0, step_size) for _ in range(dim)]
        cand = add_step(x, step, bounds)
        fc = func(cand)

        if fc < fx: 
            delta_f = fx - fc
            move = l2_dist(cand, x)
            x, fx = cand, fc

            if delta_f < epsilon or move < epsilon:
                break
        else:
            step_size *= step_cooling

        if step_size < epsilon:
            break

    return x, fx


def random_local_search(
    func: Objective,
    bounds: Bounds,
    iterations: int = 1000,
    epsilon: float = 1e-6,
    neighborhood: float = 0.5,
) -> Tuple[Vector, float]:
    
    dim = len(bounds)
    x_best = random_point(bounds)
    f_best = func(x_best)

    for it in range(iterations):
        if random.random() < 0.1:
            cand = random_point(bounds)
        else:
            step = [random.uniform(-neighborhood, neighborhood) for _ in range(dim)]
            cand = add_step(x_best, step, bounds)

        fc = func(cand)
        delta = f_best - fc
        if delta > 0:
            x_best, f_best = cand, fc
            if delta < epsilon:  
                break

    return x_best, f_best


# --- Simulated Annealing -------------------------------------------------------
def simulated_annealing(
    func: Objective,
    bounds: Bounds,
    iterations: int = 1000,
    temp: float = 10.0,
    cooling_rate: float = 0.97,
    epsilon: float = 1e-6,
    init_step: float = 0.8,
) -> Tuple[Vector, float]:
    """
    Приймає гірші кроки з імовірністю exp(-Δf / T). Температура охолоджується геометрично.
    Зупинка: T < epsilon або зміни надто малі.
    """
    dim = len(bounds)
    x = random_point(bounds)
    fx = func(x)
    best_x, best_f = x[:], fx
    step_size = init_step

    for _ in range(iterations):
        if temp < epsilon:
            break

        local_sigma = max(step_size * temp, epsilon)
        step = [random.gauss(0.0, local_sigma) for _ in range(dim)]
        cand = add_step(x, step, bounds)
        fc = func(cand)
        delta = fc - fx

        if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-12)):
            move = l2_dist(cand, x)
            x, fx = cand, fc
            if fx < best_f:
                best_x, best_f = x[:], fx
            if abs(delta) < epsilon and move < epsilon:
                break

        temp *= cooling_rate

    return best_x, best_f


if __name__ == "__main__":
    random.seed(42) 

    bounds = [(-5, 5), (-5, 5)] 

    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds, iterations=5000)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds, iterations=5000)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(
        sphere_function, bounds, iterations=5000, temp=10.0, cooling_rate=0.97
    )
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
