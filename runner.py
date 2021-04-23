from solver import solve
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score

import fire
import glob
import pickle
import heapq

from pathlib import Path
from os.path import exists, basename, normpath

from dataclasses import dataclass, field
from typing import List, Tuple, Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

size_mapping = {
  "small": (1, 15),
  "medium": (3, 50),
  "large": (5, 100)
}

def get_all_inputs(input_dir: str, input_type=None) -> List[Tuple[str, int, int]]:
  inputs_files = glob.glob('inputs/**/*.in')
  inputs_with_sizes = []
  for file_path in inputs_files:
    path = Path(file_path)
    if input_type == None or input_type == path.parent.name:
      max_cities, max_edges = size_mapping[path.parent.name]
      inputs_with_sizes.append((file_path, max_cities, max_edges))
  return inputs_with_sizes

def get_cached_run(input_path: str):
  cache_path = 'result-logs/' + basename(normpath(input_path))[:-3] + '.pkl'
  if exists(cache_path):
    with open(cache_path, "rb") as cache_file:
      return pickle.load(cache_file)
  else:
    return None

def write_cached_run(input_path: str, result):
  cache_path = 'result-logs/' + basename(normpath(input_path))[:-3] + '.pkl'
  with open(cache_path, "wb") as cache_file:
    return pickle.dump(result, cache_file)

def runner(input_dir="inputs", output_dir="outputs", input_type=None):
  inputs = get_all_inputs(input_dir, input_type)
  to_run_heap = []
  non_optimal_count = 0
  total_gaps = 0
  for input_path, max_cities, max_edges in inputs:
    cached = get_cached_run(input_path)
    if cached == None:
      heapq.heappush(to_run_heap, PrioritizedItem(0, {
        "in_path": input_path,
        "out_path": 'outputs/' + input_path[6:][:-3] + '.out',
        "existing_solution": None,
        "max_cities": max_cities,
        "max_edges": max_edges,
        "last_timeout": 5,
        "is_optimal": False,
        "best_gap": 100
      }))
      non_optimal_count += 1
      total_gaps += 100
    elif not cached["is_optimal"]:
      heapq.heappush(to_run_heap, PrioritizedItem(cached["last_timeout"], cached))
      non_optimal_count += 1
      total_gaps += cached["best_gap"]
    else:
      total_gaps += 0

  while len(to_run_heap) > 0:
    next_task = heapq.heappop(to_run_heap).item
    if not next_task["is_optimal"]:
      G = read_input_file(next_task["in_path"])
      next_timeout = next_task["last_timeout"] * 2
      in_path = next_task["in_path"]
      print()
      print()
      print("-----------------------------------------------------------")
      print()
      print()
      print(f"Running {in_path} with timeout {next_timeout} (remaining non-optimal: {non_optimal_count}/{len(inputs)}, average gap: {(total_gaps / non_optimal_count) * 100:.2f}%)")
      print()
      solve_result = solve(
        G, next_task["max_cities"], next_task["max_edges"],
        next_timeout, next_task["existing_solution"]
      )

      if solve_result:
        assert is_valid_solution(G, solve_result[0], solve_result[1])
        write_output_file(G, solve_result[0], solve_result[1], next_task["out_path"])
        print("Shortest Path Difference: {}".format(calculate_score(G, solve_result[0], solve_result[1])))
        if solve_result[2]:
          non_optimal_count -= 1
      new_gap = solve_result[3] if solve_result else 100
      total_gaps -= next_task["best_gap"]
      total_gaps += new_gap
      if next_task["existing_solution"] and new_gap > next_task["best_gap"]:
        orig_best = next_task["best_gap"]
        print(f"WARNING WARNING WARNING WARNING: new gap {new_gap} was larger than previous best gap {orig_best}")

      new_task = {
        "in_path": next_task["in_path"],
        "out_path": next_task["out_path"],
        "existing_solution": (solve_result[0], solve_result[1]) if solve_result else None,
        "max_cities": next_task["max_cities"],
        "max_edges": next_task["max_edges"],
        "last_timeout": next_timeout,
        "is_optimal": (solve_result != None) and solve_result[2],
        "best_gap": min(new_gap, next_task["best_gap"])
      }

      write_cached_run(new_task["in_path"], new_task)
      if not new_task["is_optimal"]:
        heapq.heappush(to_run_heap, PrioritizedItem(next_timeout, new_task))

if __name__ == '__main__':
  fire.Fire(runner)
