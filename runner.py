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

@dataclass
class PrioritizedItem:
  item: Any=field(compare=False)

  @property
  def estimated_timeout_to_complete(self):
    if self.item["existing_solution"] == None:
      return self.item["last_timeout"] * 4
    elif abs(self.item["gap_change"]) <= 0.0001 or self.item["gap_change"] > 0:
      return self.item["last_timeout"] * 4
    else:
      seconds_per_change = self.item["timeout_change"] / -self.item["gap_change"]
      time_for_remaining_gap = self.item["best_gap"] * seconds_per_change
      return round(self.item["last_timeout"] + time_for_remaining_gap)

  @property
  def priority_func(self):
    tight_estimate = self.estimated_timeout_to_complete
    return (tight_estimate, self.item["last_timeout"], self.item["best_gap"])
  
  def __eq__(self, o: "PrioritizedItem") -> bool:
    return self.priority_func == o.priority_func

  def __lt__(self, o: "PrioritizedItem") -> bool:
    return self.priority_func < o.priority_func

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

def solver_loop(input_dir="inputs", output_dir="outputs", input_type=None):
  inputs = get_all_inputs(input_dir, input_type)
  to_run_heap = []
  non_optimal_count = 0
  total_gaps = 0
  for input_path, max_cities, max_edges in inputs:
    cached = get_cached_run(input_path)
    if cached == None:
      heapq.heappush(to_run_heap, PrioritizedItem({
        "in_path": input_path,
        "out_path": 'outputs/' + input_path[6:][:-3] + '.out',
        "existing_solution": None,
        "max_cities": max_cities,
        "max_edges": max_edges,
        "last_timeout": 1,
        "is_optimal": False,
        "best_gap": 100,
        "gap_change": 0,
        "timeout_change": 0
      }))
      non_optimal_count += 1
      total_gaps += 100
    elif not cached["is_optimal"]:
      if "gap_change" not in cached:
        cached["gap_change"] = 0
      if "timeout_change" not in cached:
        cached["timeout_change"] = cached["last_timeout"] / 2
      heapq.heappush(to_run_heap, PrioritizedItem(cached))
      non_optimal_count += 1
      total_gaps += cached["best_gap"]
    else:
      total_gaps += 0

  while len(to_run_heap) > 0:
    next_task_wrap = heapq.heappop(to_run_heap)
    next_task = next_task_wrap.item
    if not next_task["is_optimal"]:
      G = read_input_file(next_task["in_path"])
      next_timeout = next_task_wrap.estimated_timeout_to_complete
      timeout_delta = next_timeout - next_task["last_timeout"]
      last_progress = next_task["gap_change"]
      in_path = next_task["in_path"]
      print()
      print()
      print("-----------------------------------------------------------")
      print()
      print()
      print(f"Remaining non-optimal: {non_optimal_count}/{len(inputs)}, average gap: {(total_gaps / non_optimal_count) * 100:.2f}%")
      print(f"Running {in_path} with timeout {next_timeout}, last iteration progress {last_progress*100:.2f}%")
      print()
      solve_result = solve(
        G, next_task["max_cities"], next_task["max_edges"],
        next_timeout, next_task["existing_solution"]
      )

      prev_score = 0
      if next_task["existing_solution"]:
        prev_score = calculate_score(G, next_task["existing_solution"][0], next_task["existing_solution"][1])
      new_score = 0
      if solve_result:
        assert is_valid_solution(G, solve_result[0], solve_result[1])
        new_score = calculate_score(G, solve_result[0], solve_result[1])

      if solve_result and new_score > prev_score:
        write_output_file(G, solve_result[0], solve_result[1], next_task["out_path"])
        print(f"Shortest Path Difference: {new_score}")
        if solve_result[2]:
          non_optimal_count -= 1
      new_gap = solve_result[3] if solve_result else 100
      total_gaps -= next_task["best_gap"]
      total_gaps += new_gap
      if next_task["existing_solution"] and next_task["best_gap"] != 100 and new_gap > next_task["best_gap"]:
        orig_best = next_task["best_gap"]
        print(f"WARNING WARNING WARNING WARNING: new gap {new_gap} was larger than previous best gap {orig_best}")
        new_gap = orig_best

      new_task = {
        "in_path": next_task["in_path"],
        "out_path": next_task["out_path"],
        "existing_solution": (solve_result[0], solve_result[1]) if solve_result and new_score > prev_score else next_task["existing_solution"],
        "max_cities": next_task["max_cities"],
        "max_edges": next_task["max_edges"],
        "last_timeout": next_timeout,
        "is_optimal": (solve_result != None) and solve_result[2],
        "best_gap": new_gap,
        "gap_change": new_gap - next_task["best_gap"] if (next_task["existing_solution"] and next_task["best_gap"] != 100) else 0,
        "timeout_change": timeout_delta
      }

      write_cached_run(new_task["in_path"], new_task)
      if not new_task["is_optimal"]:
        heapq.heappush(to_run_heap, PrioritizedItem(new_task))

def stats(input_dir="inputs", input_type = None):
  import matplotlib.pyplot as plt
  import seaborn as sns

  inputs = get_all_inputs(input_dir, input_type)
  cached_runs = []
  for input_path, max_cities, max_edges in inputs:
    cached = get_cached_run(input_path)
    if cached:
      if "gap_change" not in cached:
        cached["gap_change"] = 0
      if "timeout_change" not in cached:
        cached["timeout_change"] = cached["last_timeout"] / 2
      cached_runs.append(cached)

  print(f"Total Inputs: {len(inputs)}")    
  print(f"Total Runned Inputs: {len(cached_runs)}")
  optimal_runs = [run for run in cached_runs if run["is_optimal"]]
  non_optimal_runs = [run for run in cached_runs if not run["is_optimal"]]
  has_solution = [run for run in cached_runs if run["existing_solution"]]
  print(f"Optimal Count: {len(optimal_runs)}")
  print(f"Has Solution: {len(has_solution)}")

  plt.hist([run["last_timeout"] for run in optimal_runs], bins=25)
  plt.savefig('optimal-timeouts.png')
  plt.clf()

  plt.hist([run["best_gap"] * 100 for run in non_optimal_runs], bins=25)
  plt.savefig('gaps.png')
  plt.clf()

  print([run["in_path"] for run in non_optimal_runs])

if __name__ == '__main__':
  fire.Fire()
