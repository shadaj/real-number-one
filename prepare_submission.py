import sys
import os
import json
from parse import read_input_file, validate_file, read_output_file

if __name__ == '__main__':
    outputs_dir = sys.argv[1]
    submission_name = sys.argv[2]
    submission = {}
    scores = {}
    for folder in os.listdir("inputs"):
        if not folder.startswith('.'):
            for input_path in os.listdir("inputs/" + folder):
                G = read_input_file(f"inputs/{folder}/{input_path}")
                graph_name = input_path.split('.')[0]
                output_file = f'{outputs_dir}/{folder}/{graph_name}.out'
                if os.path.exists(output_file) and validate_file(output_file):
                    output = open(f'{outputs_dir}/{folder}/{graph_name}.out').read()
                    submission[input_path] = output
                    scores[graph_name] = round(read_output_file(G, output_file) * 1000) / 1000
    with open(submission_name, 'w') as f:
        f.write(json.dumps(submission))
    with open("scores.json", 'w') as f:
        f.write(json.dumps(scores))
