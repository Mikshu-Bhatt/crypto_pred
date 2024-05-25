import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_output')
    args = parser.parse_args()

    return args

args = parse_args()