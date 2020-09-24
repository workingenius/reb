import os
import sys
import argparse
import importlib
from pathlib import Path


def main():
    from reb import Pattern

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pattern', help='import path of the pattern object')
    ap.add_argument('-i', '--input', help='file to extract info', type=Path)
    ap.add_argument('-e', '--engine', help='which engine to use', default='plain', choices=['plain', 'vm'])
    args = ap.parse_args()
    
    input_file_path: Path
    pattern_import_path, input_file_path = args.pattern, args.input
    
    if not input_file_path.is_file():
        print('File does not exist')
        sys.exit(2)

    pattern = get_pattern(pattern_import_path)
    if not isinstance(pattern, Pattern):
        print('Path given is not a pattern')
        sys.exit(3)
    
    with open(input_file_path, 'r') as fo:
        text = fo.read()
        extraction = pattern.extractiter(text, engine=args.engine)
        
        for segment in extraction:
            segment.pp()


def get_pattern(import_path):
    if ':' not in import_path:
        import_path += ':pattern'
    assert import_path.count(':') == 1
    mod, var = import_path.split(':')
    module = importlib.import_module(mod)
    return getattr(module, var, None)


if __name__ == '__main__':
    main()
