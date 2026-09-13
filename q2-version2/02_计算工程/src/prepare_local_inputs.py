"""Copy the four existing Q1/Q2 input files locally after checking their hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=root.parents[1] / '03_计算工程/data/raw/附件')
    args = parser.parse_args()
    expected = json.loads((root / 'configs/raw_inputs_sha256.json').read_text(encoding='utf-8'))
    jobs = []
    for relative, digest in expected.items():
        source = args.raw_dir / relative
        target = root / 'data/raw/附件' / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing input: {source}; use --raw-dir to specify the attachment directory')
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input hash differs from the validated dataset: {source}')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Refusing to overwrite a different local file: {target}')
        jobs.append((source, target, digest))
    for source, target, digest in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    print(f'Prepared {len(jobs)} local input files; raw inputs are ignored by Git.')


if __name__ == '__main__':
    main()
