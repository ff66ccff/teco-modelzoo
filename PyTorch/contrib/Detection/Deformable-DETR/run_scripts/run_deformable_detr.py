# BSD 3-Clause License Copyright (c) 2026, Tecorigin Co., Ltd. All rights reserved.
import os
import sys
import subprocess
from argument import parse_args

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    main_py = os.path.join(model_dir, "main.py")

    cmd = [
        sys.executable,
        main_py,
        "--coco_path", args.coco_path,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output_dir", args.output_dir,
    ]
    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=model_dir)
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
