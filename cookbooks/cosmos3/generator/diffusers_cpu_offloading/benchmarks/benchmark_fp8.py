"""Full Cosmos3-Nano FP8 benchmark using the checkpoint's native Diffusers policy."""

from runner import main

if __name__ == "__main__":
    raise SystemExit(main(precision="fp8"))
