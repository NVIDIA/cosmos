import pickle
import random

import numpy as np
import torch


def get_rand_state_dict() -> dict[str, torch.Tensor]:
    """
    Get the random state dictionary. used to save the random state to a checkpoint.
    """
    numpy_packed_len, numpy_packed_bytes = pack_numpy_state(np.random.get_state())
    random_packed_len, random_packed_bytes = pack_random_state(random.getstate())

    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "numpy_packed_len": torch.tensor(numpy_packed_len, dtype=torch.long),  # []
        "numpy_packed_bytes": torch.frombuffer(numpy_packed_bytes, dtype=torch.uint8),  # [MAX_PACKED_LEN]
        "random_packed_len": torch.tensor(random_packed_len, dtype=torch.long),  # []
        "random_packed_bytes": torch.frombuffer(random_packed_bytes, dtype=torch.uint8),  # [MAX_PACKED_LEN]
    }


def set_rand_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Set the random state dictionary. used to restore the random state from a checkpoint.
    """
    torch.set_rng_state(state_dict["torch"])
    torch.cuda.set_rng_state(state_dict["torch_cuda"])
    np.random.set_state(
        unpack_numpy_state(state_dict["numpy_packed_len"].item(), bytes(state_dict["numpy_packed_bytes"].tolist()))
    )
    random.setstate(
        unpack_random_state(state_dict["random_packed_len"].item(), bytes(state_dict["random_packed_bytes"].tolist()))
    )


# MAX padding length for the random state
# numpy and python random state are based on Mersenne Twister algorithm and state has aprox 625 integers.
# When we convert this data through pickle generated output can be of variable size.
# Pad the output to a fixed size to ensure that the size of the random state is always the same.
# Fixed size buffer is required for DCP checkpoint functions as it uses pinned preallocated memory to copy the checkpoint data.
MAX_PACKED_LEN = 4096


def pad_packed_bytes(packed_bytes: bytes) -> bytearray:
    padded = packed_bytes.ljust(MAX_PACKED_LEN, b"\0")

    return bytearray(padded)


def pack_numpy_state(state: tuple[int, ...]) -> tuple[int, bytearray]:
    packed_bytes = pickle.dumps(state)
    return len(packed_bytes), pad_packed_bytes(packed_bytes)


def pack_random_state(state: tuple[int, ...]) -> tuple[int, bytearray]:
    packed_bytes = pickle.dumps(state)
    return len(packed_bytes), pad_packed_bytes(packed_bytes)


def unpack_numpy_state(packed_len: int, packed_bytes: bytearray) -> tuple[int, ...]:
    packed_bytes = packed_bytes[:packed_len]
    # unpickle the state
    return pickle.loads(packed_bytes)


def unpack_random_state(packed_len: int, packed_bytes: bytearray) -> tuple[int, ...]:
    packed_bytes = packed_bytes[:packed_len]
    # unpickle the state
    return pickle.loads(packed_bytes)


if __name__ == "__main__":
    print("--- Initializing Seeds ---")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # random operations to advance the state from the initial seed
    for _ in range(10):
        _ = torch.randn(1)
        _ = np.random.standard_normal()
        _ = random.gauss(0, 1)

    print("Capturing state...")
    state_dict = get_rand_state_dict()

    # Ensure tensors are actually padded to fixed size
    assert state_dict["numpy_packed_bytes"].numel() == MAX_PACKED_LEN, (
        f"Numpy buffer size mismatch! Expected {MAX_PACKED_LEN}, got {state_dict['numpy_packed_bytes'].numel()}"
    )
    assert state_dict["random_packed_bytes"].numel() == MAX_PACKED_LEN, (
        f"Random buffer size mismatch! Expected {MAX_PACKED_LEN}, got {state_dict['random_packed_bytes'].numel()}"
    )
    print(f"State captured. Buffers verified at {MAX_PACKED_LEN} bytes.")

    # generate a sequence of random numbers immediately after saving.
    print("Generating Ground Truth Sequence (A)...")
    seq_a = []
    for _ in range(5):
        vals = (
            torch.randn(1).item(),
            np.random.rand(),
            np.random.uniform(),
            random.random(),
        )
        seq_a.append(vals)

    # modify the state by generating random numbers
    print("Modifying state (Generating junk)...")
    for _ in range(20):
        _ = torch.rand(1)
        _ = np.random.rand()
        _ = random.random()

    # restore the state
    print("Restoring random state...")
    set_rand_state_dict(state_dict)

    # If restore works, this must match Sequence A exactly.
    print("Generating Post-Restore Sequence (B)...")
    seq_b = []
    for _ in range(5):
        vals = (
            torch.randn(1).item(),
            np.random.rand(),
            np.random.uniform(),
            random.random(),
        )
        seq_b.append(vals)

    print("\n--- Verification Results ---")
    all_match = True
    for i, (truth, actual) in enumerate(zip(seq_a, seq_b)):
        # Compare all values in the tuple with small tolerance for float precision
        row_match = all(abs(t - a) < 1e-9 for t, a in zip(truth, actual))
        print(f"  Row {i}: {truth} vs {actual} - {row_match}")
        if row_match:
            print(f"Step {i}: MATCH ")
        else:
            print(f"Step {i}: MISMATCH ")
            print(f"  Expected: {truth}")
            print(f"  Actual:   {actual}")
            all_match = False

    if all_match:
        print("\n All random states (Torch, Numpy, Python) restored successfully.")
    else:
        raise RuntimeError("Test Failed: Random states did not match after restore.")
