# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""One-shot generator for the 8 IonQ evaluation keys (offline FNV-1a tags)."""

SECRET = "qector-ionq-1.7.8-7day-eval-v1"


def fnv1a64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) % (1 << 64)
    return h


def tag(key_id: str, date: str) -> str:
    return format(fnv1a64(f"{SECRET}|{key_id}|{date}".encode()), "016x")


def main() -> None:
    for n in range(1, 8):
        key_id = f"EVAL0{n}"
        print(f"QIONQ7-{key_id}-20260909-{tag(key_id, '20260909')}")
    print(f"QIONQ7-DEV000-00000000-{tag('DEV000', '00000000')}")


if __name__ == "__main__":
    main()
