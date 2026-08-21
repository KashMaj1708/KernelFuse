"""Phase 1 entry point — same harness, naive backend only."""

from test_rmsnorm_correctness import main
import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--backend", "naive", *sys.argv[1:]]
    raise SystemExit(main())
