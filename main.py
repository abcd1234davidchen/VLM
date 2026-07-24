"""Legacy default entry point for multiclass GRU training."""

from _compat import ensure_src_path

ensure_src_path()

from vlm.training.multiclass import main as train_main


def main() -> None:
    train_main()


if __name__ == "__main__":
    main()
