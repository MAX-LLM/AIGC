"""Prepare and check local retrieval resources before starting the web app."""

import argparse
import json
import os

from tools import warm_knowledge_retrieval
from model_config import load_environment


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true",
        help="Allow downloading the public embedding model during setup.",
    )
    args = parser.parse_args()
    if args.download:
        os.environ["MND_EMBEDDING_LOCAL_ONLY"] = "0"
        if os.environ.get("MND_RETRIEVAL_MODE", "").lower() not in {"hybrid", "semantic"}:
            os.environ["MND_RETRIEVAL_MODE"] = "hybrid"
    print(json.dumps(warm_knowledge_retrieval(), indent=2))


if __name__ == "__main__":
    main()
