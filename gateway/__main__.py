"""Run the gateway: python -m gateway --config config.yaml --port 8000"""

import argparse
import logging

import uvicorn

from .app import create_app
from .config import load


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible LLM gateway")
    ap.add_argument("--config", default="config.example.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(load(args.config)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
