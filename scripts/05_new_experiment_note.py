#!/usr/bin/env python
"""Create a Markdown experiment note draft for one run."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--docs-root", default="docs/experiments")
    args = parser.parse_args()

    run_dir = Path(args.run_root) / args.run_id
    docs_root = Path(args.docs_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    out_path = docs_root / f"{args.run_id}.md"
    out_path.write_text(
        "\n".join(
            [
                f"# {args.run_id}",
                "",
                "## 目标",
                "",
                "## 数据",
                "",
                "## 配置",
                f"- run_dir: `{run_dir}`",
                "",
                "## 命令",
                "",
                "## 结果",
                f"```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```",
                "",
                "## 结论",
                "",
                "## 下一步",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()

