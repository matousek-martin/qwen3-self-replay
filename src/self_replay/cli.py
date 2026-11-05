"""Small command-line surface for preparation, methods, evaluation and cloud jobs."""

import argparse
import json
import shlex
from pathlib import Path

from self_replay.methods import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("recipes", help="List runnable scientific recipes")
    p = sub.add_parser("init", help="Create an isolated local workspace; no network or GPU")
    p.add_argument("workspace", type=Path)
    p.add_argument("--recipe", choices=runtime.recipes(), default="self-replay")
    p = sub.add_parser("verify", help="Verify the materialized workspace source")
    p.add_argument("workspace", type=Path)
    p = sub.add_parser("train", help="Run a recipe on an existing worker; never rent a GPU")
    p.add_argument("recipe", choices=runtime.recipes())
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("prepare", help="Run a dataset preparation recipe")
    p.add_argument(
        "recipe",
        choices=[
            "initial",
            "select",
            "four-domain",
            "self-replay",
            "opencoder",
            "coding-frontier",
            "coding-curriculum",
        ],
    )
    p.add_argument("--workspace", type=Path, required=True)
    p = sub.add_parser("stage-data", help="Verify and copy exact frozen train/development data")
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--plan", type=Path)
    sub.add_parser("evaluate", help="Pass arguments to the unchanged official evaluator")
    p = sub.add_parser("compare", help="Official paired-prompt comparison")
    p.add_argument("base", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("cloud", help="Single canonical controller for RunPod and Vast")
    p.add_argument("operation", choices=["plan", "run", "status", "watch", "guard", "quote"])
    p.add_argument("--workspace", type=Path)
    p.add_argument("--spec", type=Path)
    p.add_argument("--approval", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--name")
    p.add_argument("--provider", choices=["runpod", "vast"], default="vast")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    p.add_argument("--count", type=int, default=1)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra and args.action not in ("train", "prepare", "evaluate"):
        parser.error("Unrecognized arguments: " + " ".join(extra))
    if args.action == "recipes":
        for name, spec in runtime.recipes().items():
            print(f"{name:25} {spec['kind']}")
    elif args.action == "init":
        print(runtime.initialize(args.workspace, args.recipe))
    elif args.action == "verify":
        print(json.dumps(runtime.verify(args.workspace), indent=2))
    elif args.action == "train":
        cmd = runtime.train(args.recipe, args.workspace, args.data, args.out, extra, args.dry_run)
        if args.dry_run:
            print(shlex.join(cmd))
    elif args.action == "prepare":
        from self_replay.data.prepare import prepare

        prepare(args.workspace, args.recipe, extra)
    elif args.action == "stage-data":
        from self_replay.data.prepare import stage_frozen

        print(stage_frozen(args.source, args.destination, args.plan))
    elif args.action == "evaluate":
        from self_replay.evaluation.official import evaluate

        evaluate(extra)
    elif args.action == "compare":
        from self_replay.evaluation.official import compare

        compare(args.base.resolve(), args.candidate.resolve(), args.out.resolve())
    elif args.action == "cloud":
        from self_replay.cloud import controller, jobs

        if args.operation == "plan":
            if not args.spec or not args.out:
                parser.error("cloud plan needs --spec and --out")
            print(json.dumps(jobs.plan(args.spec, args.out), indent=2))
        else:
            if not args.workspace:
                parser.error("cloud operation needs --workspace")
            if args.operation == "run":
                if not args.spec or not args.approval:
                    parser.error("cloud run needs --spec and --approval")
                controller.launch(args.workspace, args.spec, args.approval)
            elif args.operation == "status":
                print(json.dumps(controller.inspect(args.workspace), indent=2))
            elif args.operation == "watch":
                if not args.name:
                    parser.error("cloud watch needs --name")
                controller.watch(args.workspace, args.name)
            elif args.operation == "guard":
                controller.guard(args.workspace)
            else:
                print(
                    json.dumps(
                        controller.quote(args.workspace, args.provider, args.gpu, args.count),
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
