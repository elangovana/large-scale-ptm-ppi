import argparse
from enum import Enum

from aws_cdk import core

from stacks.ci_pipeline_stack import CIPipelineStack
from stacks.repos_stack import ReposStack


class StackType(str, Enum):
    CIPipeline = "CIPipeline"
    Repos = "Repos"


def get_stack_name(stack_prefix: str, stack_type: StackType):
    return f"{stack_prefix}-{stack_type}"


def run(stack_prefix):
    app = core.App()
    print(get_stack_name(stack_prefix, StackType.CIPipeline))

    CIPipelineStack(app, "cipipelinestack", stack_name=get_stack_name(stack_prefix, StackType.CIPipeline))
    ReposStack(app, "reposstack", stack_name=get_stack_name(stack_prefix, StackType.Repos))

    app.synth()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", help="Pass an optional stack prefix", required=False, default="ppi")

    args = parser.parse_args()
    run(args.stack)
