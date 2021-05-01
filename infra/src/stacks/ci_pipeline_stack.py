from aws_cdk import core
from aws_cdk.core import CfnParameter

from custom_constructs.ci_pipeline_dockerise_construct import CIPipelineDockeriseConstruct


class CIPipelineStack(core.Stack):
    """
    This sets up the CI pipeline that builds and tests python and cdk.
    """

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        cfn_source_repo_parameter = core.CfnParameter(self, "GithubUrl", type="String",
                                                      default="https://github.com/elangovana/ppi-aimed",
                                                      description="The github url. e.g. https://github.com/elangovana/ppi-aimed")
        source_repo_arn = cfn_source_repo_parameter.value_as_string

        cfn_auth_arn_parameter = core.CfnParameter(self, "secretarn", type="String",
                                                   description="The arn of the secret manager with gthub auth token")
        secret_arn = cfn_auth_arn_parameter.value_as_string

        # branch name
        cfn_branch_name_parameter = core.CfnParameter(self, "BranchName", type="String",
                                                      description="The code branch, e.g. master", default="main")
        branch_name = cfn_branch_name_parameter.value_as_string

        cfn_buildspec_parameter = CfnParameter(self, "BuildSpec", default="ci_build/buildspec.build.yml",
                                               type="String",
                                               description="The name of the codebuild spec file. e.g. codebuild/buildspec.yml")

        buildspec = cfn_buildspec_parameter.value_as_string

        cfn_build_image_parameter = CfnParameter(self, "BuildImage", type="String",
                                                 description="The codebuild image as specified in https://docs.aws.amazon.com/codebuild/latest/userguide/codebuild-env-ref-available.html. e.g. aws/codebuild/amazonlinux2-x86_64-standard:2.0",
                                                 default="aws/codebuild/standard:4.0")
        build_image = cfn_build_image_parameter.value_as_string

        # CFN parameters - docker repo
        cfn_docker_repo_arn_parameter = core.CfnParameter(self, "DockerRepoArn", type="String",
                                                          description="The docker image repo arn to use")

        docker_repo_arn = cfn_docker_repo_arn_parameter.value_as_string

        CIPipelineDockeriseConstruct(self, id="CodePipeline", repo_url=source_repo_arn, secret_arn=secret_arn,
                                     branch=branch_name, build_image=build_image, buildspec=buildspec,
                                     docker_repository_arn=docker_repo_arn)
