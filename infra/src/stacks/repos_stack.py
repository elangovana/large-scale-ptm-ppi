from aws_cdk import core, aws_ecr, aws_secretsmanager
from aws_cdk.core import CfnParameter, CfnOutput


class ReposStack(core.Stack):
    """
    This sets up code commit, ecr repo..
    """

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        cfn_docker_repo_parameter = CfnParameter(self, "DockerRepoName", type="String",
                                                 default="ppiaimed",
                                                 description="The docker ecr repo to use")
        docker_repo_name = cfn_docker_repo_parameter.value_as_string

        # Secrets Manager
        secret = aws_secretsmanager.Secret(self, "github_auth")

        # Docker
        ecr = aws_ecr.Repository(scope=self, id="ECR", repository_name=docker_repo_name)

        # CFN output
        CfnOutput(scope=self, id="OutputECRARN", value=ecr.repository_arn)

        # CFN output
        CfnOutput(scope=self, id="github_secret", value=secret.secret_arn)
