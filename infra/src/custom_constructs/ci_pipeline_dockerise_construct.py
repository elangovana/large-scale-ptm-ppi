from aws_cdk import (
    core,
    aws_codepipeline_actions,
    aws_codepipeline,
    aws_codebuild, aws_iam, aws_ecr, aws_secretsmanager)
from aws_cdk.aws_codepipeline import Pipeline
from aws_cdk.aws_codepipeline_actions import GitHubTrigger
from aws_cdk.aws_iam import PolicyStatement, AccountPrincipal
from aws_cdk.core import Stack, Fn, Aws


class CIPipelineDockeriseConstruct(Pipeline):

    def __init__(self, scope: core.Construct, id: str, repo_url: str, secret_arn: str, branch: str, buildspec: str,
                 build_image: str,
                 docker_repository_arn: str) -> None:
        """
        Create a code pipeline with github repo as source to build docker images and publish any artifacts to s3
        :param scope:
        :param id:
        :param repo_url:
        :param branch:
        :param buildspec:
        :param build_image:
        :param docker_repository_arn:
        """
        super().__init__(scope, id,
                         restart_execution_on_update=True)

        outh = aws_secretsmanager.Secret.from_secret_arn(id=f"{id}secret", scope=self,
                                                         secret_arn=secret_arn).secret_value

        # Source
        source_artifact = aws_codepipeline.Artifact("source")

        address_no_scheme = Fn.select(1, Fn.split("://", repo_url))
        owner = Fn.select(1, Fn.split("/", address_no_scheme))
        repo = Fn.select(2, Fn.split("/", address_no_scheme))
        repo_action = aws_codepipeline_actions.GitHubSourceAction(oauth_token=outh, owner=owner, repo=repo,
                                                                  branch=branch, trigger=GitHubTrigger.NONE,
                                                                  output=source_artifact, action_name="source")

        self.add_stage(stage_name="Source", actions=[repo_action])

        # Build Test
        stage_name = "BuildAndTest"
        code_build_project = aws_codebuild.PipelineProject(
            scope,
            "{}CodeBuild".format(stage_name),
            environment=aws_codebuild.BuildEnvironment(
                build_image=aws_codebuild.LinuxBuildImage.from_code_build_image_id(
                    build_image),
                privileged=True),
            build_spec=aws_codebuild.BuildSpec.from_source_filename(buildspec)
        )

        # Docker push & login permissions
        docker_login = aws_iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"])
        code_build_project.add_to_role_policy(docker_login)

        docker_repo_push = aws_iam.PolicyStatement(actions=["ecr:GetDownloadUrlForLayer",
                                                            "ecr:BatchGetImage",
                                                            "ecr:BatchCheckLayerAvailability",
                                                            "ecr:PutImage",
                                                            "ecr:InitiateLayerUpload",
                                                            "ecr:UploadLayerPart",
                                                            "ecr:CompleteLayerUpload"],
                                                   resources=[docker_repository_arn])
        code_build_project.add_to_role_policy(docker_repo_push)

        # Pytorch docker permissions
        pytorch_docker_repo, pytorch_docker_repo_permissions = self._get_pytorch_docker_pull_permissions(id)
        code_build_project.add_to_role_policy(pytorch_docker_repo_permissions)

        # Environment variables
        # Docker Repo
        docker_repo_name = Fn.select(1, Fn.split("/", docker_repository_arn))
        docker_repo = aws_ecr.Repository.from_repository_attributes(scope=self, id="DockerRepo",
                                                                    repository_arn=docker_repository_arn,
                                                                    repository_name=docker_repo_name)
        enviornment_variables_dict = {"docker_image": docker_repo.repository_uri,
                                      "pytorch_training_image": pytorch_docker_repo.repository_uri
                                      }

        env_variables = self._get_codebuild_variables(enviornment_variables_dict)

        # Add code build action
        code_build_variables_namespace = "{}Variables".format(stage_name)
        build_artifact = aws_codepipeline.Artifact("{}Artifacts".format(stage_name))
        build_test_action = aws_codepipeline_actions.CodeBuildAction(outputs=[build_artifact],
                                                                     action_name=f"{stage_name}",
                                                                     project=code_build_project,
                                                                     input=source_artifact,
                                                                     type=aws_codepipeline_actions.CodeBuildActionType.BUILD,
                                                                     run_order=1,
                                                                     variables_namespace=code_build_variables_namespace,
                                                                     environment_variables=env_variables
                                                                     )

        self.add_stage(stage_name="BuildTest", actions=[build_test_action])

        # Publish artifacts
        build_test_actions = [build_test_action]
        outputs = []
        for build_action in build_test_actions:
            outputs.extend(build_action.action_properties.outputs)
        publish_action = aws_codepipeline_actions.S3DeployAction(bucket=self.artifact_bucket,
                                                                 object_key="BuildArtifacts",
                                                                 action_name="PublishArtifacts", input=outputs[0],
                                                                 extract=True

                                                                 )

        self.add_stage(stage_name="PublishArtifacts", actions=[publish_action])

        # Add decrypt permissions
        self.artifact_bucket.encryption_key.add_to_resource_policy(
            PolicyStatement(principals=[AccountPrincipal(Stack.of(self).account)],
                            actions=["kms:Decrypt"]
                            , resources=["*"]
                            )
        )

    def _get_pytorch_docker_pull_permissions(self, id):

        pytorch_map_config = {
            "us-east-2": {
                "trainingrepoarn": "arn:aws:ecr:us-east-2:763104351884:repository/pytorch-training"}
        }

        pytorch_map = core.CfnMapping(scope=self, id=f"{id}pytorchmap", mapping=pytorch_map_config)

        # Provide pull permissions to pytorch
        pytorch_repo_arn = pytorch_map.find_in_map(Aws.REGION, "trainingrepoarn")
        pytorch_repo_name = Fn.select(1, Fn.split("/", pytorch_repo_arn))

        pytorch_docker_repo = aws_ecr.Repository.from_repository_attributes(scope=self, id="PytorchDockerRepo",
                                                                            repository_arn=pytorch_repo_arn,
                                                                            repository_name=pytorch_repo_name)
        # Docker push & login permissions
        pytorch_docker_repo_permissions = aws_iam.PolicyStatement(actions=["ecr:GetDownloadUrlForLayer",
                                                                           "ecr:BatchGetImage",
                                                                           "ecr:BatchCheckLayerAvailability",
                                                                           "ecr:PullImage"],
                                                                  resources=[pytorch_repo_arn])
        return pytorch_docker_repo, pytorch_docker_repo_permissions

    def _get_codebuild_variables(self, enviornment_variables_dict):
        env_variables = {}
        for k, v in enviornment_variables_dict.items():
            env_variables[k] = aws_codebuild.BuildEnvironmentVariable(value=v,
                                                                      type=aws_codebuild.BuildEnvironmentVariableType.PLAINTEXT)
        return env_variables
