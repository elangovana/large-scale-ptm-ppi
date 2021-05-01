```bash
npm install aws-cdk@1.96.0
```

## Set up

```
pip install -r ./infra/src/requirements.txt
```

## CDK Deploy

### Deploy Repos stack

This stack creates code commit and ECR repo .

This deploys the stack into the aws profile pointed to using the example named profile "default"

```bash
     export PYTHONPATH=./infra/src
     cdk --app "python infra/src/app.py" deploy  ppi-Repos  --parameters DockerRepoName=ppi --profile default 
```

Make a note of the output arns e.g below to use in the next step

```text
Ppiaimed-Repos.OutputECRARN = arn:aws:ecr:us-west-2:11111:repository/ppi
Ppiaimed-Repos.OutputECRARN =  arn:aws:secretsmanager:us-west-2:111:secret:githubauthCD82D024-18om8waIrUBI-JRbHkQ

```

### Deploy CI Pipeline stack

To deploy a continous integration pipeline using AWS codepipeline to build a docker image and push to ecr created in the
previous step

```bash
cdk --app "python infra/src/app.py" deploy  ppi-CIPipeline --parameters  GithubUrl="https://github.com/elangovana/ppi-aimed"  --parameters BranchName=main  --parameters DockerRepoArn=arn:aws:ecr:us-west-2:11111:repository/ppi  --parameters secretarn=arn:aws:secretsmanager:us-west-2:111:secret:githubauthCD82D024-18om8waIrUBI-JRbHkQ
```
