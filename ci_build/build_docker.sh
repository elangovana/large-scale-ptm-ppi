DOCKER_BASE_DIR=$1
IMAGE_REPO=$2 # e.g. of image repo 111111.dkr.ecr.us-east-2.amazonaws.com/image
# Note: The account needs to match the pytorch release images, see https://github.com/aws/deep-learning-containers/blob/master/available_images.md
PYTORCH_DOCKER_ACCOUNT=763104351884

AWS_ACCOUNT_ID=`echo $IMAGE_REPO | cut -f 1 -d "."`
ECR_REGION=`echo $IMAGE_REPO | cut -f 4 -d "."`
PYTORCH_DOCKER_ACCOUNT_URL=${PYTORCH_DOCKER_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com
echo Building the Docker image $IMAGE_REPO from base ${PYTORCH_DOCKER_ACCOUNT_URL}...


## TODO: Automate version tagging based on datetime for now, ideally should be tied to release tags
VERSION=$(date '+%Y%m%d%H%M')
device=gpu

LATEST_TAG=$device-latest
VERSION_TAG=$device-$VERSION

# Login to pytorch ECR before build
# aws ecr get-login-password --region region | docker login --username AWS --password-stdin ${PYTORCH_DOCKER_ACCOUNT_URL}
$(aws ecr get-login --no-include-email --region $ECR_REGION --registry-ids ${PYTORCH_DOCKER_ACCOUNT} )

docker build -t $IMAGE_REPO:$LATEST_TAG   -f docker/Dockerfile $DOCKER_BASE_DIR --build-arg device=$device --build-arg account_url=${PYTORCH_DOCKER_ACCOUNT_URL}
docker tag $IMAGE_REPO:$LATEST_TAG $IMAGE_REPO:$VERSION_TAG

# Log into ecr
echo Logging in to Amazon ECR...
$(aws ecr get-login --no-include-email --region $ECR_REGION)

echo Pushing the Docker image...
docker push $IMAGE_REPO:$LATEST_TAG
docker push $IMAGE_REPO:$VERSION_TAG