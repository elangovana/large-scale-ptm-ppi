DOCKER_BASE_DIR=$1
IMAGE_REPO=$2 # e.g. of image repo 111111.dkr.ecr.us-east-2.amazonaws.com/image

AWS_ACCOUNT_ID=`echo $IMAGE_REPO | cut -f 1 -d "."`

ECR_REGION=`echo $IMAGE_REPO | cut -f 4 -d "."`



echo Building the Docker image $IMAGE_REPO ...


## TODO: Automate version tagging based on datetime for now, ideally should be tied to release tags
LATEST_TAG=latest
VERSION_TAG=$(date '+%Y%m%d%H%M')

docker build -t $IMAGE_REPO:$LATEST_TAG   -f docker/Dockerfile $DOCKER_BASE_DIR
docker tag $IMAGE_REPO:$LATEST_TAG $IMAGE_REPO:$VERSION_TAG

# Log into ecr
echo Logging in to Amazon ECR...
$(aws ecr get-login --no-include-email --region $ECR_REGION)

echo Pushing the Docker image...
docker push $IMAGE_REPO:$LATEST_TAG
docker push $IMAGE_REPO:$VERSION_TAG