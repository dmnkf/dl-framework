#!/bin/bash

# Default values
IMAGE_NAME="projection"
TAG="latest"
CONVERT_TO_SINGULARITY=false
PUSH_TO_REGISTRY=false
REGISTRY_URL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --singularity)
            CONVERT_TO_SINGULARITY=true
            shift
            ;;
        --push)
            PUSH_TO_REGISTRY=true
            shift
            ;;
        --registry)
            REGISTRY_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build Docker image
echo "Building Docker image ${IMAGE_NAME}:${TAG}..."
docker build --pull -t "${IMAGE_NAME}:${TAG}" -f docker/Dockerfile .

if [ "$PUSH_TO_REGISTRY" = true ]; then
    if [ -z "$REGISTRY_URL" ]; then
        echo "Error: Registry URL not provided. Use --registry to specify."
        exit 1
    fi
    echo "Logging into Docker registry ${REGISTRY_URL}..."
    docker login "$REGISTRY_URL"
    echo "Pushing to registry ${REGISTRY_URL}..."
    docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY_URL}/${IMAGE_NAME}:${TAG}"
    docker push "${REGISTRY_URL}/${IMAGE_NAME}:${TAG}"

    if [ "$TAG" = "latest" ]; then
        echo "Tagging and pushing the latest version..."
        docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY_URL}/${IMAGE_NAME}:latest"
        docker push "${REGISTRY_URL}/${IMAGE_NAME}:latest"
    fi
fi

if [ "$CONVERT_TO_SINGULARITY" = true ]; then
    echo "Converting to Singularity image using singularityware/docker2singularity..."
    mkdir -p singularity_images
    docker run --privileged -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd)/singularity_images:/output singularityware/docker2singularity ${IMAGE_NAME}:${TAG}
    echo "Singularity image created: singularity_images/${IMAGE_NAME}_${TAG}.sif"
fi

echo "Build process completed!"
