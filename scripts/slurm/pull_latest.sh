#!/bin/bash

#SBATCH -t 0-00:30:00
#SBATCH -p performance
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=projection_pull_latest
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

usage() {
    echo "Usage: $0 -t <tag> [-r <registry>] [-b <branch>] [-s]"
    echo "  -t <tag>       : Specify the tag for the projection."
    echo "  -r <registry>  : Specify the registry (default: docker://ipoleprojection/projection)."
    echo "  -b <branch>    : Specify the branch (default: main). The branch ID will be used as the tag, with '/' replaced by '-'."
    echo "  -s            : Skip git operations, only do singularity pull."
    exit 1
}

REGISTRY="docker://ipoleprojection/projection"
BRANCH="main"
SKIP_GIT=false

while getopts ":t:r:b:s" opt; do
  case $opt in
    t)
      TAG=$OPTARG
      ;;
    r)
      REGISTRY=$OPTARG
      ;;
    b)
      BRANCH=$OPTARG
      ;;
    s)
      SKIP_GIT=true
      ;;
    *)
      usage
      ;;
  esac
done

if [ -n "$BRANCH" ]; then
    TAG=$(echo "$BRANCH" | tr '/' '-')
fi

if [ -z "$TAG" ]; then
    echo "Error: Tag parameter is required."
    usage
fi

if [ "$SKIP_GIT" = false ]; then
    cd $HOME/projection || { echo "Error: Directory $HOME/projection does not exist."; exit 1; }
    git reset --hard
    git pull origin "$BRANCH"
fi

cd $HOME || { echo "Error: Could not navigate to $HOME."; exit 1; }
singularity pull $REGISTRY:$TAG
