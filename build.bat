docker build -t erickfmm/mlexp:latest .
docker tag erickfmm/mlexp erickfmm/mlexp


docker run -it --rm -v %CD%/train_data:/usr/src/app/train_data erickfmm/mlexp