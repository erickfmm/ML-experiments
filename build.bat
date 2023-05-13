docker build -t erickfmm/mlexp:latest .
docker tag erickfmm/mlexp erickfmm/mlexp


docker run -it --rm -v %CD%/:/usr/src/app/ -w /usr/src/app/ erickfmm/mlexp