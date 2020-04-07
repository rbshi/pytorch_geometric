#!/bin/bash

PYG_ROOT_PATH=$(cd `dirname $0` || exit; pwd)/..

# check arguments
if [ $# != 2 ]; then
    echo " Usage: ./run_gcn.sh model_config_file.csv result.txt" >&2
    exit 1
fi

if [ ! -f $1 ]; then
    echo "input file does not exist!" >&2
    exit
fi

# check the pip command.
if ! [ -x "$(command -v pip)" ]; then
  echo 'Error: pip is not installed.' >&2
  exit 1
fi

echo "STEP1: Install dependencies."
pip install torch torchvision --user
# change +cu101 to your current CUDA version.
pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-cluster==latest+cu101 torch-spline-conv==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0.html --user
# setup local PyG
cd  $PYG_ROOT_PATH || exit
python setup.py install --user

echo "STEP2: Download pre-trained models"
cd $PYG_ROOT_PATH/models || exit
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Cora-16.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/CiteSeer-16.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Pubmed-16.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Reddit-64.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Nell-64.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Cora-128.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/CiteSeer-128.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Pubmed-128.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Reddit-128.pth
wget -nc https://github.com/rbshi/public/raw/master/models/gcn/Nell-128.pth
cd  $PYG_ROOT_PATH/misc || exit

echo "STEP3: Test evaluation"
rm -f $2
IFS=","
sed 1d $1 | while read dataset hiddensize encpu engpu na1 na2 ; do
  echo -e "dataset: $dataset \t hiddensize: $hiddensize"
  time_cpu=0; time_gpu=0
  if [ $encpu = 1 ]; then
    # get CPU time
    res_cpu=$(python $PYG_ROOT_PATH/examples/gcn.py --dataset $dataset --hsize $hiddensize --runmode test --device cpu)
    time_cpu=${res_cpu: 10}
  fi
  if [ $engpu = 1 ]; then
    # get GPU time
    res_gpu=$(python $PYG_ROOT_PATH/examples/gcn.py --dataset $dataset --hsize $hiddensize --runmode test --device cuda)
    time_gpu=${res_gpu: 10}
  fi
  echo -e "$time_cpu\t $time_gpu" >> $2
done

echo "Done"


