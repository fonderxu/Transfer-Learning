# Experiment
#!/usr/bin/env bash
# Cwru
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/cwru
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/cwru
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/cwru
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/cwru -d Cwru -s D -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/cwru


# Gearbox
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/gearbox
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/gearbox
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/gearbox
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s D -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/gearbox


# Jnu
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/jnu
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/jnu
CUDA_VISIBLE_DEVICES=0 python dataset_distribution.py D:/repository/pycharm/datasets/jnu -d Jnu -s C -t C -a alexnet --scratch --trade-off 1. -b 220 -i0.00 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dataset_distribution/jnu