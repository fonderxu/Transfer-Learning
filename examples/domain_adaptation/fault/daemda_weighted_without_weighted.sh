#!/usr/bin/env bash

# Cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off .1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/cwru


# Gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off .1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/gearbox


# Jnu
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off .1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/jnu
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off .1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/jnu
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_ablation_without_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off .1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted_ablation_without_weighted/jnu
