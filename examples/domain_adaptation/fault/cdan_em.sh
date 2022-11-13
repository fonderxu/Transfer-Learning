#!/usr/bin/env bash

# Cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch -b 220 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/cwru


# Gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch -b 205 --epochs 50 -i 100 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/gearbox


# Jnu
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch -b 220 --epochs 50 -i 200 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/jnu
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch -b 220 --epochs 50 -i 200 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/jnu
CUDA_VISIBLE_DEVICES=0 python cdan_em.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch -b 220 --epochs 50 -i 200 -lr 0.001 --weight-decay 1e-4 --trade-off 1. --seed 0 -p 50 --log logs/cdan_em[with_a]/jnu




