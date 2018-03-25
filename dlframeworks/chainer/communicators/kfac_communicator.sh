#!/bin/bash

mpirun -np 8 -host localhost:8 python -W ignore kfac_communicator.py
