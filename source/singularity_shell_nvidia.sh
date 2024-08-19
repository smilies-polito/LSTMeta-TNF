#!/bin/bash

singularity shell --nv --fakeroot --bind /home/users/smeriglio/:/mnt $1
