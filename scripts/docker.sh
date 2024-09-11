#!/bin/bash

# Download tensorflow docker...
docker build -t mh_phd_works .

# Run tensorflow docker...
docker run -v ${PWD}:/usr/src/app/ --rm -it mh_phd_works bash -l