#!/bin/bash

docker build -t fyp-image .

docker run --name fyp-container -d -v $(pwd)/src:/code/src -v $(pwd)/data:/code/data fyp-image