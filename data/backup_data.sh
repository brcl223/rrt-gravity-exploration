#!/bin/bash

mv current data.bkup.$(date "+%Y-%m-%d-%H_%M_%S")
mkdir current
mkdir current/cache
mkdir current/data-cleaned
mkdir current/data-initial
mkdir current/data-nn
mkdir current/graphs
mkdir current/logs
mkdir current/metadata
mkdir current/results
mkdir current/weights-initial
mkdir current/weights-trained
