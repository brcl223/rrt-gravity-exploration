#!/bin/bash

START_DOF=2
END_DOF=2
START=2000
END=20000
STEP=2000
CMDS=x

for((i=START;i<=END;i+=STEP)); do
    for((j=START_DOF;j<=END_DOF;j+=1)); do
        ./runner.sh -m "panda-${j}dof" -i "$i" "-$CMDS" &> /dev/null &
    done
    # Run each of these processes in the background,
    # but to avoid too many processes, wait for each
    # of these to finish up first
    echo "Waiting on round $i to finish..."
    wait < <(jobs -p)
done

# CMDS=t
# for((i=START;i<=END;i+=STEP)); do
#     for((j=START_DOF;j<=END_DOF;j+=1)); do
#         echo "Currently running iters: ${i} for panda-${j}dof"
#         ./runner.sh -m "panda-${j}dof" -i "$i" "-$CMDS" &> /dev/null #"./data/logs/panda-${j}dof-iters.log"
#     done
# done
