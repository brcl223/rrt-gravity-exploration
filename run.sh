#!/bin/bash

START_DOF=5
END_DOF=7
START=2000
END=20000
STEP=2000
CMDS=e

# Now collect data for larger node count, higher DOF systems
# Just collect for 5 and 7 to save time
for((i=START;i<=END;i+=STEP)); do
    for((j=START_DOF;j<=END_DOF;j+=2)); do
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
#     for((j=START_DOF;j<=END_DOF;j+=2)); do
#         echo "Currently running iters: ${i} for panda-${j}dof"
#         ./runner.sh -m "panda-${j}dof" -i "$i" "-$CMDS" &> /dev/null #"./data/logs/panda-${j}dof-iters.log"
#     done
# done

# CMDS=x
# for((i=START;i<=END;i+=STEP)); do
#     for((j=START_DOF;j<=END_DOF;j+=2)); do
#         ./runner.sh -m "panda-${j}dof" -i "$i" "-$CMDS" &> /dev/null &
#     done
#     # Run each of these processes in the background,
#     # but to avoid too many processes, wait for each
#     # of these to finish up first
#     echo "Waiting on round $i to finish..."
#     wait < <(jobs -p)
# done
