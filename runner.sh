set -e

EXECUTE=unset

usage() {
    echo "Usage: runner.sh -m <model> [-i <sample_iters>] [-bceght]"
    echo
    echo "Options:"
    echo "-b|--backup <dirname>: Backup data in new directory"
    echo "-c|--clean: Clean points (must have previously explored)"
    echo "-e|--explore: Explore new points"
    echo "-g|--graph: Graph Loss and Variance of trained points"
    echo "-h|--help: This menu"
    echo "-i|--iters: Number of sample node iterations"
    echo "-m|--model: Panda model to use"
    echo "-s|--sample: Sample 100,000 test points for testing NN"
    echo "-t|--train: Train cleaned points with NN"
    echo "-x|--test: Test points"
}

PARSED_ARGUMENTS=$(getopt -a -n runner.sh -o b:ceghi:m:stx --long backup:,clean,explore,graph,help,model:,iters:,sample,train,test -- "$@")

eval set -- "$PARSED_ARGUMENTS"

BACKUP=0
CLEAN=0
EXPLORE=0
GRAPH=0
ITERS=""
SAMPLE_TEST_POINTS=0
TRAIN=0
TEST=0
MODEL=""

while :
do
    case "$1" in
        -b | --backup) BACKUP=1 ; shift ; shift ;;
        -c | --clean) CLEAN=1 ; shift ;;
        -e | --explore) EXPLORE=1 ; shift ;;
        -g | --graph) GRAPH=1 ; shift;;
        -h | --help) usage ; exit 0 ;;
        -m | --model) MODEL="$2" ; shift ; shift ;;
        -i | --iters) ITERS="$2" ; shift ; shift ;;
        -s | --sample) SAMPLE_TEST_POINTS=1 ; shift ;;
        -t | --train) TRAIN=1 ; shift ;;
        -x | --test) TEST=1 ; shift ;;
        -- ) shift ; break ;;
        *) echo "Unexpected option: $1"
           usage
           exit 1 ;;
    esac
done

if [ $BACKUP -eq 1 ]; then
    echo "Backup not currently implemented. Exiting..."
    exit 2
fi

if [ $EXPLORE -eq 1 ]; then
   poetry run python src/explore.py "$MODEL" rrt_star "$ITERS"
   poetry run python src/explore.py "$MODEL" rrt "$ITERS"
   poetry run python src/explore.py "$MODEL" random "$ITERS"
fi

if [ $CLEAN -eq 1 ]; then
    poetry run python src/grav_point_cleaner.py "$MODEL" rrt "$ITERS"
    poetry run python src/grav_point_cleaner.py "$MODEL" rrt_star "$ITERS"
    poetry run python src/grav_point_cleaner.py "$MODEL" random "$ITERS"
fi

if [ $TRAIN -eq 1 ]; then
    poetry run python src/grav_explore_trainer.py "$MODEL" rrt "$ITERS"
    poetry run python src/grav_explore_trainer.py "$MODEL" rrt_star "$ITERS"
    poetry run python src/grav_explore_trainer.py "$MODEL" random "$ITERS"
fi

if [ $GRAPH -eq 1 ]; then
    poetry run python src/graph_explore_results.py "$MODEL" rrt_star avg_loss "$ITERS"
    poetry run python src/graph_explore_results.py "$MODEL" rrt_star avg_var "$ITERS"
    poetry run python src/graph_explore_results.py "$MODEL" rrt avg_loss "$ITERS"
    poetry run python src/graph_explore_results.py "$MODEL" rrt avg_var "$ITERS"
    poetry run python src/graph_explore_results.py "$MODEL" random avg_loss "$ITERS"
    poetry run python src/graph_explore_results.py "$MODEL" random avg_var "$ITERS"
fi

if [ $SAMPLE_TEST_POINTS -eq 1 ]; then
    poetry run python src/explore.py "$MODEL" random 100000
    poetry run python src/grav_point_cleaner.py "$MODEL" random 100000
fi

if [ $TEST -eq 1 ]; then
    poetry run python src/test_nn_manual.py "$MODEL" rrt "$ITERS"
    poetry run python src/test_nn_manual.py "$MODEL" rrt_star "$ITERS"
    poetry run python src/test_nn_manual.py "$MODEL" random "$ITERS"
fi
