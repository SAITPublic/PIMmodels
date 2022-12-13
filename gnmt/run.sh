#!/bin/bash

working_dir=`pwd`
accuracy_test=false
use_pim=false
repo_name="training"
repo_url="https://github.com/mlcommons/training.git"

export PATH=/home/user/.local/bin:$PATH
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

function log()
{
    echo -e " \033[92;1m"$1"\033[m"
}

function err()
{
    echo -e " \033[91;1m"$1"\033[m"
}

function usage()
{
    echo ""
    echo "usage: $0 [options] [argument]"
    echo " Option               Argument"
    echo " --accuracy           measure accuracy (default: performance)"
    echo " --clean              clean submodule repository"
    echo " --use_pim            enable PIM (default: false)"
    echo ""
    exit 0
}

function clone_repo()
{
    git clone $1
    pushd $repo_name
    git checkout 16ad074a2a2c655 -b gnmt-pim
    popd
}

function patch_for_pim()
{
    # Patch to enable PIM operator
    log "patch to enable PIM operator on GNMT model"
    pushd "$working_dir/$repo_name"
    if [ ! -f "$working_dir/$repo_name/rnn_translator/pytorch/run_eval.sh" ]; then
        for patch_file in $working_dir/patch/*.patch
        do
            cp $patch_file .
        done

        for patch_file in ./*.patch
        do
            git am $patch_file
            rm $patch_file
        done
    fi
}

while [ ! -z "$1" ]; do
    case "$1" in
        --accuracy)
	    accuracy_test=true
	    ;;

        --clean)
	    err "warning: clean ${repo_name}"
        rm -rf $repo_name
	    exit 0
	    ;;
        --use_pim)
            use_pim=true
	    ;;
        *)
            usage
	    ;;
    esac
    shift
done

if [ ! -d "$repo_name/rnn_translator" ]; then
    clone_repo $repo_url
    patch_for_pim
    # Lazy git submodule update due to patch required (for ignoring community repo)
    git submodule update --init --recursive

    # Installation of prerequisites
    log "Install python3 requirements"
    pushd "$working_dir/training/rnn_translator/pytorch"
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org sacrebleu
    popd
fi

# Preparing dataset and model
pushd "$working_dir/$repo_name/rnn_translator"
if [ ! -d "data" ]; then
    log "Prepare dataset"
    bash download_dataset.sh
    bash verify_dataset.sh
fi
popd

if [ ! -f "model/model_best.pth" ]; then
    log "Download model weight"
    mkdir -p model
    wget https://zenodo.org/record/2581623/files/model_best.pth --no-check-certificate -O ./model/model_best.pth
fi

pushd "$working_dir/$repo_name/rnn_translator"
if [ ! -d "data" ]; then
    err "cannot find data folder"
    exit 0
fi
if [ ! -f "pytorch/model_best.pth" ]; then
    ln -s $working_dir/model/model_best.pth ./pytorch/model_best.pth
fi
popd

# Run model
pushd "$working_dir/$repo_name/rnn_translator/pytorch"
work_dir=`pwd`
options="--input ../data/newstest2014.tok.clean.bpe.32000.en \
         --output output_file \
         --model model_best.pth \
         --reference ../data/newstest2014.de \
         --beam-size 1 \
         --batch-size 1 \
         --math fp16 \
         --dataset-dir ../data \
         --cuda"

if $accuracy_test; then
    log "measure Accuracy"
    options="${options} --mode accuracy"
fi

if $use_pim; then
    options="${options} --backend=NNCompiler"
    log "run evaluation with options : $options"
    ENABLE_PIM=1 python3 translate.py $options
else
    log "run evaluation with options : $options"
    ENABLE_PIM=0 python3 translate.py $options
fi

popd
