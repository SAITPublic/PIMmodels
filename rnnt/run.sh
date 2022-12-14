#!/bin/bash

working_dir=`pwd`
accuracy_test=false
use_pim=false
repo_name="inference"
repo_url="https://github.com/mlcommons/inference.git"

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
    if [ ! -d $repo_nane ]; then
        err "failed to clone git repository ${repo_name}"
        exit 0
    fi
    pushd "$working_dir/$repo_name"
    git checkout r0.7
    git submodule update --init --recursive
    popd
}

function patch_for_pim()
{
    # Patch to enable PIM operator
    log "patch to enable PIM operator on ${repo_name} model"
    pushd "$working_dir/$repo_name"
    if [ ! -f "$working_dir/$repo_name/speech_recognition/rnnt/run_simple.sh" ]; then
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
    popd
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
        --patch)
            patch_for_pim
            exit 0
            ;;
        *)
            usage
	    ;;
    esac
    shift
done

if [ ! -d "$repo_name/speech_recognition" ]; then
    clone_repo $repo_url
    patch_for_pim

    # Installation of prerequisites
    log "Install python3 requirements"
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org absl-py numpy unidecode inflect tqdm toml librosa pandas

    if [ ! -x "sox" ]; then
        sudo apt install sox
    fi

    # Installation MLperf Inference
    log "Install MLperf inference"
    pushd "$working_dir/$repo_name/loadgen"
    CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel
    pip3 install dist/*.whl
    popd
fi

# Preparing dataset and model
if [ ! -f "model/rnnt.pt" ]; then
    log "Download model weight"
    mkdir -p model
    wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 --no-check-certificate -O ./model/rnnt.pt
fi

pushd "$working_dir/$repo_name/speech_recognition/rnnt"
log "Prepare dataset"
if [ ! -d "./local_data/LibriSpeech" ]; then
    log "Download dataset"
    mkdir -p local_data/LibriSpeech
    python3 pytorch/utils/download_librispeech.py pytorch/utils/librispeech-inference.csv local_data/LibriSpeech -e local_data
fi

if [ ! -f "./local_data/dev-clean-wav.json" ]; then
    python3 pytorch/utils/convert_librispeech.py --input_dir local_data/LibriSpeech/dev-clean --dest_dir local_data/dev-clean-wav --output_json local_data/dev-clean-wav.json
fi

if [ ! -f "rnnt.pt" ]; then
    ln -s $working_dir/model/rnnt.pt rnnt.pt
fi
popd

# Run model
pushd "$working_dir/$repo_name/speech_recognition/rnnt"
work_dir=`pwd`
options="--dataset_dir ${work_dir}/local_data \
         --manifest ${work_dir}/local_data/dev-clean-wav.json \
         --pytorch_config_toml pytorch/configs/rnnt.toml \
         --pytorch_checkpoint ${work_dir}/rnnt.pt \
         --scenario SingleStream \
         --log_dir ${work_dir}/run_logs/test/fp16/SingleStream"

if $use_pim; then
    log "enable PIM"
    options="${options} --backend=NNCompiler"
fi

if $accuracy_test; then
    log "measure Accuracy"
    options="${options} --accuracy"
fi

log "run evaluation with options : $options"
python3 run.py $options
popd
