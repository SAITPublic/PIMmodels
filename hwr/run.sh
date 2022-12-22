#!/bin/bash

working_dir=`pwd`
accuracy_test=false
use_pim=false
repo_name="handwritten-text-recognition"
repo_url="https://github.com/arthurflor23/handwritten-text-recognition.git"

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
    git checkout ad485c09 -b hwr-pim
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
        *)
            usage
	    ;;
    esac
    shift
done

if [ ! -d "$repo_name/src" ]; then
    clone_repo $repo_url

    # Patch to enable PIM operator
    log "patch to enable PIM operator on HWR model"
    pushd "$working_dir/$repo_name"
    if [ ! -f "$working_dir/$repo_name/src/run_eval.sh" ]; then
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

    # Installation of prerequisites
    LOG "Install python3 requirements"
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org package/torchvision-0.7.0a0+78ed10c-cp36-cp36m-linux_x86_64.whl
    sudo pip3 uninstall tensorflow tensorboard tensorboard-data-server tensorboard-plugin-wit tensorflow-estimator
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org tensorflow
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org pytorch_lightning==1.5.10
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org sklearn
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn
    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r handwritten-text-recognition/requirements.txt
fi

# Preparing dataset and model

pushd "$working_dir/handwritten-text-recognition/src"
if [ ! -f "iam.hdf5" ] || [ ! -f "last.ckpt" ]; then
    log "decompress dataset and model"
    cat $working_dir/data/xa* > dataset.tar
    tar xvf dataset.tar
    rm dataset.tar
fi
popd

# Run model
pushd "$working_dir/handwritten-text-recognition/src"
options="--arch=puigcerver --source=iam --test --batch_size=1 --epochs=100 --gpus 1 --resume_from --checkpoint=last.ckpt --precision=FP16"

if $accuracy_test; then
    log "measure Accuracy"
    options="${options} --accuracy"
fi

if $use_pim; then
    log "enable PIM"
    options="${options} --backend=NNCompiler"
    log "run evaluation with options : $options"
    ENABLE_PIM=1 python3 pytorch_test.py $options
else
    options="${options} --backend=pytorch"
    log "run evaluation with options : $options"
    ENABLE_PIM=0 python3 pytorch_test.py $options
fi
popd
