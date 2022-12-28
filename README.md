# PIMmodels

This repository includes PIM acceleratable models.  
You can inference models on PIM SDK environment which is provided as **docker**.     

## PimAiCompiler supporting models

Below models are currently supported through PimAiCompiler.   
**Prerequisites** : PimAiCompiler installation provided by PIM SDK is essential.   


|No|Model|Reference|
|---|------|---|
|1|RNNT (RNN Transducer) |[link](https://github.com/mlcommons/inference)     |
|2|GNMT (Google Neural Machine Translation) |[link](https://github.com/mlcommons/training)      |
|3|HWR (Hand-Written text Recognition) |[link](https://github.com/arthurflor23/handwritten-text-recognition)     |   

Each model folder has **run.sh** bash script for execute model.  
> Basic options are identical, however model specific options might be existed.

```bash
$ ./run.sh --help

usage: ./run.sh [options] [argument]
 Option               Argument
  --accuracy           measure accuracy (default: performance)
  --clean              clean submodule repository
  --use_pim            enable PIM (default: false)
```

### Example for executing model
When you want to measure accuracy with enabled PIM, you can execute script as following:
```bash
$ cd (gnmt|rnnt|hwr ...)
$ ./run.sh --accuracy --use_pim
```
When you want to measure performance (end-to-end latency) with enabled PIM, you can execute script as following:
```bash
$ cd (gnmt|rnnt|hwr ...)
$ ./run.sh --use_pim
```

## Known issues

|Model|Issues|Status|
|------|---|------|
|GNMT|Occasionally the output of GPU-PIM is inaccuracte in comparison with the output of GPU.|In progress|

