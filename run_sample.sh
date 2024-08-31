# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

export CUDA_VISIBLE_DEVICES=0

rm -rf log ncu && mkdir -p log ncu

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: is_causal, $8: is_hybrid, $9: prefill_fraction, $10: log_path
evaluate_fai() {
    echo "Evaluating ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8} * ${9} * ${10}"
    $WORK_PATH/output/bin/flash_attention_inference -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -is_causal=$7 -num_splits=0 -is_alibi=false -is_hybrid=$8 -prefill_fraction=${9} -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/${10}/fai_${1}_${2}_${3}_${4}_${5}_${6}_${9}.log 2>&1
    sleep 3
}

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: is_causal, $8: is_hybrid, $9: prefill_fraction, $10: log_path
ncu_fai() {
    echo "NCU ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8} * ${9} * ${10}"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/${10}/fai_${1}_${2}_${3}_${4}_${5}_${6}_${9} $WORK_PATH/output/bin/flash_attention_inference -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -is_causal=$7 -num_splits=0 -is_alibi=false -is_hybrid=$8 -prefill_fraction=${9} -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/${10}/ncu_fai_${1}_${2}_${3}_${4}_${5}_${6}_${9}.log 2>&1
    sleep 3
}

benchmark_fai_prefill_seq() {
    echo "Evaluating Prefill Seq"
    b=1
    seq=(1 8 16 32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
    hq=32
    hk=32
    d=128
    ic=true
    ih=false
    pf=0
    lp=prefill_seq

    mkdir -p log/$lp ncu/$lp

    for s in ${seq[@]};
    do
        evaluate_fai $b $s $s $hq $hk $d $ic $ih $pf $lp
        # ncu_fai $b $s $s $hq $hk $d $ic $ih $pf $lp
    done
}

benchmark_fai_prefill_batch() {
    echo "Evaluating Prefill Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    s=128
    hq=32
    hk=32
    d=128
    ic=true
    ih=false
    pf=0
    lp=prefill_batch

    mkdir -p log/$lp ncu/$lp

    for b in ${batch[@]};
    do
        evaluate_fai $b $s $s $hq $hk $d $ic $ih $pf $lp
        # ncu_fai $b $s $s $hq $hk $d $ic $ih $pf $lp
    done
}

benchmark_fai_decoding_seq() {
    echo "Evaluating Decoding Seq"
    b=1
    sq=1
    seq_k=(1 8 16 32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
    hq=32
    hk=32
    d=128
    ic=false
    ih=false
    pf=0
    lp=decoding_seq

    mkdir -p log/$lp ncu/$lp

    for sk in ${seq_k[@]};
    do
        evaluate_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
        # ncu_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
    done
}

benchmark_fai_decoding_batch() {
    echo "Evaluating Decoding Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    sq=1
    sk=128
    hq=32
    hk=32
    d=128
    ic=false
    ih=false
    pf=0
    lp=decoding_batch

    mkdir -p log/$lp ncu/$lp

    for b in ${batch[@]};
    do
        evaluate_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
        # ncu_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
    done
}

benchmark_fai_hybrid() {
    echo "Evaluating Hybrid"
    b=100
    sq=128
    sk=128
    hq=32
    hk=32
    d=128
    ic=true
    ih=true
    prefill_fraction=(0 10 20 30 40 50 60 70 80 90 100)
    lp=hybrid

    mkdir -p log/$lp ncu/$lp

    for pf in ${prefill_fraction[@]};
    do
        evaluate_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
        # ncu_fai $b $sq $sk $hq $hk $d $ic $ih $pf $lp
    done
}

benchmark_fai() {
    benchmark_fai_prefill_seq
    benchmark_fai_prefill_batch
    benchmark_fai_decoding_seq
    benchmark_fai_decoding_batch
    benchmark_fai_hybrid
}

# Prefill
nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_256_256_32_32_128_0.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_256_256_32_32_128_0 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_256_256_32_32_128_0.log 2>&1

# Decoding
# nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_1_256_32_32_128_0.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_1_256_32_32_128_0 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_1_256_32_32_128_0.log 2>&1

# GQA/MQA
# nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=64 -hk=8 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_256_256_64_8_128_0.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_256_256_64_8_128_0 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=64 -hk=8 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_256_256_64_8_128_0.log 2>&1

# Hybrid
# nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=true -prefill_fraction=50 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_256_256_32_32_128_50.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_256_256_32_32_128_50 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=false -is_hybrid=true -prefill_fraction=50 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_256_256_32_32_128_50.log 2>&1

# Alibi
# nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=true -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_256_256_32_32_128_0.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_256_256_32_32_128_0 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -num_splits=0 -is_alibi=true -is_hybrid=false -prefill_fraction=0 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_256_256_32_32_128_0.log 2>&1

# benchmark_fai
