!/usr/bin/env bash

# Prepare submission


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
echo "exporting OPENAI_API_KEY"

export OPENAI_API_KEY="your_openai_api_key_here"

sleep 1

echo
echo "running program >>>>>>>>>>>>>>>>>>>"
echo



python gov_report.py --compress_rate 0.2 --start_point 0 --end_point 2 --evaluation_model gpt


echo
echo "program finished <<<<<<<<<<<<<<<<<<<<"
echo

