# Evaluation

## Run

For example, to run the baseline experiment for DeepSeek-V2-Lite-Chat, run the following command:
```bash
sh benchmark_deepseek_v2_lite_baseline.sh $MODEL_PATH $DATASET_PATH
```

## Show Result

To show the result, run the following command:
```bash
python show_result.py --path ./result_v2_semi_pd
```

You can copy the result in stdout to draw the figure.
