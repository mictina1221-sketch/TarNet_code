## Generate

Due to GitHub’s file size limits, we are unable to upload the complete `hidden_states` files. To reproduce our results, please run `Creat_text.py` on `6th_to_10th.csv` to generate the complete set of hidden states.

**Note:** `Creat_text.py` was originally designed to run on Google Colab.

### Our example command

The following example generates hidden states for the specified prompt. Please adjust the file paths and prompt file as needed.

```python
import os

os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache"

!python /content/drive/MyDrive/gpi_replication/create_data/create_text.py \
    --prompt_file "/content/drive/MyDrive/gpi_replication/生成prompt/農民to公務人員_prompt" \
    --checkpoint "meta-llama/Llama-3.1-8B-Instruct" \
    --task_type "create" \
    --cache_dir "/content/hf_cache" \
    --save_hidden "/content/drive/MyDrive/gpi_replication/data/hidden_states_農民to公務人員" \
    --save_name "/content/drive/MyDrive/gpi_replication/data/generated_texts_農民to公務人員" \
    --max_new_tokens 500
```
