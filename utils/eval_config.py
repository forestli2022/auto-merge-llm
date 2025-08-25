from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

# NOTE: 'path' will be overridden from CLI via --hf-path, so this placeholder is fine.
models = [
    dict(
        type=HuggingFaceCausalLM,
        path="/PLACEHOLDER_MODEL_PATH",   # overridden via CLI
        # tokenizer_path can be omitted if tokenizer files are in the same folder
        model_kwargs=dict(trust_remote_code=True, low_cpu_mem_usage=True),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        abbr='pruned-llm',
        max_seq_len=2048,
        max_out_len=128,
        batch_size=1,
        batch_padding=False,
        run_cfg=dict(num_gpus=1),
    )
]

with read_base():
    # Your exact dataset imports
    from opencompass.configs.datasets.CLUE_cmnli.CLUE_cmnli_ppl import cmnli_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from opencompass.configs.datasets.piqa.piqa_ppl import piqa_datasets
    from opencompass.configs.datasets.FewCLUE_chid.FewCLUE_chid_gen import chid_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl import WSC_datasets as WSC_ppl_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen import WSC_datasets as WSC_gen_datasets
    from opencompass.configs.datasets.commonsenseqa.commonsenseqa_ppl import commonsenseqa_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
    from opencompass.configs.datasets.race.race_ppl import race_datasets
    from opencompass.configs.datasets.Xsum.Xsum_gen import Xsum_datasets
    from opencompass.configs.datasets.CLUE_C3.CLUE_C3_ppl import C3_datasets

# Combine: full GPTailor suite (CHID/XSum gen; WSC ppl+gen; others ppl/acc)
datasets = []
datasets += cmnli_datasets
datasets += hellaswag_datasets
datasets += piqa_datasets
datasets += chid_datasets
datasets += WSC_ppl_datasets
datasets += WSC_gen_datasets
datasets += commonsenseqa_datasets
datasets += BoolQ_datasets
datasets += mmlu_datasets
datasets += cmmlu_datasets
datasets += race_datasets
datasets += Xsum_datasets
datasets += C3_datasets

# Optional pretty summary/CSV
summarizer = dict(type='NaiveSummarizer', summary_types=['performance', 'csv'])
