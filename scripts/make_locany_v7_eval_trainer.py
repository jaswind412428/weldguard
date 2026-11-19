"""Create a V7-only copy of NVIDIA's trainer with finite validation-loss evaluation.

The upstream stream trainer hardcodes eval_dataset=None. This patch never changes
the original trainer. It aborts if the expected source structure has changed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from textwrap import indent


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "eagle-vlm/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
TARGET = REPO / "eagle-vlm/Embodied/eaglevl/train/locany_v7_eval_stream.py"
CONTINUATION_TARGET = TARGET.with_name("locany_v8_n2_whole_eval_stream.py")


EVAL_CLASS = '''class FiniteValidationMTP(Dataset):
    """Exactly one pass over the validation JSONL, one sample per batch."""

    def __init__(self, dataset, max_tokens):
        self.dataset = dataset
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # NVIDIA's dataset seeds the process-global Python/NumPy RNG. Restore
        # it so validation does not perturb the following training steps.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            sample = self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        length = int(sample["input_ids"].shape[0])
        if length > self.max_tokens:
            raise ValueError(f"Validation sample {index} has {length} tokens, limit={self.max_tokens}")
        sample["sub_sample_lengths"] = torch.tensor([length], dtype=torch.int32)
        return sample


'''

EVAL_DATALOADER = '''    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("V7 validation dataset is missing")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=PackedCollatorMTP(dataset=None), pin_memory=False)
        return self.accelerator.prepare(loader)

'''

EVAL_BUILD = '''    # V7 finite validation set: kept separate from the infinite training stream.
    validation_recipe_path = os.environ.get("WELDGUARD_V7_VAL_RECIPE")
    if not validation_recipe_path:
        raise ValueError("Set WELDGUARD_V7_VAL_RECIPE to the held-out validation recipe")
    with open(validation_recipe_path, encoding="utf-8") as validation_file:
        validation_recipe = json.load(validation_file)
    if len(validation_recipe) != 1:
        raise ValueError("V7 expects one validation dataset")
    validation_name, validation_meta = next(iter(validation_recipe.items()))
    validation_base = LazySupervisedDatasetMTP(
        validation_name, validation_meta, processor,
        block_size=model_args.block_size, repeat_time=1.0,
        target_fps=data_args.target_fps, max_frames=data_args.max_frames,
        video_total_pixels=data_args.video_total_pixels,
    )
    if len(validation_base) == 0:
        raise ValueError("V7 validation dataset is empty")
    validation_dataset = FiniteValidationMTP(validation_base, data_args.max_num_tokens_per_sample)
    logger.info(f"V7 validation samples: {len(validation_dataset)}")

'''


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one trainer anchor, found {count}: {old[:90]!r}")
    return source.replace(old, new, 1)


def use_existing_llm_adapter(source: str, model_version: str = "V8-N2") -> str:
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    loads, wraps = [], []
    for node in ast.walk(main):
        match node:
            case ast.Assign(
                targets=[ast.Name(id="model")],
                value=ast.Call(func=ast.Attribute(
                    value=ast.Name(id="LocateAnythingForConditionalGeneration"), attr="from_pretrained"
                )),
            ):
                loads.append(node)
            case ast.Expr(value=ast.Call(func=ast.Attribute(
                value=ast.Name(id="model"), attr="wrap_llm_lora"
            ))):
                wraps.append(node)
    if len(loads) != 1 or len(wraps) != 1:
        raise ValueError(f"Expected one LocateAnything load and one LLM LoRA setup, got {len(loads)} and {len(wraps)}")

    load, wrap = loads[0], wraps[0]
    call = load.value
    arguments = [*call.args, *(kw for kw in call.keywords if kw.arg != "output_loading_info")]
    argument_text = "".join(f"    {ast.unparse(arg)},\n" for arg in arguments)
    checked_load = f"""model, loading_info = {ast.unparse(call.func)}(
{argument_text}    output_loading_info=True,
)
adapter_errors = [
    key for field in ("missing_keys", "unexpected_keys") for key in loading_info.get(field, [])
    if ".lora_" in key
]
if adapter_errors or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
    raise ValueError(f"Could not restore {model_version} checkpoint weights: {{loading_info}}")
"""
    restore = f"""adapter = getattr(model.language_model, "peft_config", {{}}).get("default")
if adapter is None or adapter.r != model_args.use_llm_lora:
    raise ValueError("{model_version}'s saved LLM adapter was not restored with the expected rank")
model.language_model.set_adapter("default")
if not any(".lora_" in name and param.requires_grad for name, param in model.language_model.named_parameters()):
    raise ValueError("{model_version}'s restored LLM adapter is not trainable")
logger.info("Continuing {model_version} from the restored LLM adapter with a fresh optimizer")
"""
    lines = source.splitlines(keepends=True)
    for statement, replacement in sorted(((load, checked_load), (wrap, restore)),
                                         key=lambda edit: edit[0].lineno, reverse=True):
        start = statement.lineno - 1
        padding = lines[start][:statement.col_offset]
        if padding.strip():
            raise ValueError("Expected the trainer statement to start on its own line")
        lines[start:statement.end_lineno] = [indent(replacement, padding)]
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-hash", action="store_true")
    parser.add_argument("--continue-llm-lora", nargs="?", const="v8-n2", choices=("v8-n2", "v7-g"))
    args = parser.parse_args()
    original = SOURCE.read_text(encoding="utf-8")
    modified = original
    modified = replace_once(modified, "class StreamPackingMTPTrainer(Trainer):", EVAL_CLASS + "class StreamPackingMTPTrainer(Trainer):")
    modified = replace_once(modified, "    def get_train_dataloader(self):", EVAL_DATALOADER + "    def get_train_dataloader(self):")
    modified = replace_once(modified,
        '    logger.info(f"Dataset built in {time.time() - t_start:.2f}s")\n',
        '    logger.info(f"Dataset built in {time.time() - t_start:.2f}s")\n\n' + EVAL_BUILD)
    modified = replace_once(modified, "        eval_dataset=None,", "        eval_dataset=validation_dataset,")
    target = TARGET
    if args.continue_llm_lora == "v8-n2":
        target = CONTINUATION_TARGET
        modified = use_existing_llm_adapter(modified)
    elif args.continue_llm_lora == "v7-g":
        target = TARGET.with_name("locany_v7_g_pore_eval_stream.py")
        modified = use_existing_llm_adapter(modified, "V7-G")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    if target.exists() and target.read_text(encoding="utf-8") != modified:
        raise FileExistsError(f"A different trainer already exists: {target}")
    target.write_text(modified, encoding="utf-8")
    print(f"Validation trainer: {target}")
    print(f"Upstream SHA256: {digest}")
    if args.show_hash:
        print(f"Validation trainer SHA256: {hashlib.sha256(modified.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
