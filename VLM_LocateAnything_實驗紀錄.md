# LocateAnything 訓練實驗紀錄

## 2026-10-19：fold0 train 653 張，v1

**狀態：訓練完成；定位品質尚未評估。此輪標註框格式與官方規格不符，不能當作正式微調成果。**

| 項目 | 紀錄 |
| --- | --- |
| 基礎模型 | `nvidia/LocateAnything-3B` |
| 原始碼 | NVIDIA Eagle `Embodied`，使用者回報 commit `783f656` |
| 訓練資料 | LoHi-WELD `kfold_high/fold0/train`，653 張；val 164 張與 test 205 張未用於此輪訓練 |
| 資料設定 | `/home/leo/weldguard/datasets/derived/locany_full/full_recipe.json`，`train_full.jsonl` 653 筆；`repeat_time=1.0`、`data_augment=false` |
| 輸出 | `/home/leo/weldguard/runs/vlm/locany_fold0_train653_v1` |
| GPU | RTX 4090，單卡；峰值 VRAM 尚未紀錄 |
| 分散式與精度 | `torch.distributed.run --nproc_per_node=1`，`LAUNCHER=pytorch`，`bf16=True`，DeepSpeed ZeRO stage 2 |
| 注意力 | `--attn_implementation sdpa`；使用者另將 Eagle 的 MoonViT 注意力硬編碼從 `flash_attention_2` 本機改為 `sdpa` |
| 凍結與 LoRA | `freeze_llm=True`、`freeze_backbone=True`、`freeze_mlp=False`、`use_llm_lora=8`、`use_backbone_lora=0` |
| 訓練長度 | `max_steps=653`；日誌 `epoch=1.0`、`train_samples=streaming`。不能據此保證每張圖恰好使用一次 |
| Batch | `per_device_train_batch_size=1`、`gradient_accumulation_steps=1`；有效 batch 約 1（單卡） |
| Token 與資料載入 | `max_seq_length=2048`、`max_num_tokens_per_sample=2048`、`max_num_tokens=2048`、`packing_buffer_size=4`、`dataloader_num_workers=0` |
| 儲存與日誌 | `save_strategy=steps`、`save_steps=100`、`save_total_limit=3`、`logging_steps=10`、`report_to=none`、`do_train=True` |
| 學習率 | 指令未明確指定，使用訓練程式預設值；日誌後段從 `1.32e-5` 下降至 `2.30e-7`。需從 `training_args.bin` 或日誌確認初始值與 scheduler |
| 結果 | `train_runtime=530.4881 s`（約 8 分 50 秒）；`train_loss=1.0656933842902891`；後段每 10 steps loss 約 0.98–1.02 |
| 權重 | 最終兩個 `model-*.safetensors`，約 4.7 GB + 2.6 GB；訓練時設每 100 steps 存 checkpoint |

### 重要資料格式問題

v1 轉換程式產出的標註回答使用 `<ref>pore</ref><box>(100,200,300,400)</box>`。NVIDIA 官方要求 `<ref>pore</ref><box><100><200><300><400></box>`，座標各自是 0–1000 的特殊 token；官方物件偵測提示也以 `</c>` 分隔類別。v1 的 loss 下降只能說明模型擬合了目前訓練文字，不能證明它學會符合官方格式的缺陷定位。此問題於 v1 完成後查出，不能回溯改寫該次實驗。

下一輪應用已修正的 `scripts/prepare_locany_pilot.py`，輸出至新的資料目錄與模型目錄，不覆蓋 v1；先抽查 JSONL 的 prompt、框 token、疊圖與影像路徑，再短訓並檢查真實生成結果，之後才做正式 validation 評估。獨立 test 不用來選提示詞、checkpoint 或超參數。

資料格式依據：[NVIDIA Eagle Data Preparation Guide](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/DATA_PREPARATION.md)。

## 2026-10-19：fold0 train 653 張，v2

**狀態：訓練完成，尚未執行 validation 推論與框級評估。** 修正後資料使用官方座標 token，例如 `<ref>deposit</ref><box><43><247><123><613></box>`，提示詞以 `</c>` 分隔四類。使用者回報 JSONL 653 筆、疊框預覽 653 張、排除 0 張；四類框數與 v1 相同（358、1117、1903、2674）。

v2 沿用 v1 的模型、單卡 RTX 4090、LoRA／凍結、bf16、sdpa、batch 1、梯度累積 1、token 限額、DeepSpeed、checkpoint 與日誌參數；只改資料設定與輸出路徑：

- `--meta_path /home/leo/weldguard/datasets/derived/locany_full_v2/full_recipe.json`
- `--output_dir /home/leo/weldguard/runs/vlm/locany_fold0_train653_v2`
- `--max_steps 653`

使用者貼出的結果：`epoch=1.0`、`train_samples=streaming`、`train_runtime=659.1338 s`（10 分 59 秒）、`train_samples_per_second=0.991`、`train_steps_per_second=0.991`、`train_loss=1.6853368023088113`。最終模型已切成兩個 safetensors 檔並保存處理器／tokenizer／推論程式。結束時有 NCCL 程序群組清理警告，沒有影響已回報的訓練完成與存檔。尚未記錄峰值 VRAM、獨立 validation 指標或實際預測圖。

v1 與 v2 的標註序列格式不同，因此兩者 train loss 不適合直接比較定位品質。後續需用同一組 validation 圖、固定提示與生成設定，比較原始預訓練權重和 v2 權重的輸出框；保留原始文字、解析後框及 GT 疊圖。test 仍留待設定固定後評估。
