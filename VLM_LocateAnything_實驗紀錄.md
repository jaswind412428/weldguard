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
| 學習率 | 指令未明確指定；使用者後續由 v2 日誌確認此版本預設 `learning_rate=5e-5`、`lr_scheduler_type=linear`、`warmup_steps=0`。v1 日誌後段從 `1.32e-5` 下降至 `2.30e-7`；v1 的完整訓練參數尚未獨立核對 |
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

### v2 初次 validation 預覽（尚非正式評估）

以 8 張 fold0 validation 影像試跑；初版預覽程式使用 hybrid、`temperature=0`、`top_p=1.0`、`repetition_penalty=1.0`、`max_new_tokens=512`。8 張全部解析出 0 個有效框，每張有 22–47 個格式上像框但被拒收的項目。使用者貼出的兩張原始回答顯示類別文字畸變（如 `depositstainuity`、`ststain`、`disstainuity`）、大量重複框及零面積框，因此不能把解析出的 0 框歸咎於單純的 regex 錯誤，也不能將這次預覽當作已完成的 Recall 指標。下一步以官方 worker 常用的 `temperature=0.7`、`top_p=0.9`、`repetition_penalty=1.1` 重跑相同兩張圖，保留原始輸出，判斷生成設定的影響。

以官方 worker 常用生成參數重跑兩張：原始預訓練模型在 `25_12.jpg` 輸出 1 個有效框、`25_125.jpg` 輸出 4 個有效框，均未被拒收；v2 模型兩張仍各有 0 個有效框、分別拒收 12 與 51 個，原始回答中有 `depositst`、`stst`、`disdis` 等非資料集類別及反覆／顛倒座標。原始模型有效格式不代表框的位置或類別已正確；但 v2 的格式退化並非只有貪婪解碼造成。

後續核對官方程式時發現：v2 指令沒有明確指定 `--block_size`。使用者在其 checkout 確認預設為 **4**，`causal_attn=False`；訓練日誌確認 `learning_rate=5e-5`、`lr_scheduler_type=linear`、`warmup_steps=0`。NVIDIA 官方 LocateAnything LoRA 範例明確設 `block_size=6`、rank 64、學習率 `2e-5`、warmup 500（範例總步數 5000）。v2 為 rank 8、block size 4、較高初始學習率且無 warmup。這些是候選原因，**尚未證明哪一項造成退化**。v2 保留的中途 checkpoint 為 500、600、653。下一步先查是否有樣本因 2048 token 限額被跳過，再設計小規模對照試驗，不直接再跑 653 steps。

## 2026-10-20：block size 6，100 steps 短試訓

使用與 v2 相同的 653 筆正確格式資料、基礎模型、LoRA rank 8、單卡 batch 1、sdpa、2048 token 限額及其餘訓練設定；明確加 `--block_size 6`，訓練縮短為 `--max_steps 100`。輸出：`/home/leo/weldguard/runs/vlm/locany_block6_probe100`。結果：`train_runtime=100.34 s`、`train_loss=2.1693`、日誌 `epoch=1.0`、`train_samples=streaming`。100-step loss 不與 v2 的 653-step loss 直接比較。

相同兩張 validation 圖，固定生成參數（hybrid、temperature 0.7、top-p 0.9、repetition penalty 1.1、seed 42）：

| 影像 | GT 框 | 解析成功 | 拒收 | 觀察 |
| --- | ---: | ---: | ---: | --- |
| `25_12.jpg` | 7 | 5 | 2 | 五框全為 `discontinuity`；另有 `dis`、`disain` 等畸變類別 |
| `25_125.jpg` | 4 | 0 | 4 | `dis`、`stcontinuity` 等畸變類別，故無有效框 |

此短試訓比 v2 最終權重在第一張的格式較好，但兩輪步數不同，不能單獨歸因於 block size。尚未人工確認五個有效框的幾何位置是否與 GT 對上，也沒有正式 Precision／Recall。下一個短試訓應固定 block size 6、LoRA rank 8、100 steps 等條件，只把初始學習率由 `5e-5` 改成 `2e-5`，維持 linear 與 warmup 0，以觀察格式穩定性；不要同時變更其他參數。
