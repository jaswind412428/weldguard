# V7、V7-G、V7-N：先訓練三個權重

本輪先訓練，**尚不執行推論或查看 test 分數**。三個權重都從原始 `nvidia/LocateAnything-3B` 獨立開始，沿用 P04、`x1` 再 `y1` 的 GT 排序、LoRA rank 8、bf16、SDPA、microbatch 1、學習率 `5e-5` 和 seed 42。

| 權重 | 訓練查詢 | 訓練時範例 | 步數 | validation loss 間隔 |
| --- | --- | --- | ---: | ---: |
| V7 | 每張圖一次問四類 | 無 | 653 | 100 steps |
| V7-G | 每張圖一次問四類 | 另一張 train 圖，直接畫出 GT 框及類名 | 653 | 100 steps |
| V7-N | 每張圖四次逐類查詢 | NVIDIA `visual_prompt=true`：正例從該張訓練圖裁一個無框缺陷，負例維持文字 | 2612 | 400 steps |

V7 和 V7-G 都以 fold0 train 653 張訓練，fold0 val 164 張監控損失。V7-N 是相同 653 張，每張四類各問一次。三組各自以同格式的 val 資料計算 `eval_loss`，保存每次 `loss` 和 `eval_loss`，訓練結束載入 validation loss 最小的 checkpoint；不以 test 選權重。V7-N 的輸入形式和查詢數與前兩組不同，因此三組的 loss 數值本身不能直接比較好壞。訓練完成後才在原 205 張獨立 test 上跑九組推論，分類加定位的正確框須類別相同且 IoU ≥ 0.5。

九組推論矩陣是 **三個權重 × 三種推論輸入**：單張原圖、加已畫 GT 的另一張 train 參考圖、加無框的 train 缺陷裁圖。三種推論方式要用相同 test 影像與固定參考圖來源；GT 參考圖絕不能是目標 test 圖自己的標註。visual prompt 推論為逐類查詢，類別已由查詢給定，報告時須與一次問四類的自主分類分開解讀。這九組的推論程式與評估會在三個權重訓練完成後製作。

## 4090 上的執行順序

在 `~/weldguard` 且已更新到包含本批腳本的 commit 後，執行以下三行。腳本會啟用 `.venv-vlm`；不用另外啟用 conda。先跑完一個，再跑下一個，避免同時佔用 4090 VRAM。

```bash
cd ~/weldguard
bash scripts/train_locany_v7_family.sh v7
bash scripts/train_locany_v7_family.sh v7-g
bash scripts/train_locany_v7_family.sh v7-n
```

每個腳本先產生 `datasets/derived/locany_v7/<版本>/` 的資料及 manifest，檢查訓練／validation 樣本的實際 token 長度與 NVIDIA visual prompt 轉換，再從原始模型訓練。若任一筆超過 2048 tokens，會在訓練前停止，避免靜默丟資料。NVIDIA 原訓練器把 `eval_dataset` 固定設為 `None`；`make_locany_v7_eval_trainer.py` 會建立**另一份 V7 專用副本**，加入有限長度的 validation loss 評估，原檔不覆寫。它會核對關鍵程式片段；如果 4090 上的 Eagle 版本與預期不同，會停止並顯示錯誤，不盲目修改。

V7-N 的資料每次執行都會重新產生，避免使用到舊版單類查詢。NVIDIA 的轉換器只辨認 `Detect all the objects in the image that belong to the category set: <類別>.` 句型；預檢會逐筆確認有標註的正例變成兩張圖、無該類的負例維持一張圖。若兩者數量不符，就會在訓練前停止。

完成後看三個資料夾：

```text
runs/vlm/locany_v7_p04_xy/
runs/vlm/locany_v7_g_p04_xy/
runs/vlm/locany_v7_n_p04_xy/
```

各有 `train.log`、`trainer_state.json`、`loss_curve.csv`、`loss_summary.json`、`training_metadata.json` 和模型權重。`loss_summary.json` 有最佳 validation loss 與 checkpoint；若訓練 loss 下降而 validation loss 上升，後段可能過擬合。validation loss 只能協助選 checkpoint，不能代替正式 test 的 P/R/F1。

資料切分提醒：老師曾建議日後把原 test 併入訓練，且訓練不限 fold0。**這輪為了與 V6 控制變因，暫時保留 fold0 train／val／test**。未來若把原 test 納入訓練，必須另建獨立 test，不能再用這 205 張作正式測試。

依據：[NVIDIA LocateAnything 訓練指南](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/TRAINING.md)、[visual prompt 與 LoRA 說明](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/lam/README.md)、[NVIDIA 訓練程式](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/train/locany_finetune_magi_stream.py)。
