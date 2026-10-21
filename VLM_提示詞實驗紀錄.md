# LocateAnything 提示詞實驗紀錄（P 系列）

**命名規則：** `P` 只代表推論提示詞；`V` 只代表訓練後模型版本。以下分數均使用原始 `nvidia/LocateAnything-3B` 權重，不能寫成新模型的訓練成果。歷史檔名若含 `v4d`、`v5` 等字樣，僅是舊提示詞檔名，報告一律改用此表的 P 代號。

## 提示詞對照

| ID | 改動與固定條件 | 標準提示詞檔／實作 | 驗證狀態 |
| --- | --- | --- | --- |
| P01 | 原始官方式四類定位問題；不附尺寸。 | `scripts/preview_locany_val.py` 的 `PROMPT` | 100 張 |
| P02 | P01 改成較長的角色、任務、框格式與限制說明；不附尺寸。這一版主要詳述任務，沒有可靠的逐類影像特徵。 | `prompts/locany_p02_detailed.txt` | 8 張初試；沒有同條件 100 張結果 |
| P03 | P01 加入當張影像的像素寬、高，並說明輸出仍用 0–1000 正規化座標。 | P01 + `--include-image-size` | 100 張 |
| P04 | 平衡描述四類焊接缺陷與輸出限制，並加入影像尺寸。 | `prompts/locany_p04_balanced.txt` + `--include-image-size` | 100 張 |
| P04-R | P04 之後試過的更簡短、較嚴格的限制變體，保留為消融紀錄；加入尺寸。**不是 P05。** | `prompts/locany_p04r_compact.txt` + `--include-image-size` | 100 張 |
| P05 | 依 GT 疊框重新描述類別：避免把暗點一律叫 pore，提醒圓斑也可能是 deposit，並區分較長的焊道中斷與不規則表面斑塊；加入尺寸。 | `prompts/locany_p05_gt_features.txt` + `--include-image-size` | 100 張 |

舊檔對應：P02=`locany_weld_detailed_v1.txt`；P04=`locany_weld_balanced_v4d.txt`；P04-R=`locany_weld_compact_v5.txt`；P05=`locany_weld_native_v6.txt`。以上舊檔保留，避免破壞既有腳本及結果路徑。P03 的尺寸文字由推論程式依每張圖片動態附加，不是一個固定寬高的文字檔。

## 同條件 100 張 validation 結果

固定 fold0 validation 的同 100 張影像、共 919 個 GT 框；生成模式 `slow`、temperature 0.7、top-p 0.9、repetition penalty 1.1、最多 512 新 token、seed 42。IoU ≥ 0.5、一對一匹配。這是固定門檻的框級指標，**不是 mAP，也不是獨立 test**。

| Prompt | 有效預測框 | 拒收數值框 | 分類＋定位 F1 | 不分類定位 Precision | 不分類定位 Recall | 不分類定位 F1 | 觀察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P01 | 602 | 0 | 0.030 | 0.234 | 0.153 | 0.185 | 原始基準。 |
| P03 | 650 | 0 | **0.032** | 0.297 | **0.210** | **0.246** | 尺寸資訊改善定位召回與 F1，但類別仍常錯。 |
| P04 | 295 | 13 | 0.028 | 0.481 | 0.155 | 0.234 | 框較謹慎；deposit 只輸出 1 個有效框。 |
| P04-R | 223 | 70 | 0.019 | **0.520** | 0.126 | 0.203 | 定位 precision 最高但漏框與格式問題增加；deposit 0 框。 |
| P05 | 420 | 待補 | 0.019 | 0.419 | 0.192 | **0.263** | 定位 F1 最高，但分類未改善；deposit 30 個預測框，TP 0。 |

**解讀：**目前沒有一版在所有指標上最佳。P03 的定位召回最高、分類＋定位 F1 最高；P04-R 的定位 precision 最高，卻漏掉更多 GT；P05 的定位 F1 最高，但分類＋定位 F1 降至 0.019。P04 是較謹慎的平衡描述提示詞。不能只因框數下降或單一 precision 上升就說模型變好。四類分類 F1 均很低，文字提示未證明能單獨解決類別辨識。

P05 類別細項：pore TP 4／FP 212，deposit TP 0／FP 30，discontinuity TP 3／FP 97，stain TP 6／FP 68。總計分類＋定位 TP 13／FP 407／FN 906；忽略類別後定位 TP 176／FP 244／FN 743。這表示有一批框對上了缺陷位置，但類別錯誤。P05 的拒收數值框數尚未收到，不能先填 0。

P02 只有 8 張初試：原始提示詞分類＋定位 F1=0；P02 F1=0.019。樣本與上表 100 張不同，**不能直接放進同一分數排名**。報告可把 P02 放在「早期嘗試與發現」欄，並標明樣本數。

## 報告與後續實驗

1. 先用一張表展示 P01→P03→P04→P04-R 的提示詞變化和上表指標，再展示 GT／預測疊圖，解釋 P03 的召回、P04 的謹慎及 P04-R 的漏框／格式退化。
2. P05 已完成同一批 100 張 validation 圖的框級評估。下一步核對原始回答與疊框圖，尤其是 deposit 的 30 個 FP：需分辨它們是落在其他類 GT 上，還是落在無標註處。P05 的 GT 視覺特徵是從部分圖片歸納的假設，詳見 `prompts/locany_weld_feature_notes.md`。
3. 固定提示詞後才考慮新的完整訓練；訓練資料的人類問題與評估問題須對齊。若同一 100 張 validation 圖反覆選提示詞，結論仍屬探索性；保留其餘 validation 圖做最後確認，不用 test 挑提示詞。

原始逐張紀錄仍在 4090 的 `runs/vlm/`；GitHub 診斷包為 `results/locany_v4d_original_prompt_diagnostics.zip` 與 `results/locany_v5_original_prompt_diagnostics.zip`。報告引用上述數值時，應保留影像數、IoU 與是否要求類別一致。
