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
| P05 | 420 | 3 | 0.019 | 0.419 | 0.192 | **0.263** | 定位 F1 最高，但分類未改善；deposit 30 個預測框，TP 0。 |

**解讀：**目前沒有一版在所有指標上最佳。P03 的定位召回最高、分類＋定位 F1 最高；P04-R 的定位 precision 最高，卻漏掉更多 GT；P05 的定位 F1 最高，但分類＋定位 F1 降至 0.019。P04 是較謹慎的平衡描述提示詞。不能只因框數下降或單一 precision 上升就說模型變好。四類分類 F1 均很低，文字提示未證明能單獨解決類別辨識。

P05 類別細項：pore TP 4／FP 212，deposit TP 0／FP 30，discontinuity TP 3／FP 97，stain TP 6／FP 68。總計分類＋定位 TP 13／FP 407／FN 906；忽略類別後定位 TP 176／FP 244／FN 743。這表示有一批框對上了缺陷位置，但類別錯誤。100 張共有 3 個拒收數值框。

P05 分數先由使用者貼出的五組評估輸出紀錄；後續已匯入 `results/locany_p05_original_prompt_diagnostics.zip`，打包程式核對 100 筆紀錄中的實際 `prompt` 均以 P05 標準文字開頭。舊腳本使用 `prompt_v6_native_size_original_100` 路徑；檔名只是歷史名稱。

逐張診斷在不分類、IoU≥0.5 的最大匹配下發現：30 個 `deposit` 預測中，10 個與 `stain` GT 配對，20 個無 GT 配對；沒有與 `deposit` GT 配對。反過來，37 個 `deposit` GT 被標成 `pore`，16 個被標成 `discontinuity`，10 個被標成 `stain`。53 個 `stain` GT 被標成 `pore`。這些交叉類別數目可能隨多框重疊時的等價配對方式略有差異，但總定位 TP=176 不變。另有 36 個預測框面積至少為整圖的 80%（`discontinuity` 24、`deposit` 7、`pore` 3、`stain` 2）。P05 同時有類別混淆、大片框和漏框問題；光靠抑制預測數量不能解決。

P02 只有 8 張初試：原始提示詞分類＋定位 F1=0；P02 F1=0.019。樣本與上表 100 張不同，**不能直接放進同一分數排名**。報告可把 P02 放在「早期嘗試與發現」欄，並標明樣本數。

## 報告與後續實驗

### P06：先定位，再考慮分類

**結果更新：**使用者回報 100 張 TP0／FP99／FN919，P/R/F1 均為 0。共有 99 個有效預測框，而非完全沒有畫框；目前未取得逐張答案，不能確定是否都是整圖框。改為單一 `weld defect` 查詢沒有改善此次定位，暫停此方向。

使用者決定暫時不要求四類判斷。P06 (`prompts/locany_p06_localization.txt`) 只詢問單一目標 `weld defect`，描述孔洞、堆積、焊道中斷與表面異常作為尋找範圍；保留每張圖的尺寸、0–1000 座標、逐個緊密框選與避免整條焊道大框的指示。原始模型、100 張 validation、生成設定仍相同。

解析器在 `--localization-only` 模式接受幾何有效框，不依輸出類別拒收；保留原始類別文字供診斷，GT 四類原始標註不改。評估只用不分類 IoU≥0.5 最大一對一匹配，不做面積過濾或刪除重複框來提高分數。超大框和重複框仍參與評分。

P03/P05 的原始回答也用相同規則重新解析，因此基準與先前「已通過類別解析後再忽略類別」的分數略有不同。已在本機診斷資料重算：P03 TP193／FP457／FN726，P0.297、R0.210、F1 0.246；P05 TP178／FP245／FN741，P0.421、R0.194、F1 0.265。這組值只用於 P06 的公平對照，不覆寫上面的歷史分數。

執行 `bash scripts/compare_locany_p06_localization_100.sh`。優先追求更高定位 precision，同時觀察 recall 是否明顯下降，並記錄大框數、空結果張數及拒收數；少畫框本身不算成功。待定位品質穩定並在其餘 validation 圖確認後，再恢復類別判斷。訓練或改資料切分仍需保留獨立評估集。

### P07：在 P05 上只補充局部框邊界要求（待測）

依使用者指示回到 P05。P07 保留 P05 原本所有文字與四類查詢，只在結尾新增兩句：沿可見局部缺陷邊界框選、細長缺陷包含完整範圍；分開不同缺陷、不要用整張圖或整條焊道代表局部缺陷。這是對已觀察到 P05 有 36 個超大框的針對性修改，不預設此改動一定有效。

提示詞：`prompts/locany_p07_local_bounds.txt`。執行：`bash scripts/compare_locany_p07_localization_100.sh`。保留四類名稱作為模型熟悉的查詢與輸出語彙，但解析與評分只要求定位；與 P03/P05 重新解析後的定位基準比較。仍使用同 100 張、同 seed、同生成設定及圖片尺寸資訊。此輪不做四類準確率優化，也不改訓練權重。

1. 先用一張表展示 P01→P03→P04→P04-R 的提示詞變化和上表指標，再展示 GT／預測疊圖，解釋 P03 的召回、P04 的謹慎及 P04-R 的漏框／格式退化。
2. P05 已完成同一批 100 張 validation 圖的框級評估及診斷。描述性的類別文字沒有改善分類，後續應先檢查資料集類別的視覺歧義、圖片範例或用相同提示詞進行小型微調，再決定是否試新的文字提示；不宜只靠更長的定義。P05 的 GT 視覺特徵是從部分圖片歸納的假設，詳見 `prompts/locany_weld_feature_notes.md`。
3. 固定提示詞後才考慮新的完整訓練；訓練資料的人類問題與評估問題須對齊。若同一 100 張 validation 圖反覆選提示詞，結論仍屬探索性；保留其餘 validation 圖做最後確認，不用 test 挑提示詞。

原始逐張紀錄仍在 4090 的 `runs/vlm/`；GitHub 診斷包為 `results/locany_v4d_original_prompt_diagnostics.zip`、`results/locany_v5_original_prompt_diagnostics.zip` 與 `results/locany_p05_original_prompt_diagnostics.zip`。前兩個 ZIP 的檔名是歷史名稱，報告應用 P 系列代號。引用上述數值時，應保留影像數、IoU 與是否要求類別一致。
### V3 尺寸資訊消融：不提示標準化座標（待測）

保留 V3 原本的四類定位提示詞於 `prompts/locany_v3_original_prompt.txt`，其內容等同 `scripts/preview_locany_val.py` 的 `PROMPT`。這次只在每張圖片後附加實際像素寬與高，不要求模型將座標標準化，也不改動模型或生成參數。執行 `bash scripts/compare_locany_v3_size_only_100.sh`，與已保存的 V3 原提示詞 100 張驗證結果比較。

解析器仍依 LocateAnything 原生的 0–1000 `<box>` token 格式解讀輸出；不要求標準化是提示詞變更，不代表將模型架構或標註轉成像素座標。如果模型改輸出像素座標，須先檢查原始回答與 `rejected_parsed_boxes`，不能將解析失敗直接解讀為偵測能力變差。這輪只用 fold0 validation，不使用 test 選提示詞。

## 2026-11-05 學長建議：P04 訓練與文字／圖片範例對照（待執行）

以下是使用者轉述的下一輪方向，**尚未完成訓練或評估**。學長認為先前選 P05 訓練的理由不夠充分，建議先用 P04 訓練一次，再決定後續擴增的起點。須保留指標差異：在原模型的同 100 張 validation 上，P04 不分類定位 precision 為 0.481、F1 為 0.234；P05 分別為 0.419、0.263；P04-R precision 為 0.520。P04 可視為偏向少誤框的候選，但**不是所有指標的最佳版本**。V4＋P05 對 V2＋P01 的分類 F1 也未改善（0.329 對 0.355）。

### 第一階段：P04 微調為 V5

1. 以 `prompts/locany_p04_balanced.txt` 加每張影像尺寸為訓練問題，重新從同一個 `nvidia/LocateAnything-3B` 原模型開始，建立 **V5**。沿用 V2／V4 的 653 筆 fold0 train、GT 答案及 653 steps，讓改動集中在提示詞。2026-11-06 使用者確認 V2 與 V4 的訓練設定相同；V5 指令逐項沿用已保存的 V4 指令。每次記錄實際參數、prompt 全文、資料版本、checkpoint 和訓練日誌。V2 部分自訂參數未完整保存在本地 metadata，這項一致性由使用者確認。
2. 在同一批 validation 圖、同一解析規則與 IoU 門檻下，比較 V2＋其訓練提示詞 P01、V4＋P05、V5＋P04；另做交叉提示詞比較，以免把推論 prompt 效果與權重效果混在一起。主要同時報告分類＋定位與不分類定位的 P/R/F1、各類別及預測框數。先明訂選擇後續起點的主要指標，不用舊 test 選模型。

### 第二階段：選定共同起點後，分成三條路線

先依 V2、V4、V5 的同條件驗證結果選擇一個共同母模型／起始 checkpoint，再進行下列對照。`V5-text`、`V5-image`、`V5-mixed` 是學長提出的**暫定實驗名稱**；即使最後選用 V2 或 V4 作母模型，也要在紀錄中寫出真正的起點，不能讓名稱暗示它們必然由 V5 接續訓練。三條路線保持相同的影像、答案、步數、batch、學習率和評估條件，僅改指定的提示資訊。

| 暫定名稱 | 相對於共同起點的改動 | 要回答的問題 |
| --- | --- | --- |
| V5-text | 在 P04 的基礎上，只增加四類焊接缺陷的視覺型態描述；不加入範例圖片。 | 更具體的文字描述能否改善定位或分類？ |
| V5-image | 不擴增 P04 的文字，改提供帶答案的圖片問答範例；先確認 LocateAnything 目前的 processor 與訓練程式支援此輸入方式。 | 圖像範例是否比新增文字描述有效？ |
| V5-mixed | 同時加入前兩組的文字描述與圖片問答範例。 | 文字與圖片範例合用是否帶來額外收益？ |

圖片範例與其答案只從**訓練切分**選取，並記錄範例 ID；不能把 validation／test 影像或標註放入訓練提示。若範例嵌入方式與純文字提示的 token／影像數不同，需記錄記憶體使用量與有效訓練樣本，避免把資源差異誤認為提示詞效果。三組完成後一起比較分數與逐張 GT 疊圖。

### 訓練前的低成本檢查

先固定同一批 validation 圖，讓尚未微調的 LocateAnything 原模型分別使用「目前 P04」和「只增加缺陷型態描述的 P04 擴增版」推論，記錄原始回答、有效／拒收框、分類與不分類 P/R/F1。這可作為篩選訊號，**不能保證微調後也有同方向改善**。使用者原話為「給 GPT 比較」；若指 ChatGPT 視覺模型，它可以協助審閱提示詞和示例，但不能以它的辨識分數替代 LocateAnything 的同模型 A/B。先確認要比較的實際模型，再執行。反覆用同 100 張 validation 選提示詞屬探索性調整；確認方案時應使用未參與選擇的保留資料，test 留到最後。

### 2026-11-06 V5 執行與訓練回報

已準備 `scripts/prepare_locany_v5_p04.py`、`scripts/train_locany_v5_p04.sh` 與 `scripts/evaluate_locany_v2_v4_v5_p04.sh`。訓練腳本沿用 V4 的單卡 LoRA 設定：原始 `nvidia/LocateAnything-3B`、653 筆 fold0 train、653 steps、microbatch 1、LoRA rank 8、learning rate 5e-5、linear scheduler、bf16、SDPA、block size 4、max sequence 2048。資料準備腳本直接重用 V2 的圖與 GT 答案，僅把 human 問題換成 P04 加每張圖的尺寸，並記錄資料雜湊；不讀取 fold0 validation 或 test 作訓練。訓練輸出路徑為 `runs/vlm/locany_v5_p04_epoch1`。

使用者回報已在 4090 完成 V5 訓練；終端的 `train metrics` 顯示 `epoch=1.0`、`train_loss=1.708`、`train_runtime=473.49 s`（約 7 分 53 秒）、`train_samples_per_second=1.379`、`train_steps_per_second=1.379`、`total_flos=20677977GF`，樣本模式為 streaming。這些數值來自使用者貼出的終端輸出；本機尚未直接核對遠端 `trainer_state.json` 與 checkpoint。單看 training loss 不能判定 V5 是否優於 V2 或 V4，且不同提示詞的 token 分布也可能影響 loss。

下一步從專案根目錄執行 `bash scripts/evaluate_locany_v2_v4_v5_p04.sh`，讓 V2、V4、V5 **都以 P04＋尺寸**在相同 100 張 validation 圖推論，列出分類與不分類定位分數。這是同提示詞的權重比較；各版搭配自己訓練提示詞的分數可另列，但不能單憑它歸因於權重差異。V5 的 validation 結果尚未取得。
