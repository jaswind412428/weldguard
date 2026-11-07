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

使用者已執行 `bash scripts/evaluate_locany_v2_v4_v5_p04.sh`，讓 V2、V4、V5 **都以 P04＋尺寸**在相同 100 張 fold0 validation 圖、919 個 GT 框推論。以下是使用者貼出的評估輸出；IoU≥0.5、一對一匹配，並非獨立 test 或 mAP。

| 權重 | 有效預測框 | 分類＋定位 P / R / F1 | 不分類定位 P / R / F1 |
| --- | ---: | --- | --- |
| V2 | 894 | 0.323 / 0.314 / 0.319 | 0.452 / 0.440 / 0.446 |
| V4 | 888 | 0.332 / 0.321 / 0.327 | **0.482** / 0.466 / **0.474** |
| V5 | 982 | 0.326 / **0.348** / **0.337** | 0.457 / **0.489** / 0.472 |

分類＋定位各類 F1：V2 pore 0.097、deposit 0.327、discontinuity 0.431、stain 0.278；V4 分別 0.068、0.202、0.492、0.303；V5 分別 0.063、0.253、**0.527**、0.300。V5 的整體分類 F1 比 V4 高 0.010，主要是 discontinuity；deposit 仍以 V2 較好，pore 三版皆弱。

V5 相對 V4 的不分類定位多 21 個 TP（449 對 428），也多 73 個 FP（533 對 460），預測框增加 94 個。因此 V5 提高召回、降低精確率，兩者 F1 差 0.002，不能宣稱 V5 的畫框品質明顯超過 V4。P04 在原模型推論時較謹慎，但以 P04 訓練的 V5 沒有呈現同樣的精確率優勢；原模型提示詞 A/B 效果不能直接預測微調後的效果。V4 與 V5 的後續共同起點應先依預定主要目標與未參與這 100 張提示詞選擇的 validation 圖確認；舊 test 不用來挑母模型。

### 2026-11-06 V5-image：one-shot 圖片問答範例（8 張初試與改版）

使用者決定接著試 V5-image。學長所說的 *few-shot* 是提供少量範例；本輪每次給 **一張有標準答案的訓練圖片**，更精確稱 *one-shot*。這與 NVIDIA 的 `visual_prompt=true` 不同：後者把目標類別的一塊影像裁圖當查詢，本輪則給另一張完整訓練圖片和它的框答案。官方程式支援 `<image-1>`、`<image-2>` 與圖片清單；但仍需先在 4090 上驗證實際 token 長度與訓練可行性。

本輪保留 P04 文字與每張目標圖的尺寸，附加一段簡短說明和「範例圖片 `<image-2>`＋該圖 GT 答案」，要求模型只回答目標 `<image-1>`。範例是**未畫框的原圖**與其 GT 座標文字，不是直接把框畫在輸入圖片上；目標圖的 GT 不放在提問中。範例只從 fold0 train 選，縮至最長邊 640 像素，正規化座標不變。validation/test 圖不作範例來源。

最初技術初試只固定兩張訓練圖，8 張 validation 的 V5 推論結果如下。兩組使用相同權重與 P04，差別只在 one-shot 區塊；IoU≥0.5，GT 共 76 框。

| 8 張初試 | 預測框 | 分類＋定位 P / R / F1 | 不分類定位 P / R / F1 |
| --- | ---: | --- | --- |
| V5＋P04 | 92 | 0.261 / 0.316 / **0.286** | 0.348 / 0.421 / **0.381** |
| V5＋P04＋固定 one-shot | 52 | **0.308** / 0.211 / 0.250 | **0.423** / 0.289 / 0.344 |

固定範例讓兩種 precision 提高、預測框減少，但 recall 與 F1 下降；其中 deposit 與 discontinuity 明顯漏框。這 8 張不足以判斷圖片範例訓練的效果。使用者因此要求正式試驗改為**每筆從全部 653 張 fold0 train 均勻偽隨機抽一張不同於目標的圖片**。程式以種子 42 與目標路徑決定抽樣，並記錄每筆配對，重跑可重現。每筆樣本都有一張範例及答案；由於 streaming packing，訓練 step 與樣本筆數不必然一對一。

改版後，使用者回報同 100 張 fold0 validation（919 個 GT 框）的 V5 權重推論初試：只用 P04 時，分類 F1 0.337、不分類定位 F1 0.472；P04 加隨機 one-shot 時，分別降為 0.160 與 0.279。兩者 precision、recall 都下降；這只說明**尚未用雙圖範例訓練的 V5** 在推論時直接加範例效果變差，不能預測 V5-image 正式訓練結果。

正式 V5-image 訓練**從與 V5 相同的原始 `nvidia/LocateAnything-3B` 權重開始**，保持 653 筆目標圖、GT、653 steps、LoRA 與其餘參數一致，差異集中在訓練時附加圖片範例；不以 V5 checkpoint 再訓 653 步後直接宣稱 one-shot 改善。訓練前對 653 筆做實際多圖 token 長度檢查，超過 2048 的樣本就中止，避免訓練器默默略過。**依使用者 2026-11-07 決定，訓練後的主要驗證讓 V5、V5-image 都只用 P04＋尺寸推論，不提供 one-shot 範例圖**，比較新增範例訓練是否改善單圖辨識；兩組使用相同 100 張 validation、生成設定與評分方式，同時報分類與不分類定位 P/R/F1。這個評估刻意讓 V5-image 的推論輸入少於訓練時的雙圖輸入，因此結果應解讀為「單圖推論遷移效果」，不代表其雙圖推論上限。反覆使用這 100 張 validation 選方案屬探索性分析，正式 test 留到最後。

對應腳本：`scripts/pilot_locany_v5_image.sh`、`scripts/train_locany_v5_image.sh`、`scripts/evaluate_locany_v5_image.sh`。上述 8 張與第一組 100 張都是**V5 權重的推論 prompt 初試**。新版評估腳本的輸出目錄為 `runs/vlm/v5_image_p04_val100_no_example`，與先前 one-shot 試驗分開保存。

### 2026-11-07 V5-image 與 V5：只用 P04 單圖推論

使用者回報 V5-image 權重的 100 張 fold0 validation 評估結果；兩個模型都使用 **P04＋圖片尺寸、沒有 one-shot 範例**，IoU≥0.5，GT 共 919 框。這表示 V5-image 模型已可推論，但尚未收到它的訓練日誌、實際 train loss 與完整 checkpoint 中繼資料。以下是 validation 的單圖推論結果，不是獨立 test 或 mAP。

| 權重 | 有效預測框 | 分類＋定位 P / R / F1 | 不分類定位 P / R / F1 |
| --- | ---: | --- | --- |
| V5（P04 訓練） | 982 | 0.326 / 0.348 / **0.337** | 0.457 / 0.489 / **0.472** |
| V5-image（P04＋圖片範例訓練） | 895 | 0.317 / 0.309 / 0.313 | 0.430 / 0.419 / 0.424 |

V5-image 相對 V5：分類 TP 320→284、FP 662→611、FN 599→635；不分類定位 TP 449→385、FP 533→510、FN 470→534。模型少輸出 87 框，但 precision 和 recall 都下降，分類 F1 降 0.024，不分類定位 F1 降 0.048。分類 F1 的最大退步來自 discontinuity（0.527→0.407）；pore 由 0.063→0.099，但 TP 都只有 6；deposit 0.253→0.247、stain 0.300→0.302，差異很小。因此目前沒有證據顯示圖片範例訓練改善 **P04 單圖推論**。V5-image 訓練時有範例圖，推論時依使用者要求不給範例圖；這項分布差異須在結論註明。正式 test 仍保留，不用這次探索性 validation 反覆調參後宣稱獨立測試成果。

### 2026-11-07 加入推論範例：四組對照

使用者再讓 V5-image 於推論時也看到一張訓練集範例原圖及其 GT 文字答案，沿用先前 V5 one-shot 初試的相同 100 張 validation、同一種子與範例配對。以下是四個已回報條件的整體結果（919 個 GT 框，IoU≥0.5）；每組皆用 P04＋目標圖片尺寸，差別在權重及有無範例。

| 權重／推論範例 | 有效預測框 | 分類＋定位 P / R / F1 | 不分類定位 P / R / F1 |
| --- | ---: | --- | --- |
| V5／無範例 | 982 | 0.326 / 0.348 / 0.337 | 0.457 / 0.489 / **0.472** |
| V5／一張範例 | 906 | 0.161 / 0.159 / 0.160 | 0.281 / 0.277 / 0.279 |
| V5-image／無範例 | 895 | 0.317 / 0.309 / 0.313 | 0.430 / 0.419 / 0.424 |
| V5-image／一張範例 | 903 | 0.341 / 0.335 / **0.338** | 0.460 / 0.452 / 0.456 |

V5-image 加入推論範例後，分類 F1 由 0.313 升至 0.338，不分類定位 F1 由 0.424 升至 0.456；相同 one-shot 輸入下也明顯好於未以範例訓練的 V5（分類 F1 0.160、定位 F1 0.279）。這顯示範例訓練改善了模型對雙圖 one-shot 輸入的處理，但**目前沒有超越不給範例的 V5 基準**：分類 F1 只差 0.001，定位 F1 仍低 0.016。`discontinuity` 在 V5-image 加推論範例後由 F1 0.407 升至 0.479，仍低於 V5 無範例的 0.527。這些數字是探索性 validation，不能當作獨立 test。

方法來源：[NVIDIA LocateAnything 訓練資料格式](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/TRAINING.md)、[NVIDIA 多模態訓練資料處理](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/train/tools.py)。

### 2026-11-07 V5-visual：四類視覺查詢與缺席類別負例（已備妥，尚未訓練）

這是與 V5-image 的「另一張完整圖片＋文字 GT 答案」不同的實驗。V5-visual 的第二張輸入是**無畫框的單一缺陷裁圖**，作為要找的視覺類別；模型只對第一張目標圖輸出該類的框。保留 P04 的四類定義、定位與 0–1000 座標規則，將第一句改為 NVIDIA 視覺查詢句型 `Detect all the objects in the image that belong to the category set: <image-2>.`，並附目標圖尺寸。這是 **P04 衍生的視覺版提示詞**，不是逐字原 P04。

每張 fold0 train 圖產生 pore、deposit、discontinuity、stain 各一筆，共 653 × 4＝2612 筆。存在該類時，答案是 `<ref>類別</ref>` 後接該類所有 GT 框；不存在時是 `<ref>類別</ref><box>None</box>`。每筆都給一張**來自另一張 fold0 train 圖**、確有該類缺陷的裁圖；同一類的正例與負例採相同參考來源規則，避免模型只靠「裁圖是否來自目標圖」猜答案。所有 reference 只由 train 製作，validation/test 均不進入訓練。資料列按目標圖及四類順序建立，但官方 trainer 可能打亂與打包，**不能宣稱四類在每四個 optimizer steps 中嚴格依序出現**。資料準備會列出各類正負例數量，避免把大量負例誤認為類別平衡。

NVIDIA 官方的 `visual_prompt=true` 會從**同一張目標圖**裁出正例當查詢，並且讓沒有框的負例維持純文字提問；因此它無法直接完成「負例也帶視覺裁圖」這個需求。本實驗使用官方支援的雙圖輸入格式和原訓練器，但由本專案預先組好跨訓練圖的裁圖與正負例，recipe 設 `visual_prompt: false` 以防二次自動裁圖。這個差異必須在報告中明示，不能稱為官方原樣的 auto-converter。來源：[NVIDIA Visual Prompt Fine-Tuning 說明](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#visual-prompt-fine-tuning)、[NVIDIA 資料轉換實作](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/train/tools.py)。

從原始 `nvidia/LocateAnything-3B` 開始，保持 V5 的 LoRA rank 8、bf16、SDPA、microbatch 1、learning rate 5e-5、linear scheduler、block size 4 與最大 2048 tokens；訓練前逐筆檢查雙圖 token 長度。預定 2612 steps 為這批 classwise 查詢各看約一次，因此**訓練步數是 V5 的四倍**，不能把未來差異只歸因於視覺提示。輸出 `runs/vlm/locany_v5_visual_epoch1`，比較輸出 `runs/vlm/v5_visual_val100`。驗證對每張固定 100 張 fold0 val 圖查四類，參考裁圖仍只取 train；同時報分類及不分類定位 IoU≥0.5 的 P/R/F1 和原始輸出。與 V5 既有單圖 P04 分數相比，推論也由每圖一次變四次，應標示查詢成本差異。正式 test 暫不使用。
