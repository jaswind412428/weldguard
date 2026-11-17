# 參考圖改善實驗：P11、錯誤拆解與正負例對齊

日期：2026-11-17。目的：先找出 FP／FN 的來源，再確認參考圖片是否有效，最後才用對齊後的資料訓練。以下為實作與待執行實驗，沒有新模型分數。

**完整 val164 更新：**七組封包已核對，P11 與新參考庫未取得整體收益；後續預設訓練改為 **P10＋原參考庫＋正負例都帶外部訓練裁圖**。P11／context 的原訓練候選仍可明確指定。完整數據見 `VLM_參考圖改善_val164分析.md`；這個候選尚未訓練完成，不能寫成改善成果。

## 查到的方法與本次採用範圍

| 來源 | 核對到的內容 | 本專案採用範圍 |
|---|---|---|
| [NVIDIA LocateAnything 官方說明](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/lam/README.md) | 現行公開 LocateAnything-3B 不直接支援 visual prompt 推論；提供自行微調的路徑 | 更正先前「原模型加裁圖應該能用」的預期。自訂微調的 V8-N 是否學會參考圖，須靠實驗證明 |
| [NVIDIA 資料準備](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/DATA_PREPARATION.md)、[轉換程式](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/train/tools.py) | 使用 0–1000 的 x1,y1,x2,y2；visual_prompt 轉換從來源圖的正例取裁圖，沒有該類的答案保留文字查詢 | 保留原生座標／答案格式；新增「正負例都用另一張 train 裁圖」的擴充資料，不宣稱是官方原封不動的 recipe |
| [TIDE，ECCV 2020](https://arxiv.org/abs/2008.08115) | 用錯類、定位、兩者皆錯、重複、背景、未偵測等原因診斷偵測器 | 採用錯誤分類思路。由於模型沒有可比較的 confidence，本實作不是官方 TIDE、不是 ΔAP；仍用既有分類 IoU 0.5 的 P/R/F1 |
| [T-Rex2，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4804_ECCV_2024_paper.php) | 文字與視覺提示有互補性，論文以對比學習對齊兩者 | 借用「類別文字要與圖片語意對齊」的思路，設計 P11；沒有實作它的模型架構或 contrastive loss，不能說重現 T-Rex2 |
| [SAHI，ICIP 2022](https://arxiv.org/abs/2202.06934) | 以切片推論與微調處理小物件 | 列為 pore 的後續候選。本輪尚未加入目標圖切片，避免與參考圖實驗混在一起；實作前需處理框座標映回、邊界與重複框 |

以上查閱的是 2026-11-17 的上游文件，不能假定和 4090 已安裝的 Eagle revision 完全一致。本輪沿用現有可運作的 SDPA 訓練程式，不更新 torch／transformers，也不要求重建環境。

## 已完成的改動

1. 新增 `scripts/analyze_locany_reference_errors.py`：直接讀舊預測，列出四類及整體六項指標、FP／FN 原因、錯類候選、每次查詢越類輸出，以及圖片輸入摘要。可以先分析既有九組，不需要新推論。
2. 新增 P11：明說當次參考裁圖是何種類別，只找該類；沒有時回答原生 None；保留尺寸、正規化及兩角座標格式。P10 與舊九組保持為歷史實驗。
3. 新增逐查詢完整記錄：原始答案、解析框、參考來源、兩圖尺寸／像素雜湊、處理後 tensor 形狀／雜湊、image grid、輸入 token 數、重新編碼的輸出長度與 EOS。不同雜湊只能證明輸入有變，不證明模型理解了圖片；同文字的 P10 四次查詢才適合用來初看參考敏感度。
4. 新增參考庫：每個 train GT 都可成為候選，裁圖加 20% 局部脈絡（每側至少 2 px），保留原始解析度；此比例是待驗證的預設值。已標框 GT 參考圖提高到長邊最多 1280，類別圖例放在新增的下方空白列，避免黑底文字蓋住缺陷。
5. 新增兩圖正負例資料：653 張 train × 四類 = 2,612 筆；164 張 val × 四類 = 656 筆。外部參考只取 train，排除目標圖本身；目標無該類時仍提供同形式參考圖。沒有讀取 test 來建立資料。

## 第一階段：固定 20 張 validation 的七個對照

20 張由 seed=42 對 164 張檔名做固定亂序取前 20 張，不依 GT／模型分數選圖。各組用同一批圖，完整順序存於資料 manifest。

| 組別 | 權重 | 查詢／參考 | 比較目的 |
|---|---|---|---|
| A | V8 | P10、none | 本輪 validation 基準 |
| B | V8-G | P10、原 GT 參考圖 | GT 基準 |
| C | V8-G | P10、改良 GT 參考圖 | 只改參考圖呈現，來源圖不變 |
| D | V8-N | P10、原緊裁參考庫 | visual 基準；同文字不同裁圖的輸入檢查 |
| E | V8-N | P11、原緊裁參考庫 | 只改提示詞，參考來源與 D 相同 |
| F | V8-N | P11、加脈絡的參考庫 | 比較參考策略；因候選庫也擴大，不能只歸因於 padding |
| G | V8-N | P10、加脈絡的參考庫 | 補齊 P10／P11 × 原參考庫／新參考庫，確認改善是否需要 P11 |

輸出四種分開標示的評分：`raw`、`dedup`、`query_filter`、`query_filter_dedup`。後兩者只適用有單類查詢記錄的組別。去重僅刪同類 IoU≥0.9 的後出框，使用原生成順序，不用 GT 排序，也不捏造 confidence。它可能刪掉正確框，必須一起看 TP／FN／R／F1。

不將輸出類別強制改為請求類別。25_155.jpg 的 stain 查詢輸出正確 deposit 就是不能直接覆寫／過濾的例子。

```bash
cd ~/weldguard
git pull --ff-only origin master &&
bash scripts/run_locany_reference_pilot.sh
```

腳本會啟用 `.venv-vlm`、先分析舊九組已存在的預測、建立參考庫、跑七組 val20 並打包。這是實際 GPU 推論，需要在 4090 執行。若任一步失敗，會停止並保留已完成的逐圖結果；同設定重跑可接續。原六組已完成時會跳過，僅新增 G 組推論並重新分析／打包。

```bash
git add -f results/reference_alignment_val20_report.zip &&
git commit -m "Add reference alignment validation pilot" &&
git push origin HEAD
```

20 張只供排錯。要確認結果能否維持，可用相同設定擴成完整 164 張：

```bash
LIMIT=164 bash scripts/run_locany_reference_pilot.sh
```

若要把相同七個條件套到三個權重，共 21 組（成本較高）：

```bash
ALL_WEIGHTS=1 bash scripts/run_locany_reference_pilot.sh
```

### 2026-11-17：使用者回報的首輪 val20 結果

以下均為分類 raw 評分；20 張共 169 個 GT。尚未取得封包，還不能核對每類細節、去重結果、輸入 trace 或實際畫框。

| 條件 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| V8 / none / P10 / original | 54 | 100 | 115 | .351 | .320 | .334 |
| V8-G / gt / P10 / original | 55 | 100 | 114 | .355 | .325 | .340 |
| V8-G / gt / P10 / context | 60 | 109 | 109 | .355 | .355 | .355 |
| V8-N / visual / P10 / original | 65 | 168 | 104 | .279 | .385 | .323 |
| V8-N / visual / P11 / original | 52 | 175 | 117 | .229 | .308 | .263 |
| V8-N / visual / P11 / context | 62 | 144 | 107 | .301 | .367 | .331 |

- 同參考庫直接把目前 P10 訓練權重的推論換成 P11：少 13 TP、多 7 FP、多 13 FN，尚未看到收益。這不能證明以 P11 再訓練一定無效。
- 同為 P11，換新參考庫後：多 10 TP、少 31 FP、少 10 FN。顯示參考策略值得追查，不能單獨歸因於 padding，因候選庫與選取結果也不同。
- P11＋新參考庫對 P10＋舊參考庫：少 3 TP、少 24 FP、多 3 FN，F1 僅多約 .008。屬於誤報／召回的交換，20 張不足以確認穩定優勢。
- V8-G 換新 GT 呈現：多 5 TP、多 9 FP、少 5 FN；P 近似不變、R 上升，F1 多約 .015。
- 因此新增 G 組 P10＋新參考庫，補足交叉比較。先看結果與逐類錯誤，再決定後續提示詞和完整訓練；不據此首輪結果啟動新的大規模訓練。

## 第二階段：正負例對齊後的訓練

完整 val164 已完成。獨立腳本預設使用 P10 與原參考庫；參考圖來源只取 train，正負例均為兩圖。

```bash
bash scripts/train_locany_reference_alignment.sh p10-original
```

預設實驗資料夾：`runs/vlm/locany_reference_aligned_p10_original_epoch1`。先用內容描述命名，尚未取新的 V 編號。原 P11 候選可用 `bash scripts/train_locany_reference_alignment.sh p11-context`，仍存到原本 P11 實驗資料夾，不混寫資料或權重。

| 項目 | 設定 |
|---|---|
| 起點 | nvidia/LocateAnything-3B，從 base 訓練，不續接 V8-N |
| 資料 | 相同 fold0 train653／val164；四類各一筆；外部 train 參考 |
| 提示詞 | P10 visual，訓練／推論同一模板 |
| 參考庫 | 原本的 train 緊裁圖，沿用已驗證的參考選取邏輯 |
| 正／負查詢 | 都是目標圖＋外部該類裁圖；負例答 None |
| 答案 | 同類 GT，x1 再 y1 排序，座標 0–1000 |
| 訓練量 | 2,612 steps，依四類查詢資料量設的名義一輪；streaming／packing 實際處理筆數仍以日誌為準 |
| batch／累積 | 1／1 |
| LR／排程 | 5e-5／linear，warmup=0，weight_decay=0 |
| LoRA | LLM rank=8；vision rank=0；MLP 可訓練 |
| attention／block | SDPA／4，沿用本專案可運作設定 |
| 精度 | BF16 |
| 保存／val loss | 每 400 steps；保留最多 3 個 checkpoint，載入 val loss 最佳者 |
| seed | 42 |

這些參數刻意與 V8-N 對齊，推論固定 P10／原參考庫。訓練改用明確的外部裁圖，正負例都是兩圖，並直接使用與推論一致的 P10 模板；仍不是只改一個負例開關，因此改善不能只歸功於單一因素。官方現行腳本另有不同的 block、LoRA rank、LR 與 warmup 預設，尚未在此輪套用，避免混淆。

訓練前會檢查全部 train／val 的兩圖數量、token 對齊與長度，超過 2048 直接中止，不截斷答案。完成後預設自動跑同一 val164，與既有七組一起評分並重新打包；使用 `EVAL_LIMIT=20` 可只先跑同一批 val20。報告包會帶訓練參數、loss 摘要、資料 manifest 與 trainer state，不包含權重。

新模型不能只以 None 回答變多判定成功；必須同時看負類誤報、四類 TP／FN／R／F1，避免變成任何問題都回答 None。raw 與同類去重仍分開列，不預設套用 query_filter。

如果要比較保留下來的 checkpoint 的實際分類 F1，而不只看 loss：

```bash
bash scripts/evaluate_locany_reference_checkpoints.sh p10-original
# 候選方法固定後，再擴至完整 val：
LIMIT=164 bash scripts/evaluate_locany_reference_checkpoints.sh p10-original
```

## 分析產物如何看

- `scores.csv`：每類＋整體 TP／FP／FN／P／R／F1，各種後處理分開。
- `error_types.csv`：FP 按預測類別、FN 按 GT 類別拆開的錯誤原因。
- `per_image_errors.jsonl`：GT、所有預測、配對索引與逐框原因，方便回到圖片。
- `query_compliance.csv`：每次要求何類、輸出多少其他類，及 EOS／輸出長度。
- `reference_sensitivity.csv`：參考與處理後像素是否不同、答案有多少種；須配合同文字條件解讀。
- `confusion_candidates.csv`：錯類 FP 接近何種 GT 的候選統計，並非一對一混淆矩陣。
- 每組 `compare/`：六張同樣目標圖的 GT／預測／參考圖對照。

FP 原因採固定優先序：同類已配對的重複框 → 同類 IoU 0.1–0.5 的定位問題 → 異類 IoU≥0.5 的分類問題 → 異類 IoU 0.1–0.5 的兩者皆錯 → 背景。FN 另標錯類同位、同類框偏、異類框偏、只與已配對框重疊、沒有 IoU≥0.1 候選。這是幾何診斷規則，不能當成模型的心理原因，也不能把各原因消失後的改善量直接相加。

後續 prompt／參數／checkpoint 選擇均使用 validation；目前 205 張 test 保留歷史結果與錯誤分析，不拿它挑新方法。若擴增其他 fold 或重新整合影像，先固定新的資料切分與未參與調整的評估集，再報告正式新 test 結果。
