# V8：P10 三權重 × 三種推論輸入

V8、V8-G、V8-N 從同一個 `nvidia/LocateAnything-3B` 原模型開始，沿用 V7 的訓練參數、653 張 fold0 訓練圖、164 張 fold0 驗證圖和以左上角 x1、y1 排序的 GT 答案。這一輪把訓練提示詞改成修正後的 P10。三種權重分別是：

| 權重 | 訓練輸入 | 訓練 steps |
| --- | --- | ---: |
| V8 | 目標原圖 + P10 | 653 |
| V8-G | 目標原圖 + 另一張訓練集 GT 標框圖 + P10 | 653 |
| V8-N | NVIDIA `visual_prompt` 轉換：四類各一筆；有該類時自動裁出參考區域，沒有時使用文字查詢 | 2612 |

V8-N 為了觸發 NVIDIA 的原生轉換，訓練資料開頭必須包含該類的英文類名與官方 `category set` 句式；其餘任務、外觀描述、答案格式與尺寸沿用 P10。這是 V8-N 與普通 P10 提示詞唯一必要的格式差別。三種權重都在 fold0/val 記錄驗證 loss，從最佳驗證 loss 的 checkpoint 載入最終模型。V8 不讀取 205 張 test 來訓練或選 checkpoint。

在 4090 的 `~/weldguard` 執行，每次只訓練一個權重：

```bash
cd ~/weldguard
git pull --ff-only origin master
bash scripts/train_locany_v8_family.sh v8
bash scripts/train_locany_v8_family.sh v8-g
bash scripts/train_locany_v8_family.sh v8-n
```

三個權重都完成後，執行同一批 205 張 test 圖、同一 P10 推論提示詞及生成參數的九組比較：

```bash
bash scripts/evaluate_locany_v8_nine_test.sh
```

每個權重各跑 `none`（只給目標圖）、`gt`（另給已標框訓練參考圖）、`visual`（另給四類各一張無框缺陷裁圖，分四次詢問）。評分保留類別，IoU 門檻 0.5。結果在 `runs/vlm/v8_p10_nine_test/`，包括九組原始預測、scores、有效／拒絕框數與各組 GT／預測並排圖。訓練 loss 在各權重的 `runs/vlm/locany_v8*_p10_xy/`。可分享的報告封包會建立在 `results/v8_p10_nine_test_report.zip`，不包含模型權重、完整原始資料集與 205 張逐筆預測。

要將結果帶回這台電腦，再執行：

```bash
git add -f results/v8_p10_nine_test_report.zip
git commit -m "Add V8 P10 nine-way test results"
git push origin HEAD
```

本輪與 V7 共用資料切分，因此可以比較，但 V8-P10 對 V7-P04 同時更動了訓練與推論提示詞；若要單獨估計訓練提示詞的影響，還需拿 V7-P10 作同條件對照。test 已用於多輪探索；不宜再把在此選出的最高分當成全新盲測結果。

## 從逐張結果挑選差異案例

九組推論完成後，可挑出最多 12 張差距較大的 test 圖。每張比較圖同時顯示 GT、訓練集 GT 參考圖、四類 visual 裁圖及九組實際標框；封包也包含 205 張逐張 TP／FP／FN／F1 表，不含完整資料集或模型權重。

```bash
cd ~/weldguard
git pull --ff-only origin master
source .venv-vlm/bin/activate
python scripts/pack_locany_v8_difference_cases.py
git add -f results/v8_p10_difference_cases.zip
git commit -m "Add selected V8 P10 disagreement examples"
git push origin HEAD
```
