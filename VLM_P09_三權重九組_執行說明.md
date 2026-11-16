# P09：V7 三權重 × 三種推論輸入

本輪**不重訓模型**。沿用 V7、V7-G、V7-N 三個權重與上輪相同的 205 張圖片、參考圖來源、隨機種子、生成設定及分類 IoU 0.5 評分。只換成 `prompts/locany_p09_inference.json` 中三段各自獨立的英文極簡推論提示詞。P09 不包含影像尺寸、四個類別名稱或輸出格式規則，這些都屬於本次要觀察的差異。

| 推論模式 | 模型看到的圖 | 查詢數／測試圖 |
|---|---|---:|
| `none` | 待辨識原圖 | 1 |
| `gt` | 待辨識原圖＋另一張訓練集的已標框圖 | 1 |
| `visual` | 待辨識原圖＋訓練集的單類缺陷裁圖；四類分開問 | 4 |

參考圖只從 fold0/train 取，不可用待辨識圖片自己的 GT 當第二張圖。visual 組仍使用模型產生的類別標籤評分，不用裁圖所屬類別代填答案。

在 4090 的 `~/weldguard` 中執行：

```bash
cd ~/weldguard
git pull --ff-only origin master
bash scripts/evaluate_locany_v7_nine_test.sh P09
```

推論與評分保存在 `runs/vlm/v7_p09_nine_test/`；`scores.txt` 是九組分類指標，`counts.txt` 記錄有效框、被拒絕的數值框及沒有有效框的圖片數，`p04_vs_p09.csv` 是與既有 P04 九組結果的逐組對照。每組保留原始答案與 8 張 GT／預測圖。本機打包到 `results/v7_p09_nine_test_report.zip`，不含模型權重、資料集或完整預測紀錄。若中斷，重跑相同指令會接續尚未完成的圖片。

若要把報告 ZIP 傳到 GitHub：

```bash
git add -f results/v7_p09_nine_test_report.zip
git commit -m "Add V7 P09 nine-way inference results"
git push origin HEAD
```

P09 的極簡文字可能改變模型輸出格式。`rejected_numeric_boxes` 只計算**可解析但無效的數值框**，不涵蓋所有自由文字或格式錯誤；請連同原始答案、有效框數與分類分數解讀。P04→P09 同時改變了多個提示詞元素，因此分數差異不能單獨歸因於「有沒有說明第二張圖」。這批 test 已反覆用於提示詞探索，後續若需正式盲測應保留新的獨立測試集。
