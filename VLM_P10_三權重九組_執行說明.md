# P10：任務、答案格式與缺陷外觀

P10 是在 P04、P09 之後新增的**推論提示詞**，不重訓 V7、V7-G、V7-N。三種輸入模式各有獨立英文提示詞，完整原文見 `prompts/locany_p10_task_format_features.json`。

提示詞只保留辨識任務、四種缺陷的**不具名外觀描述**、LocateAnything 的 `<ref>`／`<box>` 輸出格式、GT 或 visual 參考圖用途；圖片實際寬高放在每個提示詞的最後一句。不列出 `pore`、`deposit`、`discontinuity`、`stain` 四個類別名稱，讓模型自行產生英文名稱。若輸出同義詞或其他標籤，既有分類評分程式不會當成指定類別；所以必須同時看原始回答、有效框數、被拒絕框數及 F1。

仍以相同的 205 張測試圖、訓練集參考圖、抽樣種子與生成參數比較三權重 × 三種輸入，共九組。`none` 只有目標圖；`gt` 再給另一張訓練集已標框圖；`visual` 再給訓練集無框缺陷裁圖，四類各查一次。參考圖不得是目標測試圖的答案。

在 4090 執行：

```bash
cd ~/weldguard
git pull --ff-only origin master
bash scripts/evaluate_locany_v7_nine_test.sh P10
```

結果保存在 `runs/vlm/v7_p10_nine_test/`：`scores.txt`、`counts.txt`、`p04_vs_p10.csv`、九組原始預測及固定八張 GT／預測對照圖。報告 ZIP 為 `results/v7_p10_nine_test_report.zip`，不含模型權重、原始資料集或完整預測 JSONL。

```bash
git add -f results/v7_p10_nine_test_report.zip
git commit -m "Add V7 P10 nine-way inference results"
git push origin HEAD
```

這是提示詞探索，test 已重複用於多輪比較；不要把由此選出的提示詞在這批 test 上的最好分數當作未見資料的最終盲測分數。
