# 本機焊接缺陷提示詞實驗

這個資料夾用 **同一張未畫框原圖、同一個 GPT 視覺模型、同一種輸出座標規則** 比較提示詞。預設比較 P04 與 P08。這是提示詞探索，**不是 LocateAnything 的訓練或測試結果**；GPT 的改善也不能直接預測 LocateAnything 微調後的改善。

## 從 4090 取得固定 100 張 validation 原圖

為了避免挑圖偏差，使用先前 P01／P03／P04／P05 實驗記錄中的**同一批 100 張 fold0 validation** 名單；本機可從中固定抽 50 張先跑。這批圖已用於提示詞探索，不能當獨立 test。壓縮檔只含原圖與 YOLO GT 標籤，**不放進 GitHub**。

在 4090 的 `~/weldguard` 更新程式碼後執行：

```bash
python scripts/pack_prompt_lab_val100.py
```

會產生 `~/weldguard/datasets/derived/prompt_lab_val100.zip`。若 Windows 能用 SSH 連線，從本專案的 Git 工作目錄執行下列指令；請將 `<4090-IP>` 換成實際位址：

```powershell
scp leo@<4090-IP>:/home/leo/weldguard/datasets/derived/prompt_lab_val100.zip .\prompt_lab\transfer\
```

也可用遠端桌面檔案傳輸或 USB，將 ZIP 放到 [`transfer`](transfer)；不需要透過 Git 傳資料。可先核對壓縮檔中的 `manifest.json`，應記錄 100 張、`fold0/val`。

## 提示詞版本

| 名稱 | 提示詞檔 | 差異 |
| --- | --- | --- |
| P04 | `../prompts/locany_p04_balanced.txt` | 已用過的平衡描述。 |
| P08 | `../prompts/locany_p08_detailed.txt` | 保留四類與 0–1000 座標；刪掉角色設定；詳述各類視覺特徵、兩個對角座標、局部框和不亂補缺陷。尚未訓練。 |

腳本會對兩版附加**完全相同的圖片寬高與 API 輸出格式文字**，避免因 P04 沒有把 `<ref>`／`<box>` 格式寫清楚而無法解析。這使本機實驗的 P04 輸入比歷史上的 P04 多了共同的格式附註；輸出不能直接跟先前 LocateAnything 分數相提並論。兩版都用 LocateAnything 的 0–1000 正規化座標；圖片的原始像素寬高只是輔助資訊。`(x1,y1)` 是單一框的左上角，`(x2,y2)` 是同一框的右下角。若圖片寬 1200 像素，模型輸出 x=500，對應的畫圖位置就是 600 像素。YOLO GT 標籤只用於本機計分，絕不送給 API。

## 準備

1. 把一張**未畫框**的 JPG／PNG 放進 [`input_images`](input_images)。不要把 `results/` 中已畫框的圖當題目。
2. 這台電腦的 Codex Python 已有 Pillow。先在 PowerShell 指定它：

   ```powershell
   $python = 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
   ```

   若要用自己安裝的 Python，可在本資料夾建立環境並安裝依賴，再把 `$python` 改成該環境的執行檔：

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   $python = (Resolve-Path .\.venv\Scripts\python.exe).Path
   ```

3. 在當前 PowerShell 工作階段設定 API key。指令會隱藏輸入；不要把 key 寫進檔案或貼到對話中。

   ```powershell
   $secureKey = Read-Host 'OpenAI API key' -AsSecureString
   $env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
   ```

   若要讓 Codex 在獨立的本機執行程序中替你跑，可把 key 存在本資料夾的 `.api_key`；此檔已由 `.gitignore` 排除。實驗完成後可刪除它：

   ```powershell
   $secureKey = Read-Host 'OpenAI API key' -AsSecureString
   [System.IO.File]::WriteAllText((Join-Path (Get-Location) '.api_key'), [System.Net.NetworkCredential]::new('', $secureKey).Password)
   ```

## 執行

從本資料夾執行，先檢查展開後的文字（不呼叫 API、不產生費用）：

```powershell
& $python .\run_prompt_ab.py --image .\input_images\你的原圖.jpg --model gpt-5.5 --dry-run
```

正式對照會對同一張圖片呼叫 API 兩次，兩次使用相同模型與影像細節設定：

```powershell
& $python .\run_prompt_ab.py --image .\input_images\你的原圖.jpg --model gpt-5.5
```

若已收到 4090 的 ZIP，先跑一張確認 API key、解析與疊框正常，再跑固定 50 張；50 張會產生 100 次 API 請求，請先留意帳戶額度：

```powershell
& $python .\run_batch.py --model gpt-5.5 --limit 1
& $python .\run_batch.py --model gpt-5.5 --limit 50
```

要對完整的同一批 100 張做比較，改用 `--limit 100`。批次輸出 `summary.csv`／`summary.json` 會把每張的 TP、FP、FN 彙總後計算分類與不分類的 precision、recall、F1。若中斷，可以指定相同的 `--batch-dir` 續跑。

如果有該圖的 YOLO 標註檔，可加 `--gt-yolo C:\path\to\labels\你的原圖.txt`，獲得分類與不分類的 IoU≥0.5、TP／FP／FN、precision／recall／F1。沒有 GT 時仍可比較疊框圖，但不能把主觀觀察寫成正式分數。要比較其他版本，可重複 `--prompt 名稱=檔案路徑`；一旦指定，需列出所有要比較的版本。

每次結果在 `runs/圖片名_時間/`，包含 `comparison.png`、各版提示詞快照、API 原始回應、模型文字回答、有效／拒收框、可選的評分，以及模型、圖片 SHA-256 和 token 使用量。這些圖片與輸出預設不進 Git。

## 比較時注意

- 先看 `comparison.png` 的框是否落在局部缺陷，再看分類；單張圖只能做初步診斷。提示詞選擇應用 validation 圖，保留獨立 test 到最後。
- 兩個 API 請求的模型與影像相同，生成結果仍可能有隨機變動；要比較穩定性可對多張 validation 圖逐張執行。
- GPT 可能在內部處理圖片尺寸；指定 `--detail original` 盡量保留定位細節，但輸出座標仍須靠 GT 與疊框檢查。若所選模型不支援 `original`，改用 `--detail high`。
- P08 目前是候選提示詞。正式拿到 4090 訓練前，應另記下固定的資料切分、完整提示詞和參數；不要用同一份獨立 test 反覆挑提示詞。

API 圖片輸入與 Responses 端點：[OpenAI Images and vision](https://developers.openai.com/api/docs/guides/images-vision)、[OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)。LocateAnything 使用 0–1000 座標的資料格式見 [NVIDIA Eagle 資料準備文件](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/DATA_PREPARATION.md)。
