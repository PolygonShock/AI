import numpy as np

# ==========================================
# 1. 共通計算ロジック（他ファイルから import して使う関数）
# ==========================================

def relu(x):
    """活性化関数 (ReLU)"""
    return np.maximum(0, x)

def softmax(x):
    """出力層の活性化関数 (Softmax)"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# エッジ検出用カーネル（縦線・横線抽出）
KERNEL_VERTICAL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

KERNEL_HORIZONTAL = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])


def process_cnn_feature_map(image_matrix_5x5):
    """
    5x5の入力画像行列から、畳み込み(Convolution) -> Pooling -> 全結合(FC) までの
    中間特徴マップと識別確率を順伝播計算する関数
    """
    img = np.array(image_matrix_5x5, dtype=float)

    # 1. 畳み込み処理 (Convolution: 5x5 -> 3x3)
    conv_vert = np.zeros((3, 3))
    conv_horiz = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            patch = img[i:i+3, j:j+3]
            conv_vert[i, j] = np.sum(patch * KERNEL_VERTICAL)
            conv_horiz[i, j] = np.sum(patch * KERNEL_HORIZONTAL)

    # ReLU 活性化
    act_vert = relu(conv_vert)
    act_horiz = relu(conv_horiz)

    # 2. プーリング処理 (Max Pooling: 3x3 -> 2x2)
    pool_vert = np.array([
        [np.max(act_vert[0:2, 0:2]), np.max(act_vert[0:2, 1:3])],
        [np.max(act_vert[1:3, 0:2]), np.max(act_vert[1:3, 1:3])]
    ])
    pool_horiz = np.array([
        [np.max(act_horiz[0:2, 0:2]), np.max(act_horiz[0:2, 1:3])],
        [np.max(act_horiz[1:3, 0:2]), np.max(act_horiz[1:3, 1:3])]
    ])

    # 3. 平坦化 (Flatten: 2x2 + 2x2 -> 8次元)
    flat_vector = np.concatenate([pool_vert.flatten(), pool_horiz.flatten()])

    # 4. 全結合層 & 分類 (3クラス)
    score_v = np.sum(pool_vert)
    score_h = np.sum(pool_horiz)
    
    logits = np.array([
        score_v * 1.5 - score_h * 0.5,  # 縦線スコア
        score_h * 1.5 - score_v * 0.5,  # 横線スコア
        (score_v + score_h) * 0.8       # クロス/その他スコア
    ])
    probabilities = softmax(logits)
    classes = ["縦線パターン", "横線パターン", "クロス / その他"]
    pred_class_idx = np.argmax(probabilities)

    return {
        "input_image": img,
        "conv_vert": act_vert,
        "conv_horiz": act_horiz,
        "pool_vert": pool_vert,
        "pool_horiz": pool_horiz,
        "probabilities": probabilities,
        "predicted_label": classes[pred_class_idx],
        "confidence": float(probabilities[pred_class_idx] * 100)
    }


# ==========================================
# 2. HTML/JS CNN インスペクターUI コード
# ==========================================

HTML_CNN_SIMULATOR_CODE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: #0b0f19; color: #e2e8f0; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }
        .sidebar { width: 320px; background-color: #111827; border-right: 1px solid #1f2937; padding: 20px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
        .sidebar h2 { font-size: 1rem; color: #f3f4f6; }
        .preset-btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        button.preset-btn { background-color: #1f2937; border: 1px solid #374151; color: #60a5fa; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: bold; }
        button.preset-btn:hover { background-color: #374151; }
        .grid-input { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-top: 10px; width: 200px; margin: 0 auto; }
        .pixel-btn { width: 36px; height: 36px; background-color: #111827; border: 1px solid #374151; border-radius: 4px; cursor: pointer; }
        .pixel-btn.active { background-color: #3b82f6; box-shadow: 0 0 8px #3b82f6; }
        .main-content { flex: 1; display: flex; flex-direction: column; position: relative; }
        .header { padding: 16px 20px; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
        .score-card { background-color: #1f2937; padding: 8px 16px; border-radius: 6px; text-align: right; }
        .score-value { font-size: 1.4rem; font-weight: bold; color: #34d399; }
        .canvas-container { flex: 1; position: relative; }
        canvas { width: 100%; height: 100%; display: block; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>入力パターン作成 (5x5)</h2>
        <p style="font-size: 0.75rem; color: #9ca3af;">マス目をクリックして自由に入力画像を当てはめてください</p>
        
        <div class="preset-btn-group">
            <button class="preset-btn" onclick="loadPreset('vertical')">縦線</button>
            <button class="preset-btn" onclick="loadPreset('horizontal')">横線</button>
            <button class="preset-btn" onclick="loadPreset('cross')">クロス</button>
            <button class="preset-btn" onclick="loadPreset('clear')">リセット</button>
        </div>

        <div style="display:flex; justify-content:center;">
            <div class="grid-input" id="gridInput"></div>
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <div>
                <h1 style="font-size: 1.2rem;">CNN 画像特徴抽出・認識シミュレーター</h1>
                <p style="color: #6b7280; font-size: 0.8rem;">入力画像 ➔ 畳み込み層 (Conv) ➔ プーリング層 (Pool) ➔ 全結合分類層 (FC)</p>
            </div>
            <div class="score-card">
                <div style="font-size: 0.7rem; color: #9ca3af;">判定予測結果</div>
                <div class="score-value" id="pred-display">-</div>
            </div>
        </div>
        <div class="canvas-container">
            <canvas id="cnnCanvas"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('cnnCanvas');
        const ctx = canvas.getContext('2d');

        let gridData = [
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0]
        ];

        const layerNames = ["入力画像\\n(5x5)", "畳み込み層\\n(Conv 3x3)", "プーリング層\\n(Pool 2x2)", "全結合 / 分類\\n(Output)"];
        const convLabels = ["縦エッジ検出", "横エッジ検出"];
        const classLabels = ["縦線パターン", "横線パターン", "クロス / その他"];

        function initGridUI() {
            const container = document.getElementById('gridInput');
            container.innerHTML = '';
            for(let r=0; r<5; r++){
                for(let c=0; c<5; c++){
                    const btn = document.createElement('button');
                    btn.className = 'pixel-btn' + (gridData[r][c] ? ' active' : '');
                    btn.onclick = () => {
                        gridData[r][c] = gridData[r][c] ? 0 : 1;
                        btn.className = 'pixel-btn' + (gridData[r][c] ? ' active' : '');
                        draw();
                    };
                    container.appendChild(btn);
                }
            }
        }

        function loadPreset(type) {
            gridData = Array(5).fill(0).map(() => Array(5).fill(0));
            if(type === 'vertical') {
                for(let i=0; i<5; i++) gridData[i][2] = 1;
            } else if(type === 'horizontal') {
                for(let j=0; j<5; j++) gridData[2][j] = 1;
            } else if(type === 'cross') {
                for(let i=0; i<5; i++) { gridData[i][2] = 1; gridData[2][i] = 1; }
            }
            initGridUI();
            draw();
        }

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }

        function relu(v) { return Math.max(0, v); }

        function calculateCNN() {
            let convV = Array(3).fill(0).map(() => Array(3).fill(0));
            let convH = Array(3).fill(0).map(() => Array(3).fill(0));
            const kV = [[-1,0,1],[-1,0,1],[-1,0,1]];
            const kH = [[-1,-1,-1],[0,0,0],[1,1,1]];

            for(let r=0; r<3; r++){
                for(let c=0; c<3; c++){
                    let sV = 0, sH = 0;
                    for(let kr=0; kr<3; kr++){
                        for(let kc=0; kc<3; kc++){
                            sV += gridData[r+kr][c+kc] * kV[kr][kc];
                            sH += gridData[r+kr][c+kc] * kH[kr][kc];
                        }
                    }
                    convV[r][c] = relu(sV);
                    convH[r][c] = relu(sH);
                }
            }

            let poolV = [
                [Math.max(convV[0][0], convV[0][1], convV[1][0], convV[1][1]), Math.max(convV[0][1], convV[0][2], convV[1][1], convV[1][2])],
                [Math.max(convV[1][0], convV[1][1], convV[2][0], convV[2][1]), Math.max(convV[1][1], convV[1][2], convV[2][1], convV[2][2])]
            ];
            let poolH = [
                [Math.max(convH[0][0], convH[0][1], convH[1][0], convH[1][1]), Math.max(convH[0][1], convH[0][2], convH[1][1], convH[1][2])],
                [Math.max(convH[1][0], convH[1][1], convH[2][0], convH[2][1]), Math.max(convH[1][1], convH[1][2], convH[2][1], convH[2][2])]
            ];

            let scoreV = poolV.flat().reduce((a,b)=>a+b, 0);
            let scoreH = poolH.flat().reduce((a,b)=>a+b, 0);
            
            let scores = [scoreV * 1.5 - scoreH * 0.5, scoreH * 1.5 - scoreV * 0.5, (scoreV + scoreH) * 0.8];
            const maxS = Math.max(...scores);
            const exps = scores.map(s => Math.exp(s - maxS));
            const sumExp = exps.reduce((a,b)=>a+b, 0);
            const probs = exps.map(e => e / sumExp);

            const maxIdx = probs.indexOf(Math.max(...probs));
            document.getElementById('pred-display').textContent = classLabels[maxIdx] + ` (${(probs[maxIdx]*100).toFixed(0)}%)`;

            return { convV, convH, poolV, poolH, probs };
        }

        function drawGrid(startX, startY, matrix, cellSize, title) {
            const rows = matrix.length;
            const cols = matrix[0].length;
            
            if(title) {
                ctx.fillStyle = '#e2e8f0';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(title, startX + (cols * cellSize)/2, startY - 10);
            }

            for(let r=0; r<rows; r++){
                for(let c=0; c<cols; c++){
                    const val = matrix[r][c];
                    const x = startX + c * cellSize;
                    const y = startY + r * cellSize;
                    
                    ctx.beginPath();
                    ctx.rect(x, y, cellSize - 2, cellSize - 2);
                    
                    const norm = Math.min(1, Math.max(0, val / 3.0));
                    ctx.fillStyle = val > 0 ? `rgb(59, 130, 246)` : `#161e2e`;
                    ctx.globalAlpha = val > 0 ? Math.max(0.4, norm) : 1.0;
                    ctx.fill();
                    ctx.globalAlpha = 1.0;
                    
                    ctx.strokeStyle = '#2a364f';
                    ctx.stroke();

                    ctx.fillStyle = '#ffffff';
                    ctx.font = '10px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(val.toFixed(0), x + cellSize/2, y + cellSize/2);
                }
            }
        }

        let animOffset = 0;
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            animOffset = (animOffset + 0.4) % 20;

            const res = calculateCNN();

            const layerX = [canvas.width * 0.12, canvas.width * 0.38, canvas.width * 0.65, canvas.width * 0.88];
            ctx.fillStyle = '#9ca3af';
            ctx.font = 'bold 13px sans-serif';
            ctx.textAlign = 'center';
            layerNames.forEach((name, i) => {
                const lines = name.split('\\n');
                lines.forEach((line, lIdx) => {
                    ctx.fillText(line, layerX[i], 35 + lIdx * 16);
                });
            });

            drawGrid(layerX[0] - 50, 110, gridData, 22, "");

            drawGrid(layerX[1] - 35, 100, res.convV, 24, convLabels[0]);
            drawGrid(layerX[1] - 35, 230, res.convH, 24, convLabels[1]);

            drawGrid(layerX[2] - 25, 110, res.poolV, 26, "Pool (縦)");
            drawGrid(layerX[2] - 25, 240, res.poolH, 26, "Pool (横)");

            ctx.strokeStyle = '#60a5fa';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.lineDashOffset = -animOffset;

            ctx.beginPath();
            ctx.moveTo(layerX[1] + 45, 135); ctx.lineTo(layerX[2] - 35, 135);
            ctx.moveTo(layerX[1] + 45, 265); ctx.lineTo(layerX[2] - 35, 265);
            ctx.stroke();
            ctx.setLineDash([]);

            const outY = [120, 190, 260];
            res.probs.forEach((p, i) => {
                const x = layerX[3];
                const y = outY[i];

                ctx.beginPath();
                ctx.arc(x, y, 18, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(52, 211, 153, ${Math.max(0.2, p)})`;
                ctx.fill();
                ctx.strokeStyle = '#34d399';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText((p * 100).toFixed(0) + '%', x, y);

                ctx.fillStyle = '#e2e8f0';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(classLabels[i], x + 26, y);

                ctx.strokeStyle = `rgba(52, 211, 153, ${Math.max(0.15, p * 0.8)})`;
                ctx.beginPath();
                ctx.moveTo(layerX[2] + 35, 135); ctx.lineTo(x - 20, y);
                ctx.moveTo(layerX[2] + 35, 265); ctx.lineTo(x - 20, y);
                ctx.stroke();
            });

            requestAnimationFrame(draw);
        }

        window.addEventListener('resize', () => { resizeCanvas(); draw(); });
        initGridUI();
        resizeCanvas();
        draw();
    </script>
</body>
</html>
"""