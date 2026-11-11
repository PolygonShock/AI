import numpy as np

# ==========================================
# 1. 共通計算ロジック（他ファイルから import して使う関数）
# ==========================================

def sigmoid(x):
    """活性化関数 (Sigmoid)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# エンコーダー重み [入力4次元 -> 潜在空間2次元]
W_enc = np.array([
    [ 1.5, -0.8,  1.2, -0.5],
    [-0.6,  1.4, -0.4,  1.1]
])
b_enc = np.array([0.1, -0.1])

# デコーダー重み [潜在空間2次元 -> 復元出力4次元]
W_dec = np.array([
    [ 1.5, -0.6],
    [-0.8,  1.4],
    [ 1.2, -0.4],
    [-0.5,  1.1]
])
b_dec = np.array([0.0, 0.1, -0.1, 0.2])


def process_autoencoder(input_vector_4d):
    """
    4次元の入力ベクトルを受け取り、
    1. エンコード (4次元 -> 2次元潜在空間 / 圧縮)
    2. デコード (2次元 -> 4次元出力 / 復元)
    を行い、復元誤差(MSE)を計算して返す関数
    """
    x = np.array(input_vector_4d, dtype=float)

    # エンコード (圧縮)
    z_latent = sigmoid(np.dot(W_enc, x) + b_enc)

    # デコード (復元)
    x_reconstructed = sigmoid(np.dot(W_dec, z_latent) + b_dec)

    # 復元誤差 (Mean Squared Error)
    mse_loss = float(np.mean((x - x_reconstructed) ** 2))

    return {
        "input_vector": x,
        "latent_representation": z_latent,
        "reconstructed_vector": x_reconstructed,
        "reconstruction_error": mse_loss,
        "is_anomaly": bool(mse_loss > 0.15)  # 誤差が大きい場合は異常検知
    }


# ==========================================
# 2. HTML/JS オートエンコーダー インスペクターUI コード
# ==========================================

HTML_AE_SIMULATOR_CODE = """
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
        button.preset-btn { background-color: #1f2937; border: 1px solid #374151; color: #a7f3d0; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: bold; }
        button.preset-btn:hover { background-color: #374151; }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 0.8rem; color: #9ca3af; display: flex; justify-content: space-between; }
        input[type="range"] { width: 100%; accent-color: #10b981; }
        
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
        <h2>入力データ (4次元)</h2>
        <p style="font-size: 0.75rem; color: #9ca3af;">入力信号を操作して、情報が圧縮・再構成される挙動を確認できます</p>
        
        <div class="preset-btn-group">
            <button class="preset-btn" onclick="setPreset([0.8, 0.2, 0.9, 0.1])">正常信号 A</button>
            <button class="preset-btn" onclick="setPreset([0.1, 0.9, 0.2, 0.8])">正常信号 B</button>
            <button class="preset-btn" onclick="setPreset([0.9, 0.9, 0.1, 0.1])">異常ノイズパターン</button>
            <button class="preset-btn" onclick="setPreset([0.5, 0.5, 0.5, 0.5])">フラット</button>
        </div>

        <div class="control-group" style="margin-top: 10px;">
            <label>入力要素 x1 <span id="val-x1">0.80</span></label>
            <input type="range" id="input-x1" min="0" max="1" value="0.8" step="0.02">
        </div>
        <div class="control-group">
            <label>入力要素 x2 <span id="val-x2">0.20</span></label>
            <input type="range" id="input-x2" min="0" max="1" value="0.2" step="0.02">
        </div>
        <div class="control-group">
            <label>入力要素 x3 <span id="val-x3">0.90</span></label>
            <input type="range" id="input-x3" min="0" max="1" value="0.9" step="0.02">
        </div>
        <div class="control-group">
            <label>入力要素 x4 <span id="val-x4">0.10</span></label>
            <input type="range" id="input-x4" min="0" max="1" value="0.1" step="0.02">
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <div>
                <h1 style="font-size: 1.2rem;">Autoencoder 次元削減・復元シミュレーター</h1>
                <p style="color: #6b7280; font-size: 0.8rem;">入力層 (4D) ➔ [Encoder] ➔ 潜在変数 (2D 瓶の首構造) ➔ [Decoder] ➔ 復元出力層 (4D)</p>
            </div>
            <div class="score-card">
                <div style="font-size: 0.7rem; color: #9ca3af;">復元誤差 (MSE Loss)</div>
                <div class="score-value" id="loss-display">0.000</div>
            </div>
        </div>
        <div class="canvas-container">
            <canvas id="aeCanvas"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('aeCanvas');
        const ctx = canvas.getContext('2d');

        const W_enc = [[1.5,-0.8,1.2,-0.5],[-0.6,1.4,-0.4,1.1]];
        const b_enc = [0.1, -0.1];
        const W_dec = [[1.5,-0.6],[-0.8,1.4],[1.2,-0.4],[-0.5,1.1]];
        const b_dec = [0.0, 0.1, -0.1, 0.2];

        function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

        function setPreset(arr) {
            document.getElementById('input-x1').value = arr[0];
            document.getElementById('input-x2').value = arr[1];
            document.getElementById('input-x3').value = arr[2];
            document.getElementById('input-x4').value = arr[3];
            updateAndDraw();
        }

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }

        function calculateAE() {
            const X = [
                parseFloat(document.getElementById('input-x1').value),
                parseFloat(document.getElementById('input-x2').value),
                parseFloat(document.getElementById('input-x3').value),
                parseFloat(document.getElementById('input-x4').value)
            ];

            document.getElementById('val-x1').textContent = X[0].toFixed(2);
            document.getElementById('val-x2').textContent = X[1].toFixed(2);
            document.getElementById('val-x3').textContent = X[2].toFixed(2);
            document.getElementById('val-x4').textContent = X[3].toFixed(2);

            // Encode (4D -> 2D)
            const Z = [];
            for(let j=0; j<2; j++) {
                let z = b_enc[j];
                for(let i=0; i<4; i++) z += X[i] * W_enc[j][i];
                Z.push(sigmoid(z));
            }

            // Decode (2D -> 4D)
            const Y = [];
            for(let j=0; j<4; j++) {
                let y = b_dec[j];
                for(let i=0; i<2; i++) y += Z[i] * W_dec[j][i];
                Y.push(sigmoid(y));
            }

            // MSE Error
            let mse = 0;
            for(let i=0; i<4; i++) mse += Math.pow(X[i] - Y[i], 2);
            mse /= 4.0;

            document.getElementById('loss-display').textContent = mse.toFixed(4);

            return { X, Z, Y, mse };
        }

        let animOffset = 0;
        function updateAndDraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            animOffset = (animOffset + 0.4) % 20;

            const { X, Z, Y, mse } = calculateAE();

            const layerX = [canvas.width * 0.18, canvas.width * 0.50, canvas.width * 0.82];
            const layerNames = ["入力層 (4次元)", "ボトルネック / 潜在空間 (2次元)", "復元出力層 (4次元)"];

            // タイトル描画
            ctx.fillStyle = '#9ca3af';
            ctx.font = 'bold 13px sans-serif';
            ctx.textAlign = 'center';
            layerNames.forEach((name, i) => {
                ctx.fillText(name, layerX[i], 35);
            });

            // 1. 入力層 -> 潜在層 (Encoder connections)
            ctx.setLineDash([4, 4]);
            ctx.lineDashOffset = -animOffset;
            for(let i=0; i<4; i++) {
                const srcY = (canvas.height / 5) * (i + 1);
                for(let j=0; j<2; j++) {
                    const dstY = (canvas.height / 3) * (j + 1);
                    const w = W_enc[j][i];
                    ctx.beginPath();
                    ctx.moveTo(layerX[0] + 16, srcY);
                    ctx.lineTo(layerX[1] - 16, dstY);
                    ctx.strokeStyle = w > 0 ? `rgba(96, 165, 250, ${Math.abs(w)*0.4})` : `rgba(244, 114, 182, ${Math.abs(w)*0.4})`;
                    ctx.lineWidth = Math.abs(w) * 1.5;
                    ctx.stroke();
                }
            }

            // 2. 潜在層 -> 出力層 (Decoder connections)
            for(let i=0; i<2; i++) {
                const srcY = (canvas.height / 3) * (i + 1);
                for(let j=0; j<4; j++) {
                    const dstY = (canvas.height / 5) * (j + 1);
                    const w = W_dec[j][i];
                    ctx.beginPath();
                    ctx.moveTo(layerX[1] + 16, srcY);
                    ctx.lineTo(layerX[2] - 16, dstY);
                    ctx.strokeStyle = w > 0 ? `rgba(52, 211, 153, ${Math.abs(w)*0.4})` : `rgba(251, 146, 60, ${Math.abs(w)*0.4})`;
                    ctx.lineWidth = Math.abs(w) * 1.5;
                    ctx.stroke();
                }
            }
            ctx.setLineDash([]);

            // 入力ノード描画 (X)
            for(let i=0; i<4; i++) {
                const y = (canvas.height / 5) * (i + 1);
                const val = X[i];
                ctx.beginPath();
                ctx.arc(layerX[0], y, 16, 0, Math.PI * 2);
                const valCol = Math.floor(val * 255);
                ctx.fillStyle = `rgb(${valCol}, ${valCol}, 255)`;
                ctx.fill();
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(2), layerX[0], y);

                ctx.fillStyle = '#9ca3af';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`x_${i+1}`, layerX[0] - 22, y);
            }

            // 潜在空間ノード描画 (Z)
            for(let i=0; i<2; i++) {
                const y = (canvas.height / 3) * (i + 1);
                const val = Z[i];
                ctx.beginPath();
                ctx.arc(layerX[1], y, 18, 0, Math.PI * 2);
                const valCol = Math.floor(val * 255);
                ctx.fillStyle = `rgb(255, ${valCol}, 150)`;
                ctx.fill();
                ctx.strokeStyle = '#f472b6';
                ctx.lineWidth = 2.5;
                ctx.stroke();

                ctx.fillStyle = '#111827';
                ctx.font = 'bold 10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(2), layerX[1], y);

                ctx.fillStyle = '#f472b6';
                ctx.font = '11px sans-serif';
                ctx.fillText(`z_${i+1} (圧縮)`, layerX[1], y - 24);
            }

            // 復元出力ノード描画 (Y)
            for(let i=0; i<4; i++) {
                const y = (canvas.height / 5) * (i + 1);
                const val = Y[i];
                const diff = Math.abs(X[i] - Y[i]);
                ctx.beginPath();
                ctx.arc(layerX[2], y, 16, 0, Math.PI * 2);
                const valCol = Math.floor(val * 255);
                ctx.fillStyle = `rgb(50, ${valCol}, 100)`;
                ctx.fill();
                ctx.strokeStyle = diff > 0.2 ? '#ef4444' : '#34d399';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(2), layerX[2], y);

                ctx.fillStyle = diff > 0.2 ? '#f87171' : '#34d399';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(`y_${i+1} (復元)`, layerX[2] + 22, y);
            }

            requestAnimationFrame(updateAndDraw);
        }

        window.addEventListener('resize', () => { resizeCanvas(); updateAndDraw(); });
        document.querySelectorAll('input').forEach(input => input.addEventListener('input', updateAndDraw));
        resizeCanvas();
        updateAndDraw();
    </script>
</body>
</html>
"""