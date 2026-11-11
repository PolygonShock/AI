import numpy as np

# ==========================================
# 1. 共通計算ロジック（他ファイルから import して使う関数）
# ==========================================

def tanh(x):
    """RNNで標準的に使われる活性化関数 (tanh)"""
    return np.tanh(x)

# 簡易RNNパラメータ（1次元時系列データ -> 隠れ状態4次元 -> 1次元出力）
# Whh: 隠れ状態間の重み, Wxh: 入力から隠れ状態への重み, Why: 隠れ状態から出力への重み
Whh = np.array([
    [ 0.6, -0.2,  0.1,  0.3],
    [-0.1,  0.5,  0.4, -0.2],
    [ 0.3, -0.3,  0.7,  0.1],
    [ 0.2,  0.1, -0.2,  0.6]
])
Wxh = np.array([[0.8], [-0.5], [0.4], [-0.3]])
bh = np.array([0.1, -0.1, 0.0, 0.2])

Why = np.array([[0.7, -0.4, 0.6, -0.3]])
by = 0.05


def predict_rnn_sequence(sequence_values):
    """
    時系列配列（例: [0.2, 0.4, 0.6, 0.8] など）を受け取り、
    タイムステップごとの隠れ状態(h)の遷移と次の値の予測出力を計算する関数
    """
    seq = np.array(sequence_values, dtype=float)
    timesteps = len(seq)
    
    hidden_states = []
    h_prev = np.zeros(4)  # 初期隠れ状態 h0

    # タイムステップごとの順伝播（記憶の引き継ぎ）
    for t in range(timesteps):
        x_t = seq[t]
        # h_t = tanh( Wxh * x_t + Whh * h_prev + bh )
        h_t = tanh(np.dot(Wxh.flatten(), x_t) + np.dot(Whh, h_prev) + bh)
        hidden_states.append(h_t)
        h_prev = h_t

    # 最終タイムステップの隠れ状態から次のステップの値を予測
    y_pred = np.dot(Why, h_prev) + by

    return {
        "input_sequence": seq,
        "hidden_states": np.array(hidden_states),
        "final_hidden_state": h_prev,
        "next_prediction": float(y_pred[0])
    }


# ==========================================
# 2. HTML/JS RNN インスペクターUI コード
# ==========================================

HTML_RNN_SIMULATOR_CODE = """
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
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 0.8rem; color: #9ca3af; display: flex; justify-content: space-between; }
        input[type="range"] { width: 100%; accent-color: #3b82f6; }
        
        .main-content { flex: 1; display: flex; flex-direction: column; position: relative; }
        .header { padding: 16px 20px; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
        .score-card { background-color: #1f2937; padding: 8px 16px; border-radius: 6px; text-align: right; }
        .score-value { font-size: 1.4rem; font-weight: bold; color: #f472b6; }
        .canvas-container { flex: 1; position: relative; }
        canvas { width: 100%; height: 100%; display: block; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>時系列データの入力 (t=1 ~ t=4)</h2>
        <p style="font-size: 0.75rem; color: #9ca3af;">各時点の値を変更すると、前後の記憶(隠れ状態)がどう引き継がれるか視覚化されます</p>
        
        <div class="preset-btn-group">
            <button class="preset-btn" onclick="setPreset([0.2, 0.4, 0.6, 0.8])">上昇トレンド</button>
            <button class="preset-btn" onclick="setPreset([0.9, 0.7, 0.4, 0.2])">下降トレンド</button>
            <button class="preset-btn" onclick="setPreset([0.3, 0.8, 0.2, 0.9])">激しい変動</button>
            <button class="preset-btn" onclick="setPreset([0.5, 0.5, 0.5, 0.5])">平坦データ</button>
        </div>

        <div class="control-group" style="margin-top: 10px;">
            <label>時点 t=1 の入力 <span id="val-t1">0.20</span></label>
            <input type="range" id="input-t1" min="-1" max="1" value="0.2" step="0.05">
        </div>
        <div class="control-group">
            <label>時点 t=2 の入力 <span id="val-t2">0.40</span></label>
            <input type="range" id="input-t2" min="-1" max="1" value="0.4" step="0.05">
        </div>
        <div class="control-group">
            <label>時点 t=3 の入力 <span id="val-t3">0.60</span></label>
            <input type="range" id="input-t3" min="-1" max="1" value="0.6" step="0.05">
        </div>
        <div class="control-group">
            <label>時点 t=4 の入力 <span id="val-t4">0.80</span></label>
            <input type="range" id="input-t4" min="-1" max="1" value="0.8" step="0.05">
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <div>
                <h1 style="font-size: 1.2rem;">RNN 時系列記憶・予測シミュレーター</h1>
                <p style="color: #6b7280; font-size: 0.8rem;">過去の「記憶(隠れ状態 h_t)」を次の時間ステップへループレシーブして未来(t=5)を予測</p>
            </div>
            <div class="score-card">
                <div style="font-size: 0.7rem; color: #9ca3af;">次時点 (t=5) の予測値</div>
                <div class="score-value" id="pred-display">0.00</div>
            </div>
        </div>
        <div class="canvas-container">
            <canvas id="rnnCanvas"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('rnnCanvas');
        const ctx = canvas.getContext('2d');

        const Whh = [[0.6,-0.2,0.1,0.3],[-0.1,0.5,0.4,-0.2],[0.3,-0.3,0.7,0.1],[0.2,0.1,-0.2,0.6]];
        const Wxh = [0.8, -0.5, 0.4, -0.3];
        const bh = [0.1, -0.1, 0.0, 0.2];
        const Why = [0.7, -0.4, 0.6, -0.3];
        const by = 0.05;

        function tanh(x) { return Math.tanh(x); }

        function setPreset(arr) {
            document.getElementById('input-t1').value = arr[0];
            document.getElementById('input-t2').value = arr[1];
            document.getElementById('input-t3').value = arr[2];
            document.getElementById('input-t4').value = arr[3];
            updateAndDraw();
        }

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }

        function calculateRNN() {
            const inputs = [
                parseFloat(document.getElementById('input-t1').value),
                parseFloat(document.getElementById('input-t2').value),
                parseFloat(document.getElementById('input-t3').value),
                parseFloat(document.getElementById('input-t4').value)
            ];

            document.getElementById('val-t1').textContent = inputs[0].toFixed(2);
            document.getElementById('val-t2').textContent = inputs[1].toFixed(2);
            document.getElementById('val-t3').textContent = inputs[2].toFixed(2);
            document.getElementById('val-t4').textContent = inputs[3].toFixed(2);

            let hiddenStates = [];
            let hPrev = [0, 0, 0, 0];

            for(let t=0; t<4; t++) {
                let x = inputs[t];
                let hNext = [];
                for(let i=0; i<4; i++) {
                    let z = Wxh[i] * x + bh[i];
                    for(let j=0; j<4; j++) z += Whh[i][j] * hPrev[j];
                    hNext.push(tanh(z));
                }
                hiddenStates.push(hNext);
                hPrev = hNext;
            }

            let yPred = by;
            for(let i=0; i<4; i++) yPred += Why[i] * hPrev[i];

            document.getElementById('pred-display').textContent = yPred.toFixed(2);

            return { inputs, hiddenStates, yPred };
        }

        let animOffset = 0;
        function updateAndDraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            animOffset = (animOffset + 0.4) % 20;

            const { inputs, hiddenStates, yPred } = calculateRNN();

            const stepX = [canvas.width * 0.15, canvas.width * 0.35, canvas.width * 0.55, canvas.width * 0.75, canvas.width * 0.90];

            // タイムステップタイトル
            ctx.fillStyle = '#9ca3af';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            for(let t=0; t<4; t++) ctx.fillText(`ステップ t=${t+1}`, stepX[t], 35);
            ctx.fillText(`予測 t=5`, stepX[4], 35);

            // ノードと接続線の描き込み
            for(let t=0; t<4; t++) {
                const x = stepX[t];
                const inputY = canvas.height * 0.75;
                const hiddenYBase = canvas.height * 0.30;

                // 入力ノード (x_t)
                ctx.beginPath();
                ctx.arc(x, inputY, 16, 0, Math.PI * 2);
                ctx.fillStyle = '#1e3a8a';
                ctx.fill();
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(inputs[t].toFixed(2), x, inputY);
                
                ctx.fillStyle = '#60a5fa';
                ctx.font = '11px sans-serif';
                ctx.fillText(`入力 x_${t+1}`, x, inputY + 28);

                // 隠れ層ノード (h_t) [4次元]
                for(let i=0; i<4; i++) {
                    const hY = hiddenYBase + i * 40;
                    const val = hiddenStates[t][i];

                    ctx.beginPath();
                    ctx.arc(x, hY, 14, 0, Math.PI * 2);
                    const norm = Math.floor(((val + 1) / 2) * 255);
                    ctx.fillStyle = `rgb(${norm}, 50, ${255 - norm})`;
                    ctx.fill();
                    ctx.strokeStyle = '#f472b6';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    ctx.fillStyle = '#ffffff';
                    ctx.font = '9px monospace';
                    ctx.fillText(val.toFixed(2), x, hY);

                    // 入力 -> 隠れ層の接続線
                    ctx.beginPath();
                    ctx.moveTo(x, inputY - 16);
                    ctx.lineTo(x, hY + 14);
                    ctx.strokeStyle = 'rgba(96, 165, 250, 0.3)';
                    ctx.stroke();

                    // 時間ループの隠れ状態受け渡し線 (h_{t-1} -> h_t)
                    if(t < 3) {
                        const nextX = stepX[t+1];
                        ctx.beginPath();
                        ctx.moveTo(x + 14, hY);
                        ctx.lineTo(nextX - 14, hY);
                        ctx.strokeStyle = 'rgba(244, 114, 182, 0.6)';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([4, 4]);
                        ctx.lineDashOffset = -animOffset;
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                }
            }

            // 最終隠れ層 (t=4) -> 予測出力 (t=5) の線
            const lastX = stepX[3];
            const outX = stepX[4];
            const outY = canvas.height * 0.45;

            for(let i=0; i<4; i++) {
                const hY = canvas.height * 0.30 + i * 40;
                ctx.beginPath();
                ctx.moveTo(lastX + 14, hY);
                ctx.lineTo(outX - 18, outY);
                ctx.strokeStyle = 'rgba(52, 211, 153, 0.5)';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // 予測出力ノード (y_pred)
            ctx.beginPath();
            ctx.arc(outX, outY, 18, 0, Math.PI * 2);
            ctx.fillStyle = '#065f46';
            ctx.fill();
            ctx.strokeStyle = '#34d399';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(yPred.toFixed(2), outX, outY);

            ctx.fillStyle = '#34d399';
            ctx.font = '11px sans-serif';
            ctx.fillText(`予測 y_5`, outX, outY + 30);

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