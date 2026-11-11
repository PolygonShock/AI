import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# 1. 共通計算ロジック（他ファイルから import して使う部分）
# ==========================================

def sigmoid(x):
    """活性化関数 (Sigmoid)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# 重みとバイアスの定義 [入力4 -> 隠れ層1 (4) -> 隠れ層2 (3) -> 出力1]
W1 = np.array([
    [ 1.2, -0.4,  0.8, -1.5],
    [ 0.8, -0.2,  1.1, -1.2],
    [-0.5,  0.9, -1.0,  0.4],
    [-1.1,  0.3, -1.4,  1.8]
])
b1 = np.array([0.1, -0.2, 0.0, 0.3])

W2 = np.array([
    [ 1.4,  1.1, -0.8, -1.2],
    [-0.6, -0.5,  1.0,  0.7],
    [ 0.9,  0.8, -0.9, -1.1]
])
b2 = np.array([0.2, -0.1, 0.1])

W3 = np.array([1.8, -1.3, 1.2])
b3 = -0.5


def predict_credit_score(income: float, years: float, debt: float, delinquency: str):
    """
    入力パラメータから信用スコアおよび各層の活性化状態を計算する関数
    """
    x1 = income / 2000.0
    x2 = years / 40.0
    x3 = debt / 1000.0
    x4 = 1.0 if (delinquency == "あり" or delinquency == 1 or delinquency is True) else 0.0
    
    X = np.array([x1, x2, x3, x4])

    z1 = np.dot(X, W1.T) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2.T) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    output_score = sigmoid(z3)

    return {
        "inputs_normalized": X,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "score_pct": float(output_score * 100),
        "is_approved": bool(output_score >= 0.6)
    }


# ==========================================
# 2. HTML/JS シミュレーターUI（各ノード名上部配置版）
# ==========================================

HTML_SIMULATOR_CODE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: #0b0f19; color: #e2e8f0; font-family: sans-serif; display: flex; height: 100vh; overflow: hidden; }
        .sidebar { width: 300px; background-color: #111827; border-right: 1px solid #1f2937; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .sidebar h2 { font-size: 1rem; color: #f3f4f6; }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 0.8rem; color: #9ca3af; display: flex; justify-content: space-between; }
        input[type="range"] { width: 100%; accent-color: #3b82f6; }
        .radio-group { display: flex; gap: 16px; font-size: 0.85rem; }
        .main-content { flex: 1; display: flex; flex-direction: column; position: relative; }
        .header { padding: 16px 20px; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
        .score-card { background-color: #1f2937; padding: 8px 16px; border-radius: 6px; text-align: right; }
        .score-value { font-size: 1.5rem; font-weight: bold; color: #60a5fa; }
        .canvas-container { flex: 1; position: relative; }
        canvas { width: 100%; height: 100%; display: block; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>入力パラメータ</h2>
        <div class="control-group">
            <label>年収（万円） <span id="val-income">650</span></label>
            <input type="range" id="input-income" min="100" max="2000" value="650" step="10">
        </div>
        <div class="control-group">
            <label>勤続年数（年） <span id="val-years">7</span></label>
            <input type="range" id="input-years" min="0" max="40" value="7" step="1">
        </div>
        <div class="control-group">
            <label>他社借入額（万円） <span id="val-debt">120</span></label>
            <input type="range" id="input-debt" min="0" max="1000" value="120" step="10">
        </div>
        <div class="control-group">
            <label>過去の滞納歴</label>
            <div class="radio-group">
                <label><input type="radio" name="delinquency" value="0" checked> なし</label>
                <label><input type="radio" name="delinquency" value="1"> あり</label>
            </div>
        </div>
    </div>
    <div class="main-content">
        <div class="header">
            <div>
                <h1 style="font-size: 1.2rem;">DNN 信用スコア審査シミュレーター</h1>
                <p style="color: #6b7280; font-size: 0.8rem;">3Blue1Brown Neural Inspector スタイル</p>
            </div>
            <div class="score-card">
                <div style="font-size: 0.7rem; color: #9ca3af;">審査通過率</div>
                <div class="score-value" id="score-display">0.0%</div>
            </div>
        </div>
        <div class="canvas-container">
            <canvas id="networkCanvas"></canvas>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('networkCanvas');
        const ctx = canvas.getContext('2d');
        const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
        const W1 = [[1.2,-0.4,0.8,-1.5],[0.8,-0.2,1.1,-1.2],[-0.5,0.9,-1.0,0.4],[-1.1,0.3,-1.4,1.8]];
        const b1 = [0.1, -0.2, 0.0, 0.3];
        const W2 = [[1.4,1.1,-0.8,-1.2],[-0.6,-0.5,1.0,0.7],[0.9,0.8,-0.9,-1.1]];
        const b2 = [0.2, -0.1, 0.1];
        const W3 = [1.8, -1.3, 1.2];
        const b3 = -0.5;

        // ノード名ラベル定義
        const nodeLabels = [
            ["年収", "勤続年数", "他社借入", "返済遅延"],                 // 入力層
            ["収入安定度", "既存負担感", "信用信頼性", "属性リスク"],     // 隠れ層 1
            ["総合返済能力", "信用リスク", "資産余力"],                   // 隠れ層 2
            ["審査通過率"]                                              // 出力層
        ];

        let nodes = [], connections = [];
        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            setupNetwork();
        }
        function setupNetwork() {
            nodes = []; connections = [];
            const layers = [4, 4, 3, 1], layerX = [0.15, 0.40, 0.65, 0.90];
            layers.forEach((count, lIndex) => {
                const x = canvas.width * layerX[lIndex];
                for (let i = 0; i < count; i++) {
                    const y = (canvas.height / (count + 1)) * (i + 1);
                    nodes.push({ layer: lIndex, index: i, x, y, val: 0 });
                }
            });
            let nodeOffset = 0;
            for (let l = 0; l < layers.length - 1; l++) {
                const srcCount = layers[l], dstCount = layers[l + 1];
                const srcStart = nodeOffset, dstStart = nodeOffset + srcCount;
                for (let i = 0; i < srcCount; i++) {
                    for (let j = 0; j < dstCount; j++) {
                        let w = (l === 0) ? W1[j][i] : (l === 1) ? W2[j][i] : W3[i];
                        connections.push({ from: nodes[srcStart + i], to: nodes[dstStart + j], weight: w });
                    }
                }
                nodeOffset += srcCount;
            }
        }
        function updateCalculation() {
            const income = parseFloat(document.getElementById('input-income').value);
            const years = parseFloat(document.getElementById('input-years').value);
            const debt = parseFloat(document.getElementById('input-debt').value);
            const delinq = parseFloat(document.querySelector('input[name="delinquency"]:checked').value);

            document.getElementById('val-income').textContent = income;
            document.getElementById('val-years').textContent = years;
            document.getElementById('val-debt').textContent = debt;

            const X = [income / 2000.0, years / 40.0, debt / 1000.0, delinq];
            for (let i = 0; i < 4; i++) nodes[i].val = X[i];

            const a1 = [];
            for (let j = 0; j < 4; j++) {
                let z = b1[j];
                for (let i = 0; i < 4; i++) z += X[i] * W1[j][i];
                a1.push(sigmoid(z));
                nodes[4 + j].val = a1[j];
            }
            const a2 = [];
            for (let j = 0; j < 3; j++) {
                let z = b2[j];
                for (let i = 0; i < 4; i++) z += a1[i] * W2[j][i];
                a2.push(sigmoid(z));
                nodes[8 + j].val = a2[j];
            }
            let z3 = b3;
            for (let i = 0; i < 3; i++) z3 += a2[i] * W3[i];
            const output = sigmoid(z3);
            nodes[11].val = output;

            document.getElementById('score-display').textContent = (output * 100).toFixed(1) + '%';
        }

        let animOffset = 0;
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            animOffset = (animOffset + 0.5) % 20;

            // エッジ描画
            connections.forEach(conn => {
                ctx.beginPath();
                ctx.moveTo(conn.from.x, conn.from.y);
                ctx.lineTo(conn.to.x, conn.to.y);
                const isPositive = conn.weight > 0;
                const alpha = Math.min(Math.abs(conn.weight) * 0.5, 0.8);
                ctx.strokeStyle = isPositive ? `rgba(96, 165, 250, ${alpha})` : `rgba(244, 114, 182, ${alpha})`;
                ctx.lineWidth = Math.abs(conn.weight) * 2;
                ctx.setLineDash([6, 6]);
                ctx.lineDashOffset = -animOffset;
                ctx.stroke();
                ctx.setLineDash([]);
            });

            // ノードおよび上部ラベル描画
            nodes.forEach(node => {
                ctx.beginPath();
                ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
                const intensity = Math.floor(node.val * 255);
                ctx.fillStyle = `rgb(${Math.max(15, intensity - 40)}, ${intensity}, ${Math.min(255, intensity + 60)})`;
                ctx.fill();
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 2;
                ctx.stroke();

                // ノード内の数値表示
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(node.val.toFixed(2), node.x, node.y);

                // ノードの真上にラベル名称を描画
                const labelText = nodeLabels[node.layer][node.index];
                ctx.fillStyle = '#e2e8f0';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(labelText, node.x, node.y - 24);
            });
            requestAnimationFrame(draw);
        }

        window.addEventListener('resize', () => { resizeCanvas(); updateCalculation(); });
        document.querySelectorAll('input').forEach(input => input.addEventListener('input', updateCalculation));
        resizeCanvas(); updateCalculation(); draw();
    </script>
</body>
</html>
"""

def main():
    st.set_page_config(page_title="DNN Inspector", layout="wide")
    components.html(HTML_SIMULATOR_CODE, height=650, scrolling=False)


if __name__ == "__main__":
    main()