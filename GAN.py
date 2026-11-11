import numpy as np

# ==========================================
# 1. 共通計算ロジック（他ファイルから import して使う関数）
# ==========================================

def sigmoid(x):
    """活性化関数 (Sigmoid)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# 簡易 Generator (ノイズ2D + 調整パラメータ -> 生成データ4D)
W_gen = np.array([
    [ 1.8, -0.9],
    [-0.7,  1.6],
    [ 1.2,  1.1],
    [-1.4, -1.3]
])
b_gen = np.array([0.1, -0.2, 0.3, -0.1])

# 簡易 Discriminator (データ4D -> 本物確率 1D)
W_disc = np.array([1.5, -1.2, 1.4, -1.6])
b_disc = -0.2


def generate_fake_data(latent_noise_2d, generator_skill=0.5):
    """
    潜在ノイズ(2次元)と生成器の熟練度(0.0~1.0)から
    4次元の偽物データ(Fake Data)を生成する関数
    """
    z = np.array(latent_noise_2d, dtype=float)
    raw_fake = sigmoid(np.dot(W_gen, z) + b_gen)
    
    # 理想的な本物データ例 [0.8, 0.2, 0.8, 0.2] に熟練度に応じて近づける
    real_target = np.array([0.8, 0.2, 0.8, 0.2])
    fake_data = raw_fake * (1.0 - generator_skill) + real_target * generator_skill

    return fake_data


def evaluate_discriminator(data_4d):
    """
    入力されたデータ(4次元)が「本物」か「偽物」かを識別器が判定する関数
    (0.0 = 確実に偽物, 1.0 = 確実に本物)
    """
    x = np.array(data_4d, dtype=float)
    logit = np.dot(W_disc, x) + b_disc
    real_probability = float(sigmoid(logit))

    return {
        "input_data": x,
        "real_probability": real_probability,
        "is_real_accepted": bool(real_probability >= 0.5)
    }


def process_gan_pipeline(latent_noise_2d, generator_skill=0.5):
    """
    GAN全体の処理パイプライン（ノイズ入力 -> 画像生成 -> 識別器による審議）
    """
    fake_data = generate_fake_data(latent_noise_2d, generator_skill)
    disc_result = evaluate_discriminator(fake_data)

    return {
        "latent_noise": np.array(latent_noise_2d),
        "generated_fake": fake_data,
        "real_probability": disc_result["real_probability"],
        "fooled_discriminator": disc_result["is_real_accepted"]
    }


# ==========================================
# 2. HTML/JS GAN インスペクターUI コード
# ==========================================

HTML_GAN_SIMULATOR_CODE = """
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
        button.preset-btn { background-color: #1f2937; border: 1px solid #374151; color: #f472b6; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: bold; }
        button.preset-btn:hover { background-color: #374151; }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 0.8rem; color: #9ca3af; display: flex; justify-content: space-between; }
        input[type="range"] { width: 100%; accent-color: #ec4899; }
        
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
        <h2>GAN パラメータ調整</h2>
        <p style="font-size: 0.75rem; color: #9ca3af;">潜在ノイズと生成器の学習レベルを変えて、識別器を騙せるかチャレンジします</p>
        
        <div class="preset-btn-group">
            <button class="preset-btn" onclick="setPreset(0.1, 0.2, 0.1)">初期状態 (雑な偽物)</button>
            <button class="preset-btn" onclick="setPreset(0.5, 0.5, 0.6)">学習途中 (精巧化)</button>
            <button class="preset-btn" onclick="setPreset(0.8, -0.3, 0.95)">完成状態 (見分け不能)</button>
            <button class="preset-btn" onclick="setPreset(0.0, 0.0, 0.0)">ランダムノイズ</button>
        </div>

        <div class="control-group" style="margin-top: 10px;">
            <label>潜在ノイズ z1 <span id="val-z1">0.20</span></label>
            <input type="range" id="input-z1" min="-1" max="1" value="0.2" step="0.05">
        </div>
        <div class="control-group">
            <label>潜在ノイズ z2 <span id="val-z2">0.50</span></label>
            <input type="range" id="input-z2" min="-1" max="1" value="0.5" step="0.05">
        </div>
        <div class="control-group">
            <label>Generator (生成器) 熟練度 <span id="val-skill">50%</span></label>
            <input type="range" id="input-skill" min="0" max="1" value="0.5" step="0.05">
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <div>
                <h1 style="font-size: 1.2rem;">GAN (敵対的生成ネットワーク) シミュレーター</h1>
                <p style="color: #6b7280; font-size: 0.8rem;">[Generator 生成器] が作ったデータ ➔ [Discriminator 識別器] が本物か見破るバトル</p>
            </div>
            <div class="score-card">
                <div style="font-size: 0.7rem; color: #9ca3af;">識別器による「本物判定確率」</div>
                <div class="score-value" id="pred-display">0.0%</div>
            </div>
        </div>
        <div class="canvas-container">
            <canvas id="ganCanvas"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('ganCanvas');
        const ctx = canvas.getContext('2d');

        const W_gen = [[1.8,-0.9],[-0.7,1.6],[1.2,1.1],[-1.4,-1.3]];
        const b_gen = [0.1, -0.2, 0.3, -0.1];
        const W_disc = [1.5, -1.2, 1.4, -1.6];
        const b_disc = -0.2;
        const realTarget = [0.8, 0.2, 0.8, 0.2];

        function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }

        function setPreset(z1, z2, skill) {
            document.getElementById('input-z1').value = z1;
            document.getElementById('input-z2').value = z2;
            document.getElementById('input-skill').value = skill;
            updateAndDraw();
        }

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }

        function calculateGAN() {
            const z1 = parseFloat(document.getElementById('input-z1').value);
            const z2 = parseFloat(document.getElementById('input-z2').value);
            const skill = parseFloat(document.getElementById('input-skill').value);

            document.getElementById('val-z1').textContent = z1.toFixed(2);
            document.getElementById('val-z2').textContent = z2.toFixed(2);
            document.getElementById('val-skill').textContent = Math.round(skill * 100) + '%';

            // 1. Generator 偽データ生成
            const fakeData = [];
            for(let i=0; i<4; i++) {
                let raw = sigmoid(z1 * W_gen[i][0] + z2 * W_gen[i][1] + b_gen[i]);
                fakeData.push(raw * (1.0 - skill) + realTarget[i] * skill);
            }

            // 2. Discriminator 判定
            let logit = b_disc;
            for(let i=0; i<4; i++) logit += fakeData[i] * W_disc[i];
            const realProb = sigmoid(logit);

            document.getElementById('pred-display').textContent = (realProb * 100).toFixed(1) + '%';

            return { z: [z1, z2], fakeData, realProb };
        }

        let animOffset = 0;
        function updateAndDraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            animOffset = (animOffset + 0.4) % 20;

            const { z, fakeData, realProb } = calculateGAN();

            const layerX = [canvas.width * 0.15, canvas.width * 0.42, canvas.width * 0.72, canvas.width * 0.90];

            // ヘッダータイトル
            ctx.fillStyle = '#9ca3af';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText("潜在ノイズ (Z)", layerX[0], 35);
            ctx.fillText("生成器 (Generator) の出力", layerX[1], 35);
            ctx.fillText("識別器 (Discriminator)", layerX[2], 35);
            ctx.fillText("最終判定", layerX[3], 35);

            // 1. ノード -> 生成データの線
            ctx.setLineDash([4, 4]);
            ctx.lineDashOffset = -animOffset;
            for(let i=0; i<2; i++) {
                const zY = (canvas.height / 3) * (i + 1);
                for(let j=0; j<4; j++) {
                    const fY = (canvas.height / 5) * (j + 1);
                    ctx.beginPath();
                    ctx.moveTo(layerX[0] + 16, zY);
                    ctx.lineTo(layerX[1] - 16, fY);
                    ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            }

            // 2. 生成データ -> 識別器の線
            for(let j=0; j<4; j++) {
                const fY = (canvas.height / 5) * (j + 1);
                ctx.beginPath();
                ctx.moveTo(layerX[1] + 16, fY);
                ctx.lineTo(layerX[2] - 18, canvas.height * 0.5);
                ctx.strokeStyle = 'rgba(96, 165, 250, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            ctx.setLineDash([]);

            // 潜在ノード Z
            for(let i=0; i<2; i++) {
                const y = (canvas.height / 3) * (i + 1);
                ctx.beginPath();
                ctx.arc(layerX[0], y, 16, 0, Math.PI * 2);
                ctx.fillStyle = '#831843';
                ctx.fill();
                ctx.strokeStyle = '#f472b6';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(z[i].toFixed(2), layerX[0], y);

                ctx.fillStyle = '#f472b6';
                ctx.font = '11px sans-serif';
                ctx.fillText(`z_${i+1}`, layerX[0] - 22, y);
            }

            // 生成データノード (偽物データ)
            for(let i=0; i<4; i++) {
                const y = (canvas.height / 5) * (i + 1);
                const val = fakeData[i];
                ctx.beginPath();
                ctx.arc(layerX[1], y, 16, 0, Math.PI * 2);
                const c = Math.floor(val * 255);
                ctx.fillStyle = `rgb(${c}, 50, ${255 - c})`;
                ctx.fill();
                ctx.strokeStyle = '#a855f7';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(2), layerX[1], y);

                ctx.fillStyle = '#e2e8f0';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`生成画像データ ${i+1}`, layerX[1] - 22, y);
            }

            // Discriminator ノード
            const discY = canvas.height * 0.5;
            ctx.beginPath();
            ctx.arc(layerX[2], discY, 20, 0, Math.PI * 2);
            ctx.fillStyle = '#1e3a8a';
            ctx.fill();
            ctx.strokeStyle = '#60a5fa';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText("D", layerX[2], discY);

            // 識別器 -> 最終出力の線
            ctx.beginPath();
            ctx.moveTo(layerX[2] + 20, discY);
            ctx.lineTo(layerX[3] - 20, discY);
            ctx.strokeStyle = realProb >= 0.5 ? '#34d399' : '#ef4444';
            ctx.lineWidth = 3;
            ctx.stroke();

            // 最終判定ノード (本物 / 偽物)
            const isAccepted = realProb >= 0.5;
            ctx.beginPath();
            ctx.arc(layerX[3], discY, 20, 0, Math.PI * 2);
            ctx.fillStyle = isAccepted ? '#065f46' : '#991b1b';
            ctx.fill();
            ctx.strokeStyle = isAccepted ? '#34d399' : '#f87171';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(isAccepted ? "本物" : "偽物", layerX[3], discY);

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