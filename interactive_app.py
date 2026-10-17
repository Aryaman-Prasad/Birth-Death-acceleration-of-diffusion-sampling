"""
Interactive Web Application for Project Testing and Evaluation.
Fulfills Deliverable 2:
'An interface where the user can input or upload the relevant project-specific inputs
to test the working of your developed mechanism/methodology in the project.'

Features:
- Pure Python backend (built on http.server, zero extra dependencies like gradio/flask needed).
- Interactive parameter controls: Dataset, Sampler (DDPM, DDIM-100, 50, 20, 10), Batch Size, Timesteps.
- Real-time generation, inference latency measurement, and metric calculation.
- Displays qualitative scatter plots, trajectory progression, and quantitative performance table.
"""

import argparse
import base64
import http.server
import io
import json
import os
import socketserver
import time
import urllib.parse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.toy_mlp import ToyScoreMLP
from src.diffusion.ddpm import GaussianDiffusion
from src.samplers.ddpm_sampler import DDPMSampler
from src.samplers.ddim_sampler import DDIMSampler
from src.diffusion.birth_death_sampler import BirthDeathDiffusionSampler
from src.metrics.evaluator import SamplingBenchmarkEvaluator
from src.utils.datasets import Synthetic2DDataset
from evaluate_baseline import get_mode_centers


class DiffusionService:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.models = {}

    def get_or_load_model(self, dataset: str, timesteps: int = 500) -> GaussianDiffusion:
        key = f"{dataset}_{timesteps}"
        if key in self.models:
            return self.models[key]

        score_model = ToyScoreMLP(input_dim=2, hidden_dim=128, num_layers=4, time_emb_dim=64).to(self.device)
        diffusion = GaussianDiffusion(model=score_model, timesteps=timesteps, beta_schedule="cosine").to(self.device)

        ckpt_path = os.path.join(self.checkpoint_dir, f"ddpm_{dataset}.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            score_model.load_state_dict(checkpoint.get("ema_state", checkpoint.get("model_state")))
        else:
            # Fast on-the-fly training
            print(f"[*] Training fast model for {dataset}...")
            from train import train_2d
            args = argparse.Namespace(
                dataset=dataset, epochs=15, batch_size=512, lr=2e-3,
                timesteps=timesteps, schedule="cosine", hidden_dim=128,
                num_layers=4, num_samples=25000, device=self.device.type,
                seed=42, log_interval=10, eval_interval=100,
                checkpoint_dir=self.checkpoint_dir, results_dir="./results"
            )
            train_2d(args)
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            score_model.load_state_dict(checkpoint.get("ema_state", checkpoint.get("model_state")))

        score_model.eval()
        self.models[key] = diffusion
        return diffusion

    def generate_and_evaluate(self, dataset: str, sampler_name: str, steps: int, num_samples: int, eta: float = 0.0):
        timesteps = 500
        diffusion = self.get_or_load_model(dataset, timesteps=timesteps)

        shape = (num_samples, 2)
        if sampler_name == "DDPM":
            sampler = DDPMSampler(diffusion)
            samples, info = sampler.sample(shape=shape, return_intermediates=True)
        elif sampler_name == "BDLS":
            sampler = BirthDeathDiffusionSampler(
                diffusion,
                birth_rate=0.8,
                langevin_steps=1,
                langevin_lr=0.03,
                exploration_noise=0.03,
                diversity_weight=0.25
            )
            samples, info = sampler.sample(shape=shape, num_inference_steps=steps, eta=eta, return_intermediates=True)
        else:
            sampler = DDIMSampler(diffusion)
            samples, info = sampler.sample(shape=shape, num_inference_steps=steps, eta=eta, return_intermediates=True)

        gt_dataset = Synthetic2DDataset(name=dataset, num_samples=num_samples, seed=999)
        mode_centers = get_mode_centers(dataset)
        evaluator = SamplingBenchmarkEvaluator(ground_truth_samples=gt_dataset.data, mode_centers=mode_centers)

        metrics = evaluator.evaluate_sampler(
            sampler_name=f"{sampler_name}-{steps}" if sampler_name != "DDPM" else f"DDPM-{timesteps}",
            generated_samples=samples,
            info=info
        )

        # Plot 1: Comparison Scatter
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        gt_np = gt_dataset.data.numpy()
        ax1.scatter(gt_np[:, 0], gt_np[:, 1], s=3, alpha=0.4, color="#1f77b4")
        ax1.set_title(f"Target Distribution ({dataset})", fontsize=11, fontweight="bold")
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.grid(True, linestyle="--", alpha=0.5)

        s_np = samples.cpu().numpy()
        ax2.scatter(s_np[:, 0], s_np[:, 1], s=3, alpha=0.5, color="#e6550d")
        ax2.set_title(f"Generated Samples ({metrics['sampler']})", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        scatter_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Plot 2: Trajectory Progression
        intermediates = info["intermediates"]
        n_stages = 5
        indices = [int(i) for i in torch.linspace(0, len(intermediates) - 1, n_stages)]
        fig, axes = plt.subplots(1, n_stages, figsize=(12, 2.5), sharex=True, sharey=True)
        for i, idx in enumerate(indices):
            step_pts = intermediates[idx].numpy()
            axes[i].scatter(step_pts[:, 0], step_pts[:, 1], s=2, alpha=0.4, color="#3b528b")
            axes[i].set_title(f"Step {idx}", fontsize=9)
            axes[i].set_xlim(-3, 3)
            axes[i].set_ylim(-3, 3)
            axes[i].grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        buf_traj = io.BytesIO()
        plt.savefig(buf_traj, format="png", dpi=150)
        plt.close()
        traj_b64 = base64.b64encode(buf_traj.getvalue()).decode("utf-8")

        return {
            "metrics": metrics,
            "scatter_img": f"data:image/png;base64,{scatter_b64}",
            "traj_img": f"data:image/png;base64,{traj_b64}"
        }


SERVICE = DiffusionService()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diffusion Sampling Acceleration - Interactive Interface</title>
    <style>
        :root { --primary: #1a73e8; --bg: #f8f9fa; --card: #ffffff; --text: #202124; --border: #dadce0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .header h1 { margin: 0 0 8px 0; font-size: 24px; }
        .header p { margin: 0; opacity: 0.9; font-size: 14px; }
        .grid { display: grid; grid-template-columns: 320px 1fr; gap: 24px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .card h2 { font-size: 16px; margin: 0 0 16px 0; border-bottom: 1px solid var(--border); padding-bottom: 8px; color: #1e3c72; }
        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
        select, input { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; box-sizing: border-box; }
        button { width: 100%; background: var(--primary); color: white; border: none; padding: 12px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #1557b0; }
        .metrics-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .metrics-table th, .metrics-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
        .metrics-table th { background: #f1f3f4; font-weight: 600; }
        .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; background: #e8f0fe; color: #1a73e8; }
        .img-container { margin-top: 16px; text-align: center; }
        .img-container img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }
        .spinner { display: none; text-align: center; padding: 20px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Birth-Death Process Diffusion Sampling Acceleration</h1>
        <p>IE 643 Course Project | Interactive Testing & Benchmark Interface (Deliverable 2)</p>
    </div>
    <div class="grid">
        <div class="card">
            <h2>Sampling Configuration</h2>
            <div class="form-group">
                <label>Target Distribution</label>
                <select id="dataset">
                    <option value="8gaussians" selected>8-Gaussians Ring</option>
                    <option value="25gaussians">25-Gaussians Grid</option>
                    <option value="swissroll">Swiss Roll</option>
                    <option value="moons">Two Moons</option>
                </select>
            </div>
            <div class="form-group">
                <label>Sampling Framework</label>
                <select id="sampler" onchange="updateSteps()">
                    <option value="BDLS" selected>Birth-Death Accelerated (BDLS-Diffusion - Proposed)</option>
                    <option value="DDIM">Accelerated DDIM (Implicit ODE)</option>
                    <option value="DDPM">Standard DDPM (Baseline - 500 steps)</option>
                </select>
            </div>
            <div class="form-group" id="steps-group">
                <label>Inference Steps (NFE)</label>
                <select id="steps">
                    <option value="100">100 Steps</option>
                    <option value="50" selected>50 Steps (10x Speedup)</option>
                    <option value="20">20 Steps (25x Speedup)</option>
                    <option value="10">10 Steps (50x Speedup)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Evaluation Sample Size</label>
                <input type="number" id="num_samples" value="1500" min="200" max="5000" step="100">
            </div>
            <button id="run-btn" onclick="runSampling()">Run Sampling & Benchmark</button>
            <div id="spinner" class="spinner">Generating samples & evaluating metrics...</div>
        </div>

        <div class="card">
            <h2>Benchmarking & Qualitative Results</h2>
            <div id="results-area">
                <p style="color: #666; font-size: 13px;">Click "Run Sampling & Benchmark" to generate samples and observe metrics in real time.</p>
            </div>
        </div>
    </div>
</div>

<script>
function updateSteps() {
    const sampler = document.getElementById("sampler").value;
    document.getElementById("steps-group").style.display = sampler === "DDPM" ? "none" : "block";
}

async function runSampling() {
    const btn = document.getElementById("run-btn");
    const spinner = document.getElementById("spinner");
    const resultsArea = document.getElementById("results-area");
    
    btn.disabled = true;
    spinner.style.display = "block";

    const payload = {
        dataset: document.getElementById("dataset").value,
        sampler: document.getElementById("sampler").value,
        steps: parseInt(document.getElementById("steps").value),
        num_samples: parseInt(document.getElementById("num_samples").value)
    };

    try {
        const response = await fetch("/api/sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        const m = data.metrics;

        resultsArea.innerHTML = `
            <table class="metrics-table">
                <tr><th>Metric</th><th>Measured Value</th><th>Description</th></tr>
                <tr><td><b>Sampler</b></td><td><span class="tag">${m.sampler}</span></td><td>Sampling algorithm</td></tr>
                <tr><td><b>NFEs</b></td><td><b>${m.nfe}</b></td><td>Neural network evaluations</td></tr>
                <tr><td><b>Inference Latency</b></td><td><b>${m.inference_time_sec} s</b></td><td>Total wall-clock generation time</td></tr>
                <tr><td><b>Throughput</b></td><td>${m.throughput_samples_per_sec} samples/s</td><td>Generation speed</td></tr>
                <tr><td><b>Sliced Wasserstein</b></td><td>${(m.sliced_wasserstein * 100).toFixed(3)} x 10⁻²</td><td>Optimal transport divergence (lower is better)</td></tr>
                <tr><td><b>Multi-scale RBF MMD</b></td><td>${(m.mmd * 1000).toFixed(3)} x 10⁻³</td><td>Maximum Mean Discrepancy</td></tr>
                ${m.mode_coverage_pct ? `<tr><td><b>Mode Coverage</b></td><td><b>${m.mode_coverage_pct}%</b> (${m.modes_covered})</td><td>Percentage of target modes recovered</td></tr>` : ""}
            </table>
            <div class="img-container">
                <h3 style="font-size: 13px; text-align: left; margin: 16px 0 8px 0;">Distribution Comparison</h3>
                <img src="${data.scatter_img}" alt="Distribution Comparison">
            </div>
            <div class="img-container">
                <h3 style="font-size: 13px; text-align: left; margin: 16px 0 8px 0;">Reverse Denoising Trajectory Evolution (x_T → x_0)</h3>
                <img src="${data.traj_img}" alt="Trajectory Evolution">
            </div>
        `;
    } catch (err) {
        resultsArea.innerHTML = `<p style="color: red;">Error running sampling: ${err.message}</p>`;
    } finally {
        btn.disabled = false;
        spinner.style.display = "none";
    }
}
</script>
</body>
</html>
"""


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/sample":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            dataset = req.get("dataset", "8gaussians")
            sampler = req.get("sampler", "DDIM")
            steps = req.get("steps", 50)
            num_samples = req.get("num_samples", 1500)

            print(f"[API] Running sampling: {sampler} on {dataset} ({steps} steps, {num_samples} samples)")
            result = SERVICE.generate_and_evaluate(dataset, sampler, steps, num_samples)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def main():
    parser = argparse.ArgumentParser(description="Run Interactive Sampling UI Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve interactive UI")
    args = parser.parse_args()

    server_address = ("", args.port)
    with socketserver.TCPServer(server_address, RequestHandler) as httpd:
        print("=" * 70)
        print(f"[*] Interactive Interface Server running at: http://localhost:{args.port}/")
        print("[*] Open the URL in your browser to interact with the diffusion sampler!")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Shutting down server.")


if __name__ == "__main__":
    main()

