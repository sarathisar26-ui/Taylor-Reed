import express from 'express';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import fs from 'fs';

const app = express();
const PORT = 3000;
const PYTHON_PORT = 5000;
const PYTHON_BASE = `http://127.0.0.1:${PYTHON_PORT}`;

app.use(express.json());

// Spawn Python FastAPI backend on internal port 5000
let pythonProcess: ChildProcess | null = null;

function startPythonBackend() {
  const scriptPath = path.join(process.cwd(), 'main.py');
  if (!fs.existsSync(scriptPath)) {
    console.warn('[SERVER] main.py not found, skipping python spawn.');
    return;
  }

  console.log(`[SERVER] Spawning Python FastAPI server on port ${PYTHON_PORT}...`);
  pythonProcess = spawn('python3', [scriptPath, String(PYTHON_PORT)], {
    cwd: process.cwd(),
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'inherit', 'inherit']
  });

  pythonProcess.on('error', (err) => {
    console.error('[SERVER] Failed to start python backend:', err);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.warn(`[SERVER] Python backend exited with code ${code}, signal ${signal}. Restarting in 2s...`);
    setTimeout(startPythonBackend, 2000);
  });
}

startPythonBackend();

// Wait for Python backend readiness helper
async function waitForPython(retries = 15, delayMs = 400): Promise<boolean> {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(`${PYTHON_BASE}/health`);
      if (res.ok) return true;
    } catch {
      await new Promise(r => setTimeout(r, delayMs));
    }
  }
  return false;
}

// ----------------- PROXY API ENDPOINTS -----------------

// Health check
app.get('/health', async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_BASE}/health`);
    if (pyRes.ok) {
      const data = await pyRes.json();
      return res.json(data);
    }
  } catch {
    // Fallback if still booting
  }
  return res.json({
    status: 'ok',
    model_loaded: fs.existsSync(path.join(process.cwd(), 'stress_model.pkl')),
    scaler_loaded: fs.existsSync(path.join(process.cwd(), 'scaler.pkl')),
    excel_file_exists: fs.existsSync(path.join(process.cwd(), 'student_data.xlsx')),
    gateway: 'express-proxy'
  });
});

// Predict endpoint
app.post('/predict', async (req, res) => {
  try {
    await waitForPython(10, 300);
    const pyRes = await fetch(`${PYTHON_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });

    const status = pyRes.status;
    const data = await pyRes.json();
    return res.status(status).json(data);
  } catch (err: any) {
    console.error('[SERVER] Prediction proxy error:', err);
    return res.status(500).json({ detail: `Backend service error: ${err.message}` });
  }
});

// History endpoint
app.get('/history', async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_BASE}/history`);
    if (pyRes.ok) {
      const data = await pyRes.json();
      return res.json(data);
    }
  } catch (err: any) {
    console.warn('[SERVER] History proxy error:', err);
  }
  return res.json({ records: [], total_count: 0 });
});

// Analytics endpoint
app.get('/analytics', async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_BASE}/analytics`);
    if (pyRes.ok) {
      const data = await pyRes.json();
      return res.json(data);
    }
  } catch (err: any) {
    console.warn('[SERVER] Analytics proxy error:', err);
  }
  return res.json({
    total_students: 0,
    distribution: { Low: 0, Medium: 0, High: 0 },
    averages: { sleep: 0, study: 0, screen: 0, physical: 0, gpa: 0 }
  });
});

// Download Excel file
app.get('/download-excel', (req, res) => {
  const excelPath = path.join(process.cwd(), 'student_data.xlsx');
  if (!fs.existsSync(excelPath)) {
    return res.status(404).send('Excel file not found.');
  }
  res.download(excelPath, 'student_data.xlsx');
});

// Serve frontend style.css and index.html
app.get('/style.css', (req, res) => {
  res.sendFile(path.join(process.cwd(), 'style.css'));
});

app.get('/', (req, res) => {
  res.sendFile(path.join(process.cwd(), 'index.html'));
});

// Fallback for SPA or other static assets
app.use(express.static(process.cwd()));
app.get('*', (req, res) => {
  res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[SERVER] Full-stack student stress predictor running at http://0.0.0.0:${PORT}`);
});
