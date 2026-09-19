// Runs the models off the main thread, now by calling the Python backend.
//
// The browser only uploads files and renders the JSON that comes back; all
// parsing, features and models live in one place (the server's door/acv/rail/
// shm.py), which is the same code that writes the submission CSVs.
//
// Receives { kind, files: [{ name, blob }] },
// posts { type: "progress" | "result" | "error" }.
const progress = (done, total, label) => postMessage({ type: "progress", done, total, label });

// One request per file: keeps each upload comfortably under Cloud Run's 32 MB
// request limit (a rail recording alone is ~17 MB), and lets progress step.
async function post(kind, file, extra = {}) {
  const body = new FormData();
  body.append("file", file.blob, file.name);
  for (const [k, v] of Object.entries(extra || {})) if (v) body.append(k, v);
  const res = await fetch(`/api/predict/${kind}`, { method: "POST", body });
  if (!res.ok) {
    let message = `Analysis failed (${res.status})`;
    try {
      const data = await res.json();
      if (data && data.detail) message = String(data.detail);
    } catch { /* keep the status message */ }
    throw new Error(message);
  }
  return res.json();
}

const handlers = {
  async Door(files) {
    progress(0, 1, "uploading stream");
    return post("door", files[0]);
  },

  async ACV(files) {
    progress(0, 1, "uploading workbook");
    return post("acv", files[0]);
  },

  async "Rail corrugation"(files) {
    const result = [];
    const bands = [];
    let bandLabels = [];
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, files.length > 1
        ? `analysing ${i + 1} of ${files.length}`
        : "analysing recording");
      const payload = await post("rail", files[i]);
      result.push(...payload.result);
      bands.push(...payload.bands);
      bandLabels = payload.bandLabels;
    }
    return { result, bands, bandLabels };
  },

  async SHM(files, data = {}) {
    const result = [];
    const detail = [];
    let fit = null;
    let state = null;
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, files.length > 1
        ? `analysing ${i + 1} of ${files.length}`
        : "rainflow counting");
      const payload = await post("shm", files[i], { stream: data.stream });
      result.push(...payload.result);
      detail.push(...payload.detail);
      fit = payload.fit;
      state = payload.state || state;
    }
    return { result, detail, fit, state };
  },
};

onmessage = async ({ data }) => {
  try {
    const handler = handlers[data.kind];
    if (!handler) throw new Error(`Unknown subsystem: ${data.kind}`);
    postMessage({ type: "result", kind: data.kind, payload: await handler(data.files, data) });
  } catch (err) {
    postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
  }
};
