import { getAuthToken } from "./src/lib/auth";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:3001";
const USE_MOCK_API = String(import.meta.env.VITE_USE_MOCK_API || "false") === "true";

async function buildAuthHeaders(base = {}) {
  const token = await getAuthToken();
  if (!token) return base;
  return {
    ...base,
    Authorization: `Bearer ${token}`,
  };
}

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  try {
    const headers = await buildAuthHeaders();
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      headers,
      body: formData,
    });

    let data;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Backend did not return valid JSON", e);
      throw new Error("Invalid response from backend");
    }

    if (!res.ok || (data && data.error)) {
      console.error("Backend returned error", res.status, data);
      throw new Error(data?.error || `Backend error: ${res.status}`);
    }

    // Return the complete response with all detailed information
    return data;
  } catch (error) {
    if (USE_MOCK_API) {
      console.warn("Upload to backend failed, falling back to mock response:", error);
      return mockUploadResponse();
    }
    throw error;
  }
}

export async function fetchModelInfo() {
  try {
    const headers = await buildAuthHeaders();
    const res = await fetch(`${API_BASE}/api/model-info`, { headers });

    let data;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Backend did not return valid JSON for /api/model-info", e);
      throw new Error("Invalid response from backend");
    }

    if (!res.ok || (data && data.error)) {
      console.error("Backend returned error for /api/model-info", res.status, data);
      throw new Error(data?.error || `Backend error: ${res.status}`);
    }

    return data;
  } catch (error) {
    console.warn("Fetching model info failed:", error);
    return {
      status: "not_loaded",
      message: "Model information unavailable"
    };
  }
}

export async function fetchDetectionLogs(params = {}) {
  try {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      query.set(key, String(value));
    });

    const headers = await buildAuthHeaders();
    const url = query.toString() ? `${API_BASE}/api/logs?${query.toString()}` : `${API_BASE}/api/logs`;
    const res = await fetch(url, { headers });

    let data;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Backend did not return valid JSON for /api/logs", e);
      throw new Error("Invalid response from backend");
    }

    if (!res.ok || (data && data.error)) {
      console.error("Backend returned error for /api/logs", res.status, data);
      throw new Error(data?.error || `Backend error: ${res.status}`);
    }

    if (Array.isArray(data)) {
      return {
        items: data,
        total: data.length,
        page: 1,
        page_size: data.length,
      };
    }
    return data;
  } catch (error) {
    if (USE_MOCK_API) {
      console.warn("Fetching detection logs failed, falling back to mock logs:", error);
      const items = mockLogsResponse();
      return {
        items,
        total: items.length,
        page: 1,
        page_size: items.length,
      };
    }
    throw error;
  }
}

export async function deleteDetectionLog(logId) {
  const headers = await buildAuthHeaders();
  const res = await fetch(`${API_BASE}/api/logs/${encodeURIComponent(logId)}`, {
    method: "DELETE",
    headers,
  });
  const data = await res.json();
  if (!res.ok || data?.error) {
    throw new Error(data?.error || `Backend error: ${res.status}`);
  }
  return data;
}

export async function clearDetectionLogs(sourceType = "") {
  const headers = await buildAuthHeaders();
  const query = sourceType ? `?source_type=${encodeURIComponent(sourceType)}` : "";
  const res = await fetch(`${API_BASE}/api/logs${query}`, {
    method: "DELETE",
    headers,
  });
  const data = await res.json();
  if (!res.ok || data?.error) {
    throw new Error(data?.error || `Backend error: ${res.status}`);
  }
  return data;
}

export async function logLiveEvent(payload) {
  const headers = await buildAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/live-events`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload || {}),
  });
  const data = await res.json();
  if (!res.ok || data?.error) {
    throw new Error(data?.error || `Backend error: ${res.status}`);
  }
  return data;
}

export async function fetchStats() {
  try {
    const headers = await buildAuthHeaders();
    const res = await fetch(`${API_BASE}/api/stats`, { headers });
    let data;
    try {
      data = await res.json();
    } catch (e) {
      throw new Error("Invalid response from backend");
    }
    if (!res.ok || data?.error) {
      throw new Error(data?.error || `Backend error: ${res.status}`);
    }
    return data;
  } catch (error) {
    console.warn("Fetching stats failed, using empty defaults:", error);
    return null;
  }
}

function mockUploadResponse() {
  return new Promise((resolve) => {
    setTimeout(() => {
      const isFake = Math.random() > 0.5;
      const confidence = 70 + Math.random() * 30;
      
      resolve({
        prediction: isFake ? "Fake" : "Real",
        confidence: confidence,
        filename: "mock_upload.jpg",
        isVideo: false,
        threat_level: confidence > 80 ? "high" : confidence > 50 ? "medium" : "low",
        model_used: "Mock Model (Backend Unavailable)",
        processing_time: {
          preprocessing_ms: 10 + Math.random() * 20,
          inference_ms: 50 + Math.random() * 100,
          total_ms: 60 + Math.random() * 120
        },
        analysis: {
          level: "Mock",
          description: "Backend unavailable. Using mock prediction.",
          recommendation: "Please ensure backend server is running for accurate results."
        },
        model_info: {
          architecture: "N/A",
          input_size: "N/A",
          framework: "Mock",
          device: "cpu"
        }
      });
    }, 800);
  });
}

function mockLogsResponse() {
  const now = new Date();
  const entries = [];

  for (let i = 0; i < 12; i++) {
    const ts = new Date(now.getTime() - i * 5 * 60_000).toISOString();
    const isFake = Math.random() > 0.6;
    const confidence = 0.6 + Math.random() * 0.35;

    entries.push({
      timestamp: ts,
      filename: `mock_upload_${String(i + 1).padStart(3, "0")}.jpg`,
      prediction: isFake ? "Fake" : "Real",
      confidence,
    });
  }

  return entries;
}
