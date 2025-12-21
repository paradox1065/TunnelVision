function getLocation() {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (position) => {
      document.getElementById("latitude").value =
        position.coords.latitude.toFixed(6);
      document.getElementById("longitude").value =
        position.coords.longitude.toFixed(6);
    },
    () => {
      alert("Location access was denied or unavailable.");
    }
  );
}

function formatDate(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = dateStr.split("-");
  return `${m}-${d}-${y}`;
}

// Dynamically determine API base URL
function getApiBase() {
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  
  // Check if running in GitHub Codespaces
  if (hostname.includes('app.github.dev') || hostname.includes('github.dev')) {
    // Replace 5500 with 8000 in the hostname
    const apiHostname = hostname.replace('5500', '8000');
    return `${protocol}//${apiHostname}`;
  }
  
  // Local development
  return 'http://127.0.0.1:8000';
}

document.getElementById("assetForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const lat = document.getElementById("latitude").value;
  const lon = document.getElementById("longitude").value;

  const payload = {
    type: document.getElementById("type").value,
    material: document.getElementById("material").value,
    soil_type: document.getElementById("soil_type").value,
    region: document.getElementById("region").value || null,
    exact_location:
      lat && lon ? [parseFloat(lat), parseFloat(lon)] : null,
    last_repair_date: formatDate(document.getElementById("last_repair").value),
    snapshot_date: formatDate(document.getElementById("snapshot_date").value),
    install_year: parseInt(document.getElementById("install_year").value),
    length_m: document.getElementById("length_m").value
      ? parseFloat(document.getElementById("length_m").value)
      : null
  };

  if (!payload.region && !payload.exact_location) {
    alert("Please provide either a region or an exact location.");
    return;
  }

  try {
    const API_BASE = getApiBase();
    
    console.log("🔗 API Base URL:", API_BASE);
    console.log("📤 Sending payload:", payload);

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    console.log("📥 Response status:", response.status);

    if (!response.ok) {
      const text = await response.text();
      console.error("❌ Error response:", text);
      throw new Error(`Prediction failed: ${response.status} - ${text}`);
    }

    const result = await response.json();
    console.log("✅ Success:", result);
    displayResults(result);

  } catch (error) {
    console.error("❌ Full error:", error);
    alert(`Error connecting to API: ${error.message}\n\nCheck console for details.`);
  }
});

function displayResults(data) {
  const outputDiv = document.getElementById("results");
  outputDiv.innerHTML = `
    <h3>🔍 TunnelVision Prediction</h3>
    <p>Failure in 30 Days: <b>${data.failure_in_30_days ? "Yes" : "No"}</b></p>
    <p>Failure Type: ${data.failure_type}</p>
    <p>Risk Score: ${data.risk_score}/100</p>
    <p>Priority Level: ${data.priority}</p>
    <p>Recommended Action: ${data.recommended_action}</p>
  `;
}