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


document.getElementById("assetForm").addEventListener("submit", async function (e) {
  e.preventDefault(); // stop page reload

  const lat = document.getElementById("latitude").value;
  const lon = document.getElementById("longitude").value;

  const payload = {
    type: document.getElementById("type").value,
    material: document.getElementById("material").value,
    soil_type: document.getElementById("soil_type").value,
    region: document.getElementById("region").value || null,
    exact_location:
      lat && lon ? [parseFloat(lat), parseFloat(lon)] : null,
    date_of_last_repair: formatDate(
      document.getElementById("last_repair").value
    ),
    snapshot_date: formatDate(
      document.getElementById("snapshot_date").value
    ),
    install_year: document.getElementById("install_year").value,
    length_m: document.getElementById("length_m").value
      ? parseFloat(document.getElementById("length_m").value)
      : null
  };

  // Validation: API requires region OR exact_location
  if (!payload.region && !payload.exact_location) {
    alert("Please provide either a region or an exact location.");
    return;
  }

  try {
    const API_BASE = window.location.hostname.includes("github.dev")
      ? "https://fictional-disco-976pvp495vj72x44p-8000.app.github.dev"
      : "http://127.0.0.1:8000";

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const result = await response.json();
    displayResults(result);

  } catch (error) {
    alert("Error connecting to TunnelVision API");
    console.error(error);
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