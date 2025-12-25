function getLocation() {
  const locBtn = document.getElementById('locBtn');
  if (!navigator.geolocation) {
    alert('Geolocation is not supported by your browser.');
    return;
  }

  locBtn.classList.add('loading');
  navigator.geolocation.getCurrentPosition(
    (position) => {
      document.getElementById('latitude').value = position.coords.latitude.toFixed(6);
      document.getElementById('longitude').value = position.coords.longitude.toFixed(6);
      locBtn.classList.remove('loading');
      locBtn.textContent = 'Location set';
      setTimeout(() => { locBtn.textContent = 'Use My Current Location'; }, 1800);
    },
    () => {
      locBtn.classList.remove('loading');
      alert('Location access was denied or unavailable.');
    }
  );
}

function formatDate(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = dateStr.split('-');
  return `${m}-${d}-${y}`;
}

// Updated API base URL - now served from same origin
function getApiBase() {
  return '';  // Empty string = same origin (FastAPI serves everything)
}

function setLoading(isLoading) {
  const btn = document.getElementById('submitBtn');
  if (isLoading) {
    btn.classList.add('loading');
    btn.setAttribute('aria-busy','true');
    btn.querySelector('.btn-text').textContent = 'Analyzing...';
  } else {
    btn.classList.remove('loading');
    btn.setAttribute('aria-busy','false');
    btn.querySelector('.btn-text').textContent = 'Analyze';
  }
}

async function submitForm(e) {
  e.preventDefault();

  const lat = document.getElementById('latitude').value;
  const lon = document.getElementById('longitude').value;

  const payload = {
    type: document.getElementById('type').value,
    material: document.getElementById('material').value,
    soil_type: document.getElementById('soil_type').value,
    region: document.getElementById('region').value || null,
    exact_location: lat && lon ? [parseFloat(lat), parseFloat(lon)] : null,
    last_repair_date: formatDate(document.getElementById('last_repair').value),
    snapshot_date: formatDate(document.getElementById('snapshot_date').value),
    install_year: parseInt(document.getElementById('install_year').value),
    length_m: document.getElementById('length_m').value ? parseFloat(document.getElementById('length_m').value) : null
  };

  if (!payload.region && !payload.exact_location) {
    alert('Please provide either a region or an exact location.');
    return;
  }

  try {
    setLoading(true);
    const API_BASE = getApiBase();
    console.log('🔗 API Base URL:', API_BASE || 'same origin');

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Prediction failed: ${response.status} - ${text}`);
    }

    const result = await response.json();
    displayResults(result);
  } catch (error) {
    console.error(error);
    alert(`Error connecting to API: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

const assetFormEl = document.getElementById('assetForm');
if (assetFormEl) assetFormEl.addEventListener('submit', submitForm);

const resetBtn = document.getElementById('resetView');
if (resetBtn) {
  resetBtn.addEventListener('click', function(){
    const resultsCard = document.getElementById('resultsCard');
    if (resultsCard) {
      resultsCard.hidden = true;
      resultsCard.classList.remove('show');
    }
    const placeholder = document.getElementById('resultsPlaceholder');
    if (placeholder) placeholder.style.display = 'block';
  });
}

function animateRing(el, value){
  // value 0-100
  el.style.background = `conic-gradient(var(--neon) 0% ${value}%, rgba(255,255,255,0.06) ${value}% 100%)`;
  el.textContent = `${Math.round(value)}`;
}

function displayResults(data) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  const card = document.getElementById('resultsCard');
  document.getElementById('riskScore').textContent = data.risk_score;
  document.getElementById('priority').textContent = data.priority;
  document.getElementById('failureType').textContent = data.failure_type;
  document.getElementById('recommendedAction').textContent = data.recommended_action;

  const badge = document.getElementById('failureBadge');
  if (data.failure_in_30_days) {
    badge.textContent = 'High Risk';
    badge.style.background = 'linear-gradient(90deg,#7C3AED,#06B6D4)';
    badge.style.color = '#021918';
  } else {
    badge.textContent = 'Low Risk';
    badge.style.background = 'linear-gradient(90deg,#0f172a,#0e2931)';
    badge.style.color = '#7af0ff';
  }

  const ring = document.getElementById('scoreRing');
  animateRing(ring, Math.max(2, Math.min(100, data.risk_score)));

  card.hidden = false;
  setTimeout(()=>card.classList.add('show'), 40);
  // smooth scroll to results on small screens
  if (window.innerWidth < 900) card.scrollIntoView({behavior:'smooth'});
}

/* --- Page entrance & interaction enhancements --- */
// Add subtle intro pulse to CTA and enable small parallax on accent
window.addEventListener('DOMContentLoaded', function(){
  // Pulse CTA once
  const cta = document.querySelector('.primary');
  if(cta){
    cta.classList.add('pulse');
    setTimeout(()=>cta.classList.remove('pulse'), 2800);
  }

  // Add intro class to panel for staggered animation
  const panel = document.querySelector('.panel');
  const hero = document.querySelector('.hero');
  if(panel) panel.classList.add('intro');
  if(hero) hero.classList.add('intro');

  // Parallax for hero accent based on mouse move
  const main = document.querySelector('.main-grid');
  const accent = document.querySelector('.hero .accent');
  if(main && accent){
    main.addEventListener('mousemove', (e)=>{
      const rect = main.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5; // -0.5 .. 0.5
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      const tx = x * 18; const ty = y * 8; const rot = x * 6;
      accent.style.transform = `translateY(${ty}px) translateX(${tx}px) rotate(${16 + rot}deg)`;
      accent.style.filter = 'blur(28px)';
    });

    main.addEventListener('mouseleave', ()=>{accent.style.transform='rotate(16deg)';accent.style.filter='blur(32px)'});
  }
});
