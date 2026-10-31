const configuredApi = new URLSearchParams(window.location.search).get("api");
const API_BASE = configuredApi || localStorage.getItem("roadguard-api") || "http://127.0.0.1:8000";
const form = document.querySelector("#prediction-form");
const errorBox = document.querySelector("#error-message");
const submitButton = document.querySelector("#submit-button");
const apiStatus = document.querySelector("#api-status");
const apiPill = document.querySelector(".api-pill");
const resultTitle = document.querySelector("#result-title");
const resultSummary = document.querySelector("#result-summary");
const scoreValue = document.querySelector("#probability-value");
const scoreMeter = document.querySelector("#probability-meter");
const resultSymbol = document.querySelector(".result-symbol");

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/`, { method: "GET" });
    if (!response.ok) throw new Error("API unavailable");
    apiStatus.textContent = "Model API connected";
    apiPill.classList.add("connected");
    apiPill.classList.remove("offline");
  } catch {
    apiStatus.textContent = "Offline mode";
    apiPill.classList.add("offline");
    apiPill.classList.remove("connected");
  }
}

function updateRange(slider) {
  const output = slider.parentElement.querySelector(".range-value");
  if (!output) return;
  const value = Number(slider.value);
  if (slider.name === "speed_limit") output.textContent = `${value} km/h`;
  else if (slider.name === "curvature") output.textContent = value.toFixed(2);
  else output.textContent = String(value);
}

function clearResult() {
  errorBox.hidden = true;
  resultTitle.textContent = "Your accident-risk estimate will appear here.";
  resultSummary.textContent = "Complete the road indicators, then select “Estimate risk”.";
  scoreValue.textContent = "—";
  scoreMeter.style.width = "0%";
  resultSymbol.textContent = "↗";
  resultSymbol.style.color = "var(--accent)";
}

function collectFeatures() {
  const features = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") features[element.name] = element.checked ? 1 : 0;
    else if (element.tagName === "SELECT") features[element.name] = element.value;
    else features[element.name] = Number(element.value);
  }
  return features;
}

function offlineEstimate(features) {
  let score = 0.18;
  score += features.speed_limit / 120 * 0.2;
  score += features.curvature * 0.18;
  score += Math.min(features.num_reported_accidents, 20) / 20 * 0.22;
  score += features.num_lanes <= 1 ? 0.06 : 0;
  score += features.weather === "rainy" ? 0.08 : features.weather === "foggy" ? 0.12 : 0;
  score += features.lighting === "night" ? 0.1 : features.lighting === "dim" ? 0.05 : 0;
  score += features.time_of_day === "evening" ? 0.03 : 0;
  score += features.holiday ? 0.02 : 0;
  score -= features.road_signs_present ? 0.04 : 0.02;
  return Math.max(0, Math.min(1, score));
}

async function requestPrediction(features) {
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    });
    if (!response.ok) throw new Error("API rejected the request");
    const payload = await response.json();
    if (!Number.isFinite(Number(payload.accident_risk))) throw new Error("Invalid API response");
    return { score: Number(payload.accident_risk), source: "model" };
  } catch (error) {
    if (!(error instanceof TypeError) && error.message !== "Failed to fetch") throw error;
    return { score: offlineEstimate(features), source: "offline" };
  }
}

document.querySelectorAll('input[type="range"]').forEach((slider) => {
  slider.addEventListener("input", () => updateRange(slider));
  updateRange(slider);
});

document.querySelector("#randomize-inputs").addEventListener("click", () => {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const min = Number(slider.min);
    const step = Number(slider.step) || 1;
    slider.value = String(min + Math.floor(Math.random() * ((Number(slider.max) - min) / step + 1)) * step);
    updateRange(slider);
  });
  document.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = Math.random() >= 0.5; });
  document.querySelectorAll("select").forEach((select) => { select.selectedIndex = Math.floor(Math.random() * select.options.length); });
  clearResult();
});

document.querySelector("#reset-form").addEventListener("click", () => {
  form.reset();
  document.querySelectorAll('input[type="range"]').forEach(updateRange);
  clearResult();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submitButton.disabled = true;
  submitButton.innerHTML = "Calculating… <span>↻</span>";
  try {
    const result = await requestPrediction(collectFeatures());
    const score = Math.max(0, Math.min(1, result.score));
    const formatted = score.toFixed(4);
    scoreValue.textContent = formatted;
    scoreMeter.style.width = `${score * 100}%`;
    resultTitle.textContent = score >= 0.5 ? "Higher accident risk indicated." : "Lower accident risk indicated.";
    resultSummary.textContent = `${result.source === "model" ? "The trained model" : "Offline mode"} estimates an accident-risk score of ${formatted}. Review the conditions before making a road-safety decision.`;
    resultSymbol.textContent = score >= 0.5 ? "!" : "✓";
    resultSymbol.style.color = score >= 0.5 ? "var(--accent-dark)" : "var(--success)";
    document.querySelector("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = error.message || "Unable to calculate the estimate.";
    errorBox.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = "Estimate risk <span>↗</span>";
    await checkApi();
  }
});

checkApi();
