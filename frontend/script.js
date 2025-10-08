document.getElementById("manual-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    gender: document.getElementById("gender").value,
    SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
    Partner: document.getElementById("Partner").value,
    Dependents: document.getElementById("Dependents").value,
    tenure: parseInt(document.getElementById("tenure").value),
    PhoneService: document.getElementById("PhoneService").value,
    MultipleLines: document.getElementById("MultipleLines").value,
    InternetService: document.getElementById("InternetService").value,
    OnlineSecurity: document.getElementById("OnlineSecurity").value,
    OnlineBackup: document.getElementById("OnlineBackup").value,
    DeviceProtection: document.getElementById("DeviceProtection").value,
    TechSupport: document.getElementById("TechSupport").value,
    StreamingTV: document.getElementById("StreamingTV").value,
    StreamingMovies: document.getElementById("StreamingMovies").value,
    Contract: document.getElementById("Contract").value,
    PaperlessBilling: document.getElementById("PaperlessBilling").value,
    PaymentMethod: document.getElementById("PaymentMethod").value,
    MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
    TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
  };

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(body)
  });
  const text = await res.text();
  document.getElementById("manual-result").innerHTML = text;
});

document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) return alert("Select CSV file.");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("http://127.0.0.1:8000/upload_csv", {
    method: "POST",
    body: formData
  });
  const data = await res.json();

  const csvResultDiv = document.getElementById("csv-result");
  if (data.download_link) {
    csvResultDiv.innerHTML = `
      ✅ ${data.message} |
      <a href="${data.download_link}" target="_blank">📥 Download Predictions</a>
    `;
  } else {
    csvResultDiv.innerHTML = `❌ ${data.error || "Error occurred"}`;
  }
});
