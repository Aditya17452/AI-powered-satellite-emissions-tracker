function loadPrediction() {

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            lat: 25.32,
            lon: 82.98
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("output").textContent =
            JSON.stringify(data, null, 2);
    })
    .catch(err => {
        document.getElementById("output").textContent =
            "Error fetching prediction";
        console.error(err);
    });
}
