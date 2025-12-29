const form = document.getElementById("irisForm");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("prediction");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = {
    sepal_length: parseFloat(document.getElementById("sepal_length").value),
    sepal_width: parseFloat(document.getElementById("sepal_width").value),
    petal_length: parseFloat(document.getElementById("petal_length").value),
    petal_width: parseFloat(document.getElementById("petal_width").value),
  };

  resultBox.classList.remove("hidden");
  predictionText.textContent = "Predicting...";

  try {
    const response = await fetch(
      "https://irish-endtoend-ml-pipeline.onrender.com/predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
      }
    );

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();

    const speciesMap = {
      0: "🌼 Iris-setosa",
      1: "🌺 Iris-versicolor",
      2: "🌸 Iris-virginica"
    };

    predictionText.textContent =
      speciesMap[result.species] || "Unknown species";

  } catch (error) {
    console.error(error);
    predictionText.textContent = "API connection failed";
  }
});
