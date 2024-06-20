document.addEventListener('DOMContentLoaded', function () {
  const imageUpload = document.getElementById('imageUpload');
  const resultDiv = document.getElementById('result');

  imageUpload.addEventListener('change', async () => {
    resultDiv.innerHTML = '<p class="loading">Analyzing...</p>';
    const file = imageUpload.files[0];
    const reader = new FileReader();

    reader.onload = async (event) => {
      const imageData = event.target.result.split(',')[1];

      try {
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();
        if (data.error) {
          resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
        } else {
          resultDiv.innerHTML = data.final_decision;
        }
      } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
      }
    };

    reader.readAsDataURL(file);
  });
});
