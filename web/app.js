async function loadModels() {
  const classificationModel = await tf.loadLayersModel('/models/tfjs_model/classification/model.json');
  const pricingModel = await tf.loadLayersModel('/models/tfjs_model/pricing/model.json');
  return { classificationModel, pricingModel };
}

const CATEGORIES = ['Cardboard', 'Food_Organics', 'Glass', 'Metal', 'Miscellaneous_Trash', 'Paper', 'Plastic', 'Textile_Trash', 'Vegetation'];
const PRICE_PER_KG = {
  'Cardboard': 2000, 'Food_Organics': 500, 'Glass': 1500, 'Metal': 8000,
  'Miscellaneous_Trash': 300, 'Paper': 2500, 'Plastic': 4000, 'Textile_Trash': 1000, 'Vegetation': 600
};

async function classifyImage() {
  const imageInput = document.getElementById('imageInput');
  const quantity = parseFloat(document.getElementById('quantity').value);
  const distance = parseFloat(document.getElementById('distance').value);
  const resultDiv = document.getElementById('result');
  const preview = document.getElementById('preview');

  if (!imageInput.files[0]) {
      resultDiv.innerHTML = 'Please select an image.';
      return;
  }

  const { classificationModel, pricingModel } = await loadModels();

  // Load and preprocess image
  const img = new Image();
  img.src = URL.createObjectURL(imageInput.files[0]);
  img.onload = async () => {
      preview.src = img.src;
      preview.style.display = 'block';

      const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(tf.scalar(255.0))
          .expandDims();

      // Classification
      const prediction = await classificationModel.predict(tensor).data();
      const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
      const predictedClass = CATEGORIES[predictedClassIndex];
      const confidence = prediction[predictedClassIndex] * 100;

      // Pricing
      const pricePerUnit = PRICE_PER_KG[predictedClass];
      const pricingInput = tf.tensor2d([[...Array(9).fill(0).map((_, i) => i === predictedClassIndex ? 1 : 0), quantity, distance, pricePerUnit]]);
      const totalPrice = (await pricingModel.predict(pricingInput).data())[0];

      // Display results
      resultDiv.innerHTML = `
          <h3>Classification Result</h3>
          <p><strong>Type:</strong> ${predictedClass}</p>
          <p><strong>Confidence:</strong> ${confidence.toFixed(2)}%</p>
          <h3>Pricing Result</h3>
          <p><strong>Price per kg:</strong> Rp ${pricePerUnit.toLocaleString()}</p>
          <p><strong>Quantity:</strong> ${quantity.toFixed(2)} kg</p>
          <p><strong>Distance:</strong> ${distance.toFixed(2)} km</p>
          <p><strong>Total Value:</strong> Rp ${totalPrice.toLocaleString(undefined, {minimumFractionDigits: 0})}</p>
      `;
  };
}