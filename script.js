document.getElementById('processBtn').addEventListener('click', async () => {
    const imgElement = document.getElementById('imageUpload').files[0];
    const quantity = 1; // Jumlah default
    const distance = 10; // Jarak default
    const shippingMethod = 'standard'; // Metode default
  
    if (imgElement) {
      const imgURL = URL.createObjectURL(imgElement);
      const img = new Image();
      img.src = imgURL;
  
      img.onload = async () => {
        const result = await processUserUpload(img, quantity, distance, shippingMethod);
        document.getElementById('results').innerHTML = `
          <h2>Results:</h2>
          <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
      };
    } else {
      alert('Please upload an image first.');
    }
  });
  
async function loadModel() {
    const model = await tf.loadLayersModel('tfjs_model/model.json');
    return model;
  }
  
  async function classifyImage(imgElement) {
    const model = await loadModel();
    
    // Pre-process gambar
    const tensor = tf.browser.fromPixels(imgElement)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(tf.scalar(255))
      .expandDims();
    
    // Prediksi
    const predictions = await model.predict(tensor);
    const results = Array.from(predictions.dataSync());
    
    // Mendapatkan kelas dengan probabilitas tertinggi
    const categories = ['Cardboard', 'Food_Organics', 'Glass', 'Metal', 
                        'Miscellaneous_Trash', 'Paper', 'Plastic', 
                        'Textile_Trash', 'Vegetation'];
    
    // Map hasil ke kategori dan probabilitas
    return results.map((prob, i) => {
      return {
        category: categories[i],
        probability: prob
      };
    }).sort((a, b) => b.probability - a.probability);
  }
  
  // Fungsi untuk menghitung estimasi harga berdasarkan klasifikasi
  function calculatePrice(classificationResults, quantity = 1) {
    const pricePerKg = {
      'Cardboard': 2000,
      'Food_Organics': 500,
      'Glass': 1500,
      'Metal': 8000,
      'Miscellaneous_Trash': 300,
      'Paper': 2500,
      'Plastic': 4000,
      'Textile_Trash': 1000,
      'Vegetation': 600
    };
    
    const topCategory = classificationResults[0].category;
    return pricePerKg[topCategory] * quantity;
  }
  
  // Fungsi untuk menghitung biaya pengiriman
  function calculateShippingCost(distance, weight, method = 'standard') {
    const baseCost = 10000;
    const costPerKm = 1000;
    const costPerKg = 500;
    
    let shippingCost = baseCost + (distance * costPerKm) + (weight * costPerKg);
    
    if (method === 'express') {
      shippingCost *= 1.5;
    } else if (method === 'economy') {
      shippingCost *= 0.8;
    }
    
    return Math.round(shippingCost);
  }
  
  // Fungsi utama untuk memproses gambar yang diunggah user
  async function processUserUpload(imgElement, quantity, distance, shippingMethod) {
    const classificationResults = await classifyImage(imgElement);
    const itemPrice = calculatePrice(classificationResults, quantity);
    const shippingCost = calculateShippingCost(distance, quantity, shippingMethod);
    const totalPrice = itemPrice - shippingCost;
    
    return {
      classification: classificationResults,
      pricing: {
        itemPrice,
        shippingCost,
        totalPrice
      }
    };
  }