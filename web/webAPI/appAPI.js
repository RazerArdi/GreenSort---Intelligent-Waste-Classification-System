document.addEventListener('DOMContentLoaded', () => {
    // Constants
    const CATEGORIES = [
        'Cardboard', 'Food_Organics', 'Glass', 'Metal',
        'Miscellaneous_Trash', 'Paper', 'Plastic',
        'Textile_Trash', 'Vegetation'
    ];

    const PRICE_PER_KG = {
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

    const COLOR_INDICATORS = {
        'Cardboard': '#cd853f',
        'Food_Organics': '#8fbc8f',
        'Glass': '#add8e6',
        'Metal': '#c0c0c0',
        'Miscellaneous_Trash': '#808080',
        'Paper': '#f5f5dc',
        'Plastic': '#87cefa',
        'Textile_Trash': '#db7093',
        'Vegetation': '#32cd32'
    };

    const CATEGORY_MAPPING = {
        'Cardboard': 'Kardus',
        'Food_Organics': 'Bahan Organik Makanan',
        'Glass': 'Kaca',
        'Metal': 'Logam',
        'Miscellaneous_Trash': 'Sampah Lainnya',
        'Paper': 'Kertas',
        'Plastic': 'Plastik',
        'Textile_Trash': 'Sampah Tekstil',
        'Vegetation': 'Vegetasi'
    };

    const API_ENDPOINTS = {
        classification: {
            url: 'https://greensort-rrqya.eastus2.inference.ml.azure.com/score',
            key: '6VKmoYSL4NHvMLu939Jtl1QUHyLrMGXn7kaqE13NAthqS1vCoet8JQQJ99BEAAAAAAAAAAAAINFRAZML1q7B'
        },
        recommendation: {
            url: 'https://greensort4-lqoyb.southeastasia.inference.ml.azure.com/score',
            key: 'AXGbYl4IRjspRFkOmUYjFAR2aUd3X5J2SNHGaDQVL4xPcnEYX7XaJQQJ99BEAAAAAAAAAAAAINFRAZML4QB2'
        }
    };

    // DOM Elements
    const modelStatus = document.getElementById('modelStatus');
    const imageInput = document.getElementById('imageInput');
    const selectedImage = document.getElementById('selectedImage');
    const noImageLabel = document.getElementById('noImageLabel');
    const classifyBtn = document.getElementById('classifyBtn');
    const quantityInput = document.getElementById('quantityInput');
    const progressBar = document.getElementById('progressBar');
    const predictionsFrame = document.getElementById('predictionsFrame');
    const priceFrame = document.getElementById('priceFrame');
    const recommendationFrame = document.getElementById('recommendationFrame');
    const processedImageModal = document.getElementById('processedImageModal');
    const processedImageContainer = document.getElementById('processedImageContainer');
    const closeModal = document.querySelector('.close-modal');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // State
    let selectedFile = null;
    let isApisReady = false;

    // Initialize
    initializeApp();

    async function initializeApp() {
        console.log('Initializing app...');

        try {
            // Test API connectivity
            await testApiConnectivity();
            isApisReady = true;
            updateModelStatus('API endpoints ready.', 'success');
        } catch (error) {
            console.error('Initialization failed:', error);
            updateModelStatus(`Failed to connect to APIs: ${error.message}`, 'error');
            alert(`Failed to connect to APIs: ${error.message}. Please check your connection or contact the administrator.`);
            classifyBtn.disabled = true;
        }

        // Event listeners
        imageInput.addEventListener('change', handleImageSelection);
        classifyBtn.addEventListener('click', classifyImage);
        closeModal.addEventListener('click', () => {
            processedImageModal.style.display = 'none';
        });
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        });
        quantityInput.addEventListener('input', updateClassifyButtonState);
    }

    async function testApiConnectivity() {
        // Test classification endpoint
        try {
            const response = await fetch(API_ENDPOINTS.classification.url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_ENDPOINTS.classification.key}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({}) // Empty payload to check connectivity
            });
            if (!response.ok) {
                throw new Error(`Classification API error ${response.status}: ${response.statusText}`);
            }
            console.log('Classification API connected successfully.');
        } catch (error) {
            throw new Error(`Classification API failed: ${error.message}`);
        }

        // Test recommendation endpoint
        try {
            const response = await fetch(API_ENDPOINTS.recommendation.url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_ENDPOINTS.recommendation.key}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({}) // Empty payload to check connectivity
            });
            if (!response.ok) {
                throw new Error(`Recommendation API error ${response.status}: ${response.statusText}`);
            }
            console.log('Recommendation API connected successfully.');
        } catch (error) {
            throw new Error(`Recommendation API failed: ${error.message}`);
        }
    }

    function updateModelStatus(text, status) {
        const statusDot = modelStatus.querySelector('.status-dot');
        const statusText = modelStatus.querySelector('.status-text');
        statusText.textContent = text;
        statusDot.className = `status-dot ${status}`;
    }

    function handleImageSelection(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                selectedImage.src = e.target.result;
                selectedImage.classList.remove('hidden');
                noImageLabel.classList.add('hidden');
                updateClassifyButtonState();
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please select a valid image file.');
            resetImageInput();
        }
    }

    function resetImageInput() {
        selectedFile = null;
        selectedImage.src = '';
        selectedImage.classList.add('hidden');
        noImageLabel.classList.remove('hidden');
        imageInput.value = '';
        updateClassifyButtonState();
    }

    function updateClassifyButtonState() {
        const quantity = parseFloat(quantityInput.value);
        classifyBtn.disabled = !isApisReady || !selectedFile || isNaN(quantity) || quantity <= 0;
    }

    function switchTab(tabId) {
        tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabId}Tab`);
        });
    }

    async function classifyImage() {
        if (!selectedFile) {
            alert('Please select an image.');
            return;
        }
        if (!isApisReady) {
            alert('APIs not ready. Please check your connection and try again.');
            return;
        }

        classifyBtn.disabled = true;
        progressBar.style.width = '10%';

        try {
            // Preprocess image
            console.log('Preprocessing image...');
            const base64Image = await preprocessImage(selectedFile);
            progressBar.style.width = '20%';

            // Perform classification
            console.log('Classifying image...');
            const predictions = await classifyWithApi(base64Image);
            progressBar.style.width = '50%';
            const { predictedClass, confidence, topPredictions } = processClassificationResponse(predictions);

            const quantity = parseFloat(quantityInput.value);
            if (isNaN(quantity) || quantity <= 0) {
                throw new Error('Invalid quantity entered');
            }

            // Get recommendations
            console.log('Generating recommendations...');
            const recommendations = await getRecommendation(predictedClass, quantity);
            progressBar.style.width = '90%';

            // Display results
            console.log('Displaying results...');
            displayResults(predictedClass, confidence, topPredictions, quantity, recommendations);
            await showProcessedImage();
            progressBar.style.width = '100%';
        } catch (error) {
            console.error('Error during classification or recommendation:', error);
            alert(`Error: ${error.message}. Please try again or contact the administrator.`);
            progressBar.style.width = '0%';
        } finally {
            classifyBtn.disabled = false;
        }
    }

    async function preprocessImage(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 224;
                canvas.height = 224;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 224, 224);
                // Convert to base64
                const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];
                URL.revokeObjectURL(img.src);
                resolve(base64Image);
            };
            img.onerror = () => reject(new Error('Failed to load image'));
        });
    }

    async function classifyWithApi(base64Image) {
        try {
            const response = await fetch(API_ENDPOINTS.classification.url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_ENDPOINTS.classification.key}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64Image
                })
            });
            if (!response.ok) {
                throw new Error(`Classification API error ${response.status}: ${response.statusText}`);
            }
            const result = await response.json();
            // Expect result to be an array of probabilities
            if (!Array.isArray(result) || result.length !== CATEGORIES.length) {
                throw new Error('Invalid classification response format');
            }
            return result;
        } catch (error) {
            throw new Error(`Classification error: ${error.message}`);
        }
    }

    function processClassificationResponse(probabilities) {
        const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities));
        const predictedClass = CATEGORIES[predictedClassIndex];
        const confidence = probabilities[predictedClassIndex] * 100;

        const topIndices = Array.from(probabilities)
            .map((val, idx) => ({ val, idx }))
            .sort((a, b) => b.val - a.val)
            .slice(0, 3)
            .map(item => item.idx);

        const topPredictions = topIndices.map((index, i) => ({
            className: CATEGORIES[index],
            confidence: probabilities[index] * 100
        }));

        return { predictedClass, confidence, topPredictions };
    }

    async function getRecommendation(inputKategori, inputBeratKg) {
        try {
            const mappedCategory = CATEGORY_MAPPING[inputKategori] || inputKategori;
            console.log(`Processing recommendation for category: ${mappedCategory}, weight: ${inputBeratKg} kg`);

            const response = await fetch(API_ENDPOINTS.recommendation.url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_ENDPOINTS.recommendation.key}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    kategori: mappedCategory,
                    berat_kg: inputBeratKg
                })
            });
            if (!response.ok) {
                throw new Error(`Recommendation API error ${response.status}: ${response.statusText}`);
            }
            const result = await response.json();

            // Expect result to match dataset.json structure
            if (!result.kategori || !result.rekomendasi || !Array.isArray(result.rekomendasi)) {
                throw new Error('Invalid recommendation response format');
            }

            return {
                kategori: result.kategori,
                berat_input_kg: inputBeratKg,
                berat_min_kg: result.berat_min_kg,
                berat_max_kg: result.berat_max_kg,
                rekomendasi: result.rekomendasi,
                message: result.message || (inputBeratKg < result.berat_min_kg || inputBeratKg > result.berat_max_kg
                    ? `Berat sampah Anda (${inputBeratKg} kg) tidak sesuai dengan rekomendasi umum untuk kategori ini (${result.berat_min_kg} kg - ${result.berat_max_kg} kg).`
                    : '')
            };
        } catch (error) {
            console.error('Recommendation error:', error);
            throw new Error(`Failed to generate recommendation: ${error.message}`);
        }
    }

    function displayResults(predictedClass, confidence, topPredictions, quantity, recommendations) {
        predictionsFrame.innerHTML = '';
        priceFrame.classList.remove('hidden');
        recommendationFrame.classList.remove('hidden');

        const predictionHeader = document.createElement('div');
        predictionHeader.className = 'prediction-header';
        predictionHeader.innerHTML = '<h3>Top Prediction:</h3>';
        predictionsFrame.appendChild(predictionHeader);

        const mainPrediction = document.createElement('div');
        mainPrediction.className = 'main-prediction';
        mainPrediction.innerHTML = `
            <div class="prediction-indicator" style="background-color: ${COLOR_INDICATORS[predictedClass] || '#dcdcdc'}"></div>
            <div class="prediction-info">
                <div class="prediction-name">${predictedClass}</div>
                <div class="prediction-confidence">Confidence: ${confidence.toFixed(2)}%</div>
            </div>
        `;
        predictionsFrame.appendChild(mainPrediction);

        if (topPredictions.length > 1) {
            const alternativeTitle = document.createElement('div');
            alternativeTitle.className = 'alternative-title';
            alternativeTitle.textContent = 'Alternative Predictions:';
            predictionsFrame.appendChild(alternativeTitle);

            const alternativePredictions = document.createElement('div');
            alternativePredictions.className = 'alternative-predictions';
            topPredictions.slice(1).forEach(pred => {
                const altPrediction = document.createElement('div');
                altPrediction.className = 'alt-prediction';
                altPrediction.innerHTML = `
                    <div class="alt-left">
                        <div class="alt-indicator" style="background-color: ${COLOR_INDICATORS[pred.className] || '#dcdcdc'}"></div>
                        <span>${pred.className}</span>
                    </div>
                    <span>${pred.confidence.toFixed(2)}%</span>
                `;
                alternativePredictions.appendChild(altPrediction);
            });
            predictionsFrame.appendChild(alternativePredictions);
        }

        const priceContent = document.getElementById('priceContent');
        priceContent.innerHTML = '';

        const pricePerUnit = PRICE_PER_KG[predictedClass] || 0;
        const totalPrice = pricePerUnit * quantity;

        priceContent.innerHTML = `
            <div class="price-row">
                <span>Waste Type:</span>
                <span>${predictedClass}</span>
            </div>
            <div class="price-row">
                <span>Price per kg:</span>
                <span>Rp ${pricePerUnit.toLocaleString()}</span>
            </div>
            <div class="price-row">
                <span>Quantity:</span>
                <span>${quantity.toFixed(2)} kg</span>
            </div>
            <div class="price-separator"></div>
            <div class="price-row">
                <span class="total-price">TOTAL VALUE:</span>
                <span class="total-price">Rp ${totalPrice.toLocaleString()}</span>
            </div>
        `;

        const recommendationContent = document.getElementById('recommendationContent');
        recommendationContent.innerHTML = '';

        if (recommendations.message) {
            recommendationContent.innerHTML += `<p class="recommendation-message">${recommendations.message}</p>`;
        }
        const recList = document.createElement('ul');
        recList.className = 'recommendation-list';
        recommendations.rekomendasi.forEach((rec, index) => {
            const li = document.createElement('li');
            li.textContent = `${index + 1}. ${rec}`;
            recList.appendChild(li);
        });
        recommendationContent.appendChild(recList);
    }

    async function showProcessedImage() {
        if (!selectedImage.src) return;

        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.src = selectedImage.src;
        await new Promise(resolve => {
            img.onload = resolve;
            img.onerror = () => console.error('Failed to load image for modal');
        });

        ctx.drawImage(img, 0, 0, 224, 224);

        processedImageContainer.innerHTML = '';
        processedImageContainer.appendChild(canvas);

        processedImageModal.style.display = 'block';
    }
});