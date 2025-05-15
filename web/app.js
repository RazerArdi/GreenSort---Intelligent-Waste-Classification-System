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
        classification: 'https://greensort-jhfbo.eastus2.inference.ml.azure.com/score',
        recommendation: 'upcoming'
    };

    const API_KEYS = {
        primary: 'Fovu7tGif73gqRtHd2iaYocpHkV0USfI5hVmqLNRYF2OtsPbbJGvJQQJ99BEAAAAAAAAAAAAINFRAZML1ISi',
        secondary: '8aFikV6o8w54LHxJCHhmN0VZoVobC90S73iK9sIXnEkwdSDMhQZyJQQJ99BEAAAAAAAAAAAAINFRAZML3Wqt'
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
    let isApiReady = false;
    let selectedFile = null;

    // Initialize
    initializeApp();

    async function initializeApp() {
        console.log('Initializing app...');

        // Check classification API availability
        try {
            await checkApiStatus(API_ENDPOINTS.classification);
            isApiReady = true;
            updateModelStatus('Classification API ready. Recommendation system upcoming.', 'success');
        } catch (error) {
            console.error('API initialization failed:', error);
            let errorMessage = 'Failed to connect to classification API.';
            if (error.message.includes('403')) {
                errorMessage += ' Authentication failed with both keys. Verify API keys.';
            } else if (error.message.includes('CORS') || error.message.includes('NetworkError')) {
                errorMessage += ' CORS issue detected. The server must allow requests from http://127.0.0.1:5500. Consider using a proxy for development.';
            }
            updateModelStatus(errorMessage, 'error');
            const demoWarning = document.createElement('div');
            demoWarning.className = 'demo-warning';
            demoWarning.innerHTML = `<strong>Demo Mode:</strong> API unavailable. ${errorMessage} Contact API administrator or set up a proxy server.`;
            document.querySelector('.left-panel').prepend(demoWarning);
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

    async function checkApiStatus(endpoint) {
        try {
            // Try primary key
            let response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEYS.primary}`,
                    'azureml-model-deployment': 'greensort-1'
                },
                body: JSON.stringify({ ping: true })
            });
            if (response.ok) return true;

            // Log headers for debugging
            console.debug('Check API Status Headers:', [...response.headers]);

            // If primary key fails with 403, try secondary key
            if (response.status === 403) {
                console.warn('Primary key failed, trying secondary key...');
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${API_KEYS.secondary}`,
                        'azureml-model-deployment': 'greensort-1'
                    },
                    body: JSON.stringify({ ping: true })
                });
                if (response.ok) return true;
                console.debug('Secondary Key Headers:', [...response.headers]);
            }
            throw new Error(`API responded with status ${response.status}`);
        } catch (error) {
            if (error.message.includes('Failed to fetch') || error.name === 'TypeError') {
                throw new Error('Classification API unavailable: NetworkError due to CORS or network issue');
            }
            throw new Error(`Classification API unavailable: ${error.message}`);
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
        classifyBtn.disabled = !selectedFile || isNaN(quantity) || quantity <= 0;
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

        classifyBtn.disabled = true;
        progressBar.style.width = '10%';

        try {
            // Convert image to base64
            const base64Image = await fileToBase64(selectedFile);
            progressBar.style.width = '20%';

            // Call classification API
            let classificationResponse;
            if (isApiReady) {
                classificationResponse = await callClassificationApi(base64Image);
            } else {
                throw new Error('Classification API unavailable');
            }
            progressBar.style.width = '50%';

            const { predictedClass, confidence, topPredictions } = processClassificationResponse(classificationResponse);
            const quantity = parseFloat(quantityInput.value);

            if (isNaN(quantity) || quantity <= 0) {
                throw new Error('Invalid quantity entered');
            }

            // Get recommendations (mock data since endpoint is upcoming)
            const recommendations = await getRecommendation(predictedClass, quantity);
            progressBar.style.width = '90%';

            // Display results
            displayResults(predictedClass, confidence, topPredictions, quantity, recommendations);
            await showProcessedImage();
            progressBar.style.width = '100%';
        } catch (error) {
            console.error('Classification error:', error);
            let errorMessage = `Error during classification: ${error.message}.`;
            if (error.message.includes('403')) {
                errorMessage += ' Authentication failed with both keys. Verify API keys.';
            } else if (error.message.includes('CORS') || error.message.includes('NetworkError')) {
                errorMessage += ' CORS issue persists. The server must allow requests from http://127.0.0.1:5500. Try using a proxy server.';
            }
            alert(`${errorMessage} Using mock data.`);
            const { predictedClass, confidence, topPredictions } = mockClassification();
            const quantity = parseFloat(quantityInput.value) || 1;
            const recommendations = await getRecommendation(predictedClass, quantity);
            displayResults(predictedClass, confidence, topPredictions, quantity, recommendations);
            await showProcessedImage();
        } finally {
            progressBar.style.width = '0%';
            classifyBtn.disabled = false;
        }
    }

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    async function callClassificationApi(base64Image) {
        try {
            // Try primary key
            let response = await fetch(API_ENDPOINTS.classification, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEYS.primary}`,
                    'azureml-model-deployment': 'greensort-1'
                },
                body: JSON.stringify({
                    image: base64Image
                })
            });

            if (response.ok) {
                return await response.json();
            }

            // Log headers for debugging
            console.debug('Primary Key Headers:', [...response.headers]);

            // If primary key fails with 403, try secondary key
            if (response.status === 403) {
                console.warn('Primary key failed, trying secondary key...');
                response = await fetch(API_ENDPOINTS.classification, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${API_KEYS.secondary}`,
                        'azureml-model-deployment': 'greensort-1'
                    },
                    body: JSON.stringify({
                        image: base64Image
                    })
                });
                if (response.ok) {
                    return await response.json();
                }
                console.debug('Secondary Key Headers:', [...response.headers]);
            }

            throw new Error(`Classification API error: ${response.statusText} (${response.status})`);
        } catch (error) {
            if (error.message.includes('Failed to fetch') || error.name === 'TypeError') {
                throw new Error('Failed to call classification API: NetworkError due to CORS or network issue');
            }
            throw new Error(`Failed to call classification API: ${error.message}`);
        }
    }

    function processClassificationResponse(response) {
        // Assuming the API returns probabilities for each category
        const probabilities = response.probabilities || response;
        
        // Find the predicted class
        const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities));
        const predictedClass = CATEGORIES[predictedClassIndex];
        const confidence = probabilities[predictedClassIndex] * 100;

        // Get top 3 predictions
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

    function mockClassification() {
        const randomIndex = Math.floor(Math.random() * CATEGORIES.length);
        const predictedClass = CATEGORIES[randomIndex];
        const confidence = 75 + Math.random() * 20;

        const topIndices = [randomIndex];
        while (topIndices.length < 3) {
            const newIndex = Math.floor(Math.random() * CATEGORIES.length);
            if (!topIndices.includes(newIndex)) {
                topIndices.push(newIndex);
            }
        }

        const topPredictions = topIndices.map((index, i) => ({
            className: CATEGORIES[index],
            confidence: i === 0 ? confidence : (confidence - (i * 15))
        }));

        return { predictedClass, confidence, topPredictions };
    }

    async function getRecommendation(inputKategori, inputBeratKg) {
        try {
            const mappedCategory = CATEGORY_MAPPING[inputKategori] || inputKategori;

            // Mock recommendations since endpoint is upcoming
            const mockRecommendations = [
                `Sort ${mappedCategory} according to local recycling guidelines.`,
                `Ensure ${mappedCategory} is clean and free of contaminants before disposal.`,
                `Consider donating usable ${mappedCategory} items to local charities.`
            ];

            const berat_min_kg = inputBeratKg * 0.8; // Mock min weight
            const berat_max_kg = inputBeratKg * 1.2; // Mock max weight

            const result = {
                kategori: mappedCategory,
                berat_input_kg: inputBeratKg,
                berat_min_kg,
                berat_max_kg,
                rekomendasi: mockRecommendations,
                message: 'Recommendation system is upcoming. Displaying general recycling guidelines.'
            };

            if (inputBeratKg < berat_min_kg || inputBeratKg > berat_max_kg) {
                result.message = `Berat sampah Anda (${inputBeratKg} kg) tidak sesuai dengan rekomendasi umum untuk kategori ini (${berat_min_kg} kg - ${berat_max_kg} kg). Berikut adalah panduan umum:`;
            }

            return result;
        } catch (error) {
            console.error('Recommendation error:', error);
            return {
                kategori: CATEGORY_MAPPING[inputKategori] || inputKategori,
                berat_input_kg: inputBeratKg,
                message: `Error generating recommendation: ${error.message}`,
                rekomendasi: ['Follow local recycling guidelines']
            };
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

        if (recommendations.message && !recommendations.rekomendasi.length) {
            recommendationContent.innerHTML = `<p class="recommendation-message">${recommendations.message}</p>`;
        } else {
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
        });

        ctx.drawImage(img, 0, 0, 224, 224);

        processedImageContainer.innerHTML = '';
        processedImageContainer.appendChild(canvas);

        processedImageModal.style.display = 'block';
    }
});