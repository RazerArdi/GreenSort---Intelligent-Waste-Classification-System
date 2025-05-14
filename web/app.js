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

    // Fallback mock recommendation data
    const RECOMMENDATION_DATA = [
        {
            kategori: 'Kardus',
            berat_min_kg: 0.5,
            berat_max_kg: 50,
            rekomendasi: [
                'Recycle at a local cardboard recycling facility.',
                'Ensure the cardboard is clean and free of tape or labels.',
                'Flatten the cardboard to save space.'
            ]
        },
        {
            kategori: 'Bahan Organik Makanan',
            berat_min_kg: 0.1,
            berat_max_kg: 20,
            rekomendasi: [
                'Compost food scraps in a backyard compost bin.',
                'Avoid including meat or dairy to prevent odors.',
                'Contact local composting services for large quantities.'
            ]
        },
        {
            kategori: 'Kaca',
            berat_min_kg: 0.5,
            berat_max_kg: 30,
            rekomendasi: [
                'Clean glass containers before recycling.',
                'Separate by color if required by local facilities.',
                'Check for local glass recycling drop-off points.'
            ]
        },
        {
            kategori: 'Logam',
            berat_min_kg: 0.2,
            berat_max_kg: 40,
            rekomendasi: [
                'Take to a scrap metal recycling center.',
                'Remove any non-metal attachments.',
                'Sort by metal type (e.g., aluminum, steel) if possible.'
            ]
        },
        {
            kategori: 'Sampah Lainnya',
            berat_min_kg: 0.1,
            berat_max_kg: 100,
            rekomendasi: [
                'Dispose of non-recyclable waste responsibly.',
                'Check local waste management guidelines.',
                'Consider upcycling for creative reuse.'
            ]
        },
        {
            kategori: 'Kertas',
            berat_min_kg: 0.3,
            berat_max_kg: 50,
            rekomendasi: [
                'Recycle at a paper recycling facility.',
                'Remove staples or bindings if required.',
                'Keep paper dry and clean for best recycling results.'
            ]
        },
        {
            kategori: 'Plastik',
            berat_min_kg: 0.2,
            berat_max_kg: 30,
            rekomendasi: [
                'Clean plastic containers before recycling.',
                'Check plastic type (e.g., PET, HDPE) for local recycling rules.',
                'Avoid mixing different plastic types.'
            ]
        },
        {
            kategori: 'Sampah Tekstil',
            berat_min_kg: 0.3,
            berat_max_kg: 20,
            rekomendasi: [
                'Donate usable textiles to charity.',
                'Recycle damaged textiles at specialized facilities.',
                'Avoid disposing textiles in regular trash.'
            ]
        },
        {
            kategori: 'Vegetasi',
            berat_min_kg: 0.5,
            berat_max_kg: 50,
            rekomendasi: [
                'Compost yard waste in a compost pile.',
                'Use for mulch if suitable.',
                'Contact local green waste collection services.'
            ]
        }
    ];

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
    let isModelLoaded = false;
    let selectedFile = null;
    let classificationModel = null;
    let recommendationModel = null;

    // Initialize
    initializeApp();

    async function initializeApp() {
        console.log('Initializing app...');
        console.log('TensorFlow.js available:', typeof tf !== 'undefined' ? tf.version.tfjs : 'Not loaded');

        // Load TensorFlow.js models
        try {
            console.log('Loading classification model...');
            classificationModel = await tf.loadLayersModel('../models/ComputerVision/tfjs_model/model.json');
            console.log('Classification model loaded');

            console.log('Loading recommendation model...');
            recommendationModel = await tf.loadLayersModel('../models/ComputerVision/recommendation/model.json');
            console.log('Recommendation model loaded');

            isModelLoaded = true;
            updateModelStatus('Models loaded successfully', 'success');
        } catch (error) {
            console.error('Model loading failed:', error);
            updateModelStatus('Failed to load models', 'error');
            const demoWarning = document.createElement('div');
            demoWarning.className = 'demo-warning';
            demoWarning.innerHTML = '<strong>Demo Mode:</strong> Running with simulated models. Classifications and recommendations are for demonstration only.';
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
            selectedFile = null;
            selectedImage.classList.add('hidden');
            noImageLabel.classList.remove('hidden');
            updateClassifyButtonState();
        }
    }

    function updateClassifyButtonState() {
        const quantity = parseFloat(quantityInput.value);
        classifyBtn.disabled = !isModelLoaded || !selectedFile || isNaN(quantity) || quantity <= 0;
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
        if (!selectedFile || !isModelLoaded) {
            alert('Please select an image and ensure models are loaded.');
            return;
        }

        classifyBtn.disabled = true;
        progressBar.style.width = '10%';

        try {
            // Process image
            await simulateDelay(500);
            progressBar.style.width = '20%';

            // Preprocess image for classification
            const img = new Image();
            img.src = selectedImage.src;
            await new Promise(resolve => { img.onload = resolve; });
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();

            // Classification
            let predictedClass, confidence, topPredictions;
            if (classificationModel) {
                progressBar.style.width = '50%';
                const prediction = await classificationModel.predict(tensor).data();
                const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
                predictedClass = CATEGORIES[predictedClassIndex];
                confidence = prediction[predictedClassIndex] * 100;

                // Get top 3 predictions
                const topIndices = Array.from(prediction)
                    .map((val, idx) => ({ val, idx }))
                    .sort((a, b) => b.val - a.val)
                    .slice(0, 3)
                    .map(item => item.idx);
                topPredictions = topIndices.map((index, i) => ({
                    className: CATEGORIES[index],
                    confidence: prediction[index] * 100
                }));
            } else {
                // Fallback to mock classification
                ({ predictedClass, confidence, topPredictions } = mockClassification());
            }

            progressBar.style.width = '70%';

            const quantity = parseFloat(quantityInput.value);
            if (isNaN(quantity) || quantity <= 0) {
                alert('Please enter a valid quantity.');
                return;
            }

            // Get recommendations
            const recommendations = await getRecommendation(predictedClass, quantity);
            progressBar.style.width = '90%';

            // Display results
            displayResults(predictedClass, confidence, topPredictions, quantity, recommendations);
            progressBar.style.width = '100%';

            // Show processed image
            await showProcessedImage();
        } catch (error) {
            console.error('Classification error:', error);
            alert('An error occurred during classification.');
        } finally {
            progressBar.style.width = '0%';
            classifyBtn.disabled = false;
        }
    }

    function simulateDelay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
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

    async function getRecommendation(inputKategori, inputBeratKg, tolerance = 0.2) {
        try {
            const mappedCategory = CATEGORY_MAPPING[inputKategori] || inputKategori;
            if (recommendationModel) {
                const categoryIndex = CATEGORIES.indexOf(inputKategori);
                const inputTensor = tf.tensor2d([[categoryIndex, inputBeratKg]]);
                
                const prediction = await recommendationModel.predict(inputTensor).data();
                const [berat_min_kg, berat_max_kg, ...recIndices] = prediction;
                
                const rekomendasi = recIndices
                    .slice(0, 3)
                    .map(idx => `Recommendation ${Math.round(idx)}: Follow local recycling guidelines.`);
                
                const result = {
                    kategori: mappedCategory,
                    berat_input_kg: inputBeratKg,
                    berat_min_kg,
                    berat_max_kg,
                    rekomendasi
                };

                if (inputBeratKg < berat_min_kg || inputBeratKg > berat_max_kg) {
                    result.message = `Berat sampah Anda (${inputBeratKg} kg) sedikit tidak sesuai dengan rekomendasi untuk kategori ini (${berat_min_kg} kg - ${berat_max_kg} kg). Namun, berikut adalah beberapa rekomendasi yang bisa diterapkan:`;
                }

                return result;
            } else {
                const matchingRows = RECOMMENDATION_DATA.filter(row => row.kategori === mappedCategory);
                if (!matchingRows.length) {
                    return { message: `Kategori '${mappedCategory}' tidak ditemukan dalam dataset.` };
                }

                const weightMatch = matchingRows.find(row =>
                    row.berat_min_kg <= inputBeratKg && inputBeratKg <= row.berat_max_kg
                );

                const bestMatch = weightMatch || matchingRows[0];
                let message = '';

                if (!weightMatch) {
                    message = `Berat sampah Anda (${inputBeratKg} kg) sedikit tidak sesuai dengan rekomendasi untuk kategori ini (${bestMatch.berat_min_kg} kg - ${bestMatch.berat_max_kg} kg). Namun, berikut adalah beberapa rekomendasi yang bisa diterapkan:`;
                }

                return {
                    kategori: bestMatch.kategori,
                    berat_input_kg: inputBeratKg,
                    berat_min_kg: bestMatch.berat_min_kg,
                    berat_max_kg: bestMatch.berat_max_kg,
                    message: message,
                    rekomendasi: bestMatch.rekomendasi
                };
            }
        } catch (error) {
            console.error('Recommendation error:', error);
            return { message: `Error generating recommendation: ${error.message}` };
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

        if (recommendations.message && !recommendations.rekomendasi) {
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