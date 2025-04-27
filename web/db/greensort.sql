-- Table: waste_categories
CREATE TABLE waste_categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(50) NOT NULL UNIQUE,
    base_market_price DECIMAL(10,2) NOT NULL
);

-- Table: users (optional)
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: classifications
CREATE TABLE classifications (
    classification_id INT PRIMARY KEY AUTO_INCREMENT,
    image_path VARCHAR(255) NOT NULL,
    category_id INT NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    classification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (category_id) REFERENCES waste_categories(category_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);

-- Table: price_predictions
CREATE TABLE price_predictions (
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    classification_id INT NOT NULL,
    weight_kg DECIMAL(10,2) NOT NULL,
    delivery_distance_km DECIMAL(10,2) NOT NULL,
    market_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (classification_id) REFERENCES classifications(classification_id)
);
