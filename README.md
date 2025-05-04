# HandDigit: Interactive Handwritten Digit Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

An interactive web application for handwritten digit recognition using Convolutional Neural Networks (CNN). This project demonstrates the power of deep learning in recognizing handwritten digits with high accuracy.

## 🌟 Features

- **Interactive Drawing Board**: Draw digits directly in your browser
- **Image Upload**: Upload images of handwritten digits
- **Real-time Predictions**: Get instant predictions with confidence scores
- **Probability Distribution**: View the probability distribution across all digits
- **Model Performance Metrics**: Detailed model architecture and performance statistics

## 🚀 Demo

![Demo GIF](static/demo.gif)

## 📊 Model Performance

- **Architecture**: Convolutional Neural Network (CNN)
- **Training Dataset**: MNIST
- **Test Accuracy**: 98.89%
- **Test Loss**: 0.0439

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-cnn.git
   cd mnist-cnn
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model (optional - pre-trained model included):
   ```bash
   python src/train_model.py
   ```

5. Run the application:
   ```bash
   streamlit run src/app.py
   ```

## 🏗️ Project Structure

```
mnist-cnn/
├── models/              # Saved model files
├── src/                 # Source code
│   ├── app.py          # Streamlit application
│   └── train_model.py  # Model training script
├── static/             # Static files (images, etc.)
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## 🧠 Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320       
conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
max_pooling2d (MaxPooling2D)(None, 12, 12, 64)        0         
dropout (Dropout)           (None, 12, 12, 64)        0         
flatten (Flatten)           (None, 9216)              0         
dense (Dense)               (None, 256)               2359552   
dropout_1 (Dropout)         (None, 256)               0         
dense_1 (Dense)             (None, 10)                2570      
=================================================================
Total params: 2,380,938
```

## 📈 Training History

- **Epochs**: 10
- **Validation Split**: 0.3
- **Final Training Accuracy**: 98.97%
- **Final Validation Accuracy**: 98.67%

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/mnist-cnn](https://github.com/yourusername/mnist-cnn)