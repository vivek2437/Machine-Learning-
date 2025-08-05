# 🧠 Machine Learning Practice Repository

This repository contains comprehensive hands-on practice with **Machine Learning** concepts, algorithms, and real-world projects implemented using Python and Jupyter Notebooks.

## 📋 Repository Overview

This repository demonstrates practical implementation of various machine learning algorithms and includes both educational notebooks and deployable applications. It covers supervised learning techniques with detailed implementations and a complete end-to-end project with web deployment.

## 🚀 Topics Covered

### Supervised Learning
- **Classification Algorithms:**
  - 🌳 Decision Trees
  - 🔍 K-Nearest Neighbors (KNN)
  - 📈 Logistic Regression
  - 🤖 Support Vector Machines (SVM)
  - 🎯 Naive Bayes

- **Regression Algorithms:**
  - 📊 Linear Regression
  - 🔧 Ridge Regression
  - 📈 Multiple Regression Models

### Real-World Projects
- 🚢 **Titanic Dataset Analysis** - Complete data science pipeline for survival prediction
- 🏠 **Regression Web Application** - Flask-based web app for house price prediction

## 📁 Project Structure

```
Machine-Learning-/
├── README.md
├── LICENSE
└── Supervised Learning/
    ├── Classification/
    │   ├── Decision_Tree.ipynb
    │   ├── KNN.ipynb
    │   ├── Logistic_Regression.ipynb
    │   ├── SVM.ipynb
    │   └── navie_bayes.ipynb
    ├── Regression/
    │   ├── Regression_Model.ipynb
    │   ├── application.py          # Flask web application
    │   ├── home.html              # Web interface
    │   ├── index.html             # Landing page
    │   ├── requirments.txt        # Dependencies
    │   ├── ridge.pkl              # Trained model
    │   └── scaler.pkl             # Feature scaler
    └── Titanic_Dataset_Project.ipynb
```

## 📦 Libraries Used

- **Core ML Libraries:**
  - `NumPy` - Numerical computing
  - `Pandas` - Data manipulation and analysis
  - `Scikit-learn` - Machine learning algorithms
  
- **Visualization:**
  - `Matplotlib` - Basic plotting
  - `Seaborn` - Statistical data visualization
  
- **Web Development:**
  - `Flask` - Web framework for model deployment
  
- **Development Environment:**
  - `Jupyter Notebook` - Interactive development

## 🧪 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/vivek2437/Machine-Learning-.git
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3. For Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to the desired notebook and run the cells.

### 4. For the Web Application
```bash
cd "Supervised Learning/Regression"
pip install -r requirments.txt
python application.py
```
Then open your browser and go to `http://localhost:5000`

## 🎯 Key Features

### Educational Notebooks
- **Step-by-step implementations** of ML algorithms from scratch
- **Detailed explanations** and mathematical intuitions
- **Practical examples** with real datasets
- **Performance evaluation** and model comparison

### Production-Ready Application
- **Flask web interface** for regression predictions
- **Pre-trained models** saved as pickle files
- **Feature scaling** pipeline included
- **User-friendly web form** for input

### Comprehensive Coverage
- **Multiple algorithms** across classification and regression
- **Famous datasets** like Titanic for practical learning
- **End-to-end pipeline** from data preprocessing to deployment

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements or additional ML implementations.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vivek Rameshbhai Nayi**

---

*Happy Learning! 🚀*
