PDC Health Diagnosis System
A parallel and distributed computing based multi‑task health diagnosis system using four body sensors.

📌 Overview
This project implements a multi‑task health diagnosis system that processes readings from four body‑mounted sensors (fever, cough, sneeze, weakness) plus a migraine pain score to simultaneously predict:

Gender (classification)

Age (regression)

Probable Disease (7‑class classification)

The system explicitly demonstrates parallel and distributed computing concepts at multiple levels:

Task parallelism across three independent models using ThreadPoolExecutor

Internal parallelism inside Random Forest models (n_jobs=-1)

Parallel inference for low‑latency multi‑output predictions
Key Features
Feature	Description
Multi‑task diagnosis	Three ML models trained on the same feature vector
Parallel training	Three models trained concurrently using threads
Parallel inference	Gender, age, and disease predictions run simultaneously
Synthetic dataset	20,000 reproducible patient records with realistic sensor patterns
Evaluation metrics	Accuracy, R², MAE, confusion matrices, correlation heatmaps
Extensible design	Ready to scale to distributed engines (Spark, Dask, Kubernetes)
What Runs Concurrently
Phase	Concurrent Tasks
Training	Three fit() operations overlap via ThreadPoolExecutor(max_workers=3)
Inside each model	Multiple decision trees built in parallel (n_jobs=-1)
Inference	Three predict() calls overlap, then results are merged
Sequential baseline: fit gender → fit age → fit disease (no overlap)
Parallel design: all three fits overlap → illustrates true task parallelism

📊 Dataset
The synthetic dataset (health_dataset_20000.csv) contains 20,000 patient records with:

Feature	Range	Description
sensor1_fever	0–5	Forehead temperature proxy
sensor2_cough	0–5	Chest/cough activity
sensor3_sneeze	0–5	Nasal/sneeze activity
sensor4_weakness	0–5	Fatigue level
migraine_pain	0–10	Headache intensity
gender	Male/Female	Demographics
age	1–95 years	Patient age
disease	7 classes	Target label
Disease classes: Typhoid, Pneumonia, Dengue, Flu, Common Cold, Malaria, Only Fever

The generator uses fixed random seed (42) for reproducibility. Small demographic adjustments make gender and age learnable while preserving disease separability.

🚀 Quick Start
Prerequisites
bash
Python 3.10+
pip install pandas numpy scikit-learn joblib matplotlib seaborn
1. Clone & Setup
bash
git clone https://github.com/yourusername/pdc-health-diagnosis.git
cd pdc-health-diagnosis
python -m venv venv_pdc
source venv_pdc/bin/activate   # or .\venv_pdc\Scripts\activate on Windows
pip install -r requirements.txt
2. Generate Dataset
bash
python dataset.py
Creates health_dataset_20000.csv (20,000 rows).

3. Train & Evaluate
bash
python model.py
Outputs:

Console logs with training status and test metrics

Saved models: model_gender.pkl, model_age.pkl, model_disease.pkl

Saved encoder: gender_encoder.pkl

Figures under plot_outputs/ (bar chart, confusion matrices, heatmaps)

Example Prediction
python
from model import predict_parallel

sensor_values = [4.8, 1.1, 0.4, 4.3, 7.2]  # fever, cough, sneeze, weakness, migraine
result = predict_parallel(sensor_values, ...)

# Output:
# {
#   "Gender": "Male",
#   "Predicted Age": 23,
#   "Top Disease": "Typhoid",
#   "Probabilities": {"Typhoid": 62%, "Malaria": 36%, ...}
# }
📈 Model Performance
Task	Model Type	Metric	Typical Score
Gender	RandomForestClassifier	Accuracy	~99%
Disease	RandomForestClassifier	Accuracy	~92%
Age	RandomForestRegressor	R² / MAE	~0.81 / 5.8 years
*Test set size: 20% of data, random_state=42*

Confusion Matrices
The system generates:

Count‑based confusion matrices (gender & disease)

Row‑normalized heatmaps (recall per true class)

Feature correlation heatmap

All figures are saved to plot_outputs/ and embedded in the report.

🔧 Project Structure
text
pdc-health-diagnosis/
├── dataset.py                    # Synthetic data generator (random_state=42)
├── model.py                      # Training, parallel inference, evaluation
├── build_pdc_report.py           # Builds the project report (DOCX)
├── health_dataset_20000.csv      # Generated dataset
├── model_gender.pkl              # Trained gender classifier
├── model_age.pkl                 # Trained age regressor
├── model_disease.pkl             # Trained disease classifier
├── gender_encoder.pkl            # LabelEncoder for gender decoding
├── plot_outputs/                 # All evaluation figures (PNG)
│   ├── metrics_bar_chart.png
│   ├── disease_confusion_matrix.png
│   ├── gender_confusion_matrix.png
│   ├── disease_confusion_heatmap.png
│   ├── gender_confusion_heatmap.png
│   └── correlation_heatmap.png
├── requirements.txt              # Dependencies
└── README.md                     # This file
⚙️ Parallel Implementation Details
Training with ThreadPoolExecutor
python
with ThreadPoolExecutor(max_workers=3) as executor:
    f1 = executor.submit(model_gender.fit, X_train, y_gender)
    f2 = executor.submit(model_age.fit, X_train, y_age)
    f3 = executor.submit(model_disease.fit, X_train, y_disease)
Internal Parallelism in Each Random Forest
python
RandomForestClassifier(n_estimators=100, n_jobs=-1)  # uses all CPU cores
RandomForestRegressor(n_estimators=100, n_jobs=-1)
Parallel Inference
python
def predict_parallel(sensor_values, ...):
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_g = executor.submit(model_gender.predict, X_in)
        fut_a = executor.submit(model_age.predict, X_in)
        fut_d = executor.submit(model_disease.predict_proba, X_in)
    # merge results after all complete
Note on GIL: scikit‑learn offloads Random Forest work to native Cython/C code where the GIL can be released, making threads a practical choice. For pure Python workloads, consider ProcessPoolExecutor.

☁️ Scaling to Distributed Computing
The same architectural patterns extend naturally to production‑grade distributed systems:

Current (Single Node)	Distributed Analogy
ThreadPoolExecutor	Apache Spark / Dask tasks
3 models on 3 threads	3 models on 3 Kubernetes pods
Shared memory X_train	Parquet files / object storage
n_jobs=-1 (cores)	Many Spark executors per model
In‑memory predictions	API gateway + microservices
Possible extensions:

Spark MLlib for distributed Random Forest over millions of rows

Dask for out‑of‑core training when data exceeds RAM

Kafka + Docker for real‑time sensor streaming with parallel inference

Kubernetes with horizontal pod autoscaling per model (separate replicas for gender, age, disease)

📋 Reproducibility Checklist
Python 3.10+ with virtual environment

Fixed random seeds (dataset.py seed=42, random_state=42 in train_test_split and RandomForest)

All dependencies versioned in requirements.txt

Same dataset generation logic for everyone

No external API calls or non‑deterministic operations

bash
# Full reproduction from scratch
python dataset.py          # generates exact same CSV
python model.py            # trains models & produces metrics + figures
🧪 Results Visualization
Figure	Description
metrics_bar_chart.png	Gender accuracy, disease accuracy, R² (×100)
disease_confusion_matrix.png	Raw counts: predicted vs true disease
gender_confusion_matrix.png	Raw counts: predicted vs true gender
disease_confusion_heatmap.png	Row‑normalized (recall per class)
gender_confusion_heatmap.png	Row‑normalized gender recall
correlation_heatmap.png	Sensor & migraine feature correlations
🤝 Contributing
Contributions are welcome! Areas for improvement:

Real sensor data collection + preprocessing pipeline

Additional models (XGBoost, LightGBM) for comparison

Docker + FastAPI deployment for REST inference

Time‑series support (multiple readings per patient)

Uncertainty quantification for disease predictions

HIPAA‑compliant logging and audit trails

📄 License
MIT License — free for academic and commercial use with attribution.

📚 References
Pedregosa et al., Scikit‑learn: Machine Learning in Python, JMLR 2011

Breiman, L., Random Forests, Machine Learning 2001

Python concurrent.futures — task parallelism

Zaharia et al., Apache Spark: A unified engine for big data processing, CACM 2016

Rocklin, M., Dask: Parallelism with analytics, 2015

👥 Authors
Name	Registration No.
Muhammad Haseeb	L1F22BSCS0190
Ghania Tanver	L1F22BSCS0420
